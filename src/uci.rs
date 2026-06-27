//! UCI protocol loop.
//!
//! A strong engine is **headless**: it never draws a board. Instead it speaks
//! UCI (Universal Chess Interface) over stdin/stdout, and a separate GUI or
//! tournament manager (Cute Chess, in this project) drives it — sending
//! positions and `go`, reading back `bestmove`. This module is that mouth and
//! ear; everything else (search, eval) hangs off the `go` handler.
//!
//! The protocol is line-oriented and almost stateless: the only state we keep
//! between commands is the current [`Board`]. We implement the subset a
//! tournament needs — `uci`, `isready`, `ucinewgame`, `position`, `go`, `stop`,
//! `quit` — and silently ignore anything else, as the spec requires (an engine
//! must not choke on commands it doesn't understand).
//!
//! ## Why the generic reader/writer
//!
//! The real loop reads `io::stdin()` and writes `io::stdout()`, but the protocol
//! logic is parameterised over [`BufRead`]/[`Write`] so tests can feed canned
//! input from a byte slice and capture output into a `Vec<u8>` — no subprocess,
//! no pipes. See the tests at the bottom.
//!
//! ## The one rule that silently breaks UCI engines: flush
//!
//! Cute Chess talks to us over a *pipe*, not a terminal. Pipe output is block-
//! buffered, so a `println!`-equivalent that isn't flushed sits in a buffer and
//! never reaches the GUI, which then hangs forever waiting for `uciok` /
//! `readyok` / `bestmove`. We therefore flush after every response.

use std::io::{self, BufRead, Write};
use std::str::FromStr;

use crate::board::Board;
use crate::movegen::generate_legal;
use crate::moves::Move;
use crate::search::search;
use crate::types::{PieceType, Square};

/// Default search depth for a bare `go` (no `depth` argument). Time management
/// (issue #21) replaces this fixed depth with a real, clock-driven budget.
const DEFAULT_DEPTH: u32 = 5;

/// The standard starting position, in FEN. Shared shape with `main.rs`'s
/// `STARTPOS`; duplicated rather than cross-referenced to keep the modules
/// independent.
const STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

const ENGINE_NAME: &str = "chess-engine-nnue";
const ENGINE_AUTHOR: &str = "Ahmad Luqman";

/// Run the UCI loop against real stdin/stdout. The binary entry point calls this.
pub fn run() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    // Lock once for the whole session: we are the only user of these streams.
    run_loop(stdin.lock(), &mut stdout.lock())
}

/// The protocol loop proper, generic over its input and output so it can be
/// driven from a test harness as well as from real stdio.
///
/// Returns when it sees `quit` or the input reaches EOF (the GUI closed the
/// pipe) — both are clean shutdowns, not errors.
pub fn run_loop<R: BufRead, W: Write>(input: R, output: &mut W) -> io::Result<()> {
    // The single piece of cross-command state. Starts at the standard position
    // so a bare `go` before any `position` still has something legal to play.
    let mut board = startpos();

    for line in input.lines() {
        let line = line?;
        let mut tokens = line.split_whitespace();
        let Some(command) = tokens.next() else {
            continue; // blank line — ignore.
        };

        match command {
            "uci" => {
                writeln!(output, "id name {ENGINE_NAME}")?;
                writeln!(output, "id author {ENGINE_AUTHOR}")?;
                // No `option` lines yet; we expose no configurable parameters.
                writeln!(output, "uciok")?;
            }
            "isready" => writeln!(output, "readyok")?,
            "ucinewgame" => board = startpos(),
            "position" => set_position(&mut board, tokens),
            "go" => {
                // Search to a fixed depth (issue #19). The only `go` argument we
                // read so far is `depth N`; the clock arguments (wtime/btime/…)
                // become a real time budget in issue #21.
                let depth = parse_go_depth(tokens).unwrap_or(DEFAULT_DEPTH);
                let result = search(&board, depth);
                writeln!(
                    output,
                    "info depth {} score cp {} nodes {}",
                    result.depth, result.score, result.nodes
                )?;
                if result.best_move == Move::NONE {
                    // UCI's "no move" sentinel — mate/stalemate; keeps the GUI
                    // from hanging.
                    writeln!(output, "bestmove 0000")?;
                } else {
                    writeln!(output, "bestmove {}", result.best_move)?;
                }
            }
            // `stop` halts a running search. Our search is instant, so there is
            // nothing to halt yet — issue #21 makes this meaningful once search
            // runs on its own thread with a stop flag.
            "stop" => {}
            "quit" => break,
            // Unknown command: the spec says ignore it, do not error.
            _ => {}
        }

        // See the module docs: without this the GUI never sees our reply.
        output.flush()?;
    }

    Ok(())
}

/// A fresh board at the standard starting position. `unwrap` is safe: the
/// constant FEN is valid by construction and covered by a test below.
fn startpos() -> Board {
    Board::from_str(STARTPOS_FEN).expect("STARTPOS_FEN is a valid FEN")
}

/// Apply a `position` command's arguments to `board`.
///
/// Grammar: `position (startpos | fen <6 fields>) [moves <m1> <m2> ...]`.
/// We rebuild the base position, then play each listed move in order. A
/// malformed command leaves the board at the base position rather than
/// panicking — the next `position`/`ucinewgame` will reset it anyway.
fn set_position<'a, I: Iterator<Item = &'a str>>(board: &mut Board, mut tokens: I) {
    let base = match tokens.next() {
        Some("startpos") => startpos(),
        Some("fen") => {
            // A FEN is six space-separated fields, so it spans multiple tokens.
            // Collect everything up to an optional `moves` keyword and rejoin it
            // before handing the whole string to the parser.
            let mut fen_fields = Vec::new();
            for tok in tokens.by_ref() {
                if tok == "moves" {
                    break;
                }
                fen_fields.push(tok);
            }
            match Board::from_str(&fen_fields.join(" ")) {
                Ok(b) => {
                    *board = b;
                    apply_moves(board, tokens);
                    return;
                }
                Err(_) => return, // unparseable FEN: ignore the command.
            }
        }
        _ => return, // missing/unknown sub-command: ignore.
    };
    *board = base;

    // For `startpos`, the remaining tokens are `[moves ...]`; skip the keyword.
    if let Some(kw) = tokens.next() {
        if kw == "moves" {
            apply_moves(board, tokens);
        }
    }
}

/// Play a sequence of UCI moves onto `board`, stopping at the first one that
/// isn't legal in the position it reaches (a malformed stream shouldn't corrupt
/// state). Each token is matched against the legal move list so the packed
/// move's flags — capture, castle, en-passant, promotion — come straight from
/// the generator instead of being re-derived here.
fn apply_moves<'a, I: Iterator<Item = &'a str>>(board: &mut Board, moves: I) {
    for tok in moves {
        match parse_uci_move(board, tok) {
            Some(mv) => {
                board.make_move(mv);
            }
            None => break,
        }
    }
}

/// Resolve a UCI move string (`e2e4`, `e1g1`, `e7e8q`) to the matching legal
/// [`Move`] in the current position, or `None` if it doesn't name one.
///
/// The robust trick: don't decode the string into flags ourselves. Generate the
/// legal moves and find the one whose `from`/`to` match — and, for the 4-char
/// promotions, whose promoted piece matches the trailing letter. Whether the
/// move is a capture, castle, or en-passant is then already encoded correctly.
fn parse_uci_move(board: &Board, s: &str) -> Option<Move> {
    // A UCI move is 4 chars (from+to) or 5 (… + promotion piece).
    if s.len() != 4 && s.len() != 5 {
        return None;
    }
    let from = Square::from_str(&s[0..2]).ok()?;
    let to = Square::from_str(&s[2..4]).ok()?;
    let promo = s.as_bytes().get(4).copied();

    generate_legal(board).into_iter().find(|mv| {
        if mv.from() != from || mv.to() != to {
            return false;
        }
        match (promo, mv.promotion_piece()) {
            // Promotion letter present: it must name this move's promoted piece.
            (Some(letter), Some(pt)) => promo_letter(pt) == Some(letter),
            // No letter, no promotion: a plain move matches.
            (None, None) => true,
            // Letter without promotion, or promotion without a letter: no match.
            _ => false,
        }
    })
}

/// The lowercase UCI letter for a promotion target, or `None` for pieces that
/// are never promotion targets (pawn, king).
fn promo_letter(pt: PieceType) -> Option<u8> {
    match pt {
        PieceType::Knight => Some(b'n'),
        PieceType::Bishop => Some(b'b'),
        PieceType::Rook => Some(b'r'),
        PieceType::Queen => Some(b'q'),
        PieceType::Pawn | PieceType::King => None,
    }
}

/// Scan a `go` command's arguments for `depth N` and return `N`, if present.
/// Other arguments (clock times, `infinite`, …) are ignored until issue #21.
fn parse_go_depth<'a, I: Iterator<Item = &'a str>>(mut tokens: I) -> Option<u32> {
    while let Some(tok) = tokens.next() {
        if tok == "depth" {
            return tokens.next().and_then(|d| d.parse::<u32>().ok());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive `run_loop` with a canned script and return everything it wrote.
    fn run(script: &str) -> String {
        let mut out = Vec::new();
        run_loop(script.as_bytes(), &mut out).expect("run_loop should not error on a byte slice");
        String::from_utf8(out).expect("output is valid UTF-8")
    }

    #[test]
    fn uci_handshake_identifies_and_acks() {
        let out = run("uci\nquit\n");
        assert!(out.contains("id name "), "missing id name: {out:?}");
        assert!(out.contains("id author "), "missing id author: {out:?}");
        assert!(out.contains("uciok"), "missing uciok: {out:?}");
    }

    #[test]
    fn isready_acks() {
        assert!(run("isready\nquit\n").contains("readyok"));
    }

    #[test]
    fn go_from_startpos_plays_a_legal_move() {
        let out = run("position startpos\ngo depth 2\nquit\n");
        let mv = bestmove(&out).expect("a bestmove line");
        let legal: Vec<String> = generate_legal(&startpos()).iter().map(|m| m.to_string()).collect();
        assert!(legal.contains(&mv), "{mv} is not legal from startpos; legal = {legal:?}");
    }

    #[test]
    fn position_applies_moves_then_go_is_legal_in_that_line() {
        let out = run("position startpos moves e2e4 e7e5\ngo depth 2\nquit\n");
        let mv = bestmove(&out).expect("a bestmove line");

        // Reconstruct the same position and confirm the move is legal there.
        let mut board = startpos();
        for m in ["e2e4", "e7e5"] {
            let parsed = parse_uci_move(&board, m).expect("known legal move");
            board.make_move(parsed);
        }
        let legal: Vec<String> = generate_legal(&board).iter().map(|m| m.to_string()).collect();
        assert!(legal.contains(&mv), "{mv} not legal after 1.e4 e5; legal = {legal:?}");
    }

    #[test]
    fn position_fen_with_spaces_is_reassembled() {
        // A FEN spans six space-separated fields; the parser must rejoin them
        // before parsing. We just assert we get some legal move back.
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let out = run(&format!("position fen {fen}\ngo depth 2\nquit\n"));
        assert!(bestmove(&out).is_some(), "no bestmove for a fen position: {out:?}");
    }

    #[test]
    fn promotion_move_is_parsed_with_correct_flag() {
        // White pawn on a7, black king out of the way: a7a8q must resolve to a
        // queen-promotion move and be playable.
        let fen = "8/P7/8/8/8/8/8/k6K w - - 0 1";
        let mut board = Board::from_str(fen).unwrap();
        let mv = parse_uci_move(&board, "a7a8q").expect("a7a8q should parse");
        assert_eq!(mv.promotion_piece(), Some(PieceType::Queen));
        // And the underqueen variants resolve to distinct pieces.
        assert_eq!(
            parse_uci_move(&board, "a7a8n").unwrap().promotion_piece(),
            Some(PieceType::Knight)
        );
        board.make_move(mv); // must not panic.
    }

    #[test]
    fn unknown_and_blank_lines_are_ignored() {
        // Garbage in the middle must neither panic nor suppress later commands.
        let out = run("frobnicate\n\nxyzzy 1 2 3\nisready\nquit\n");
        assert!(out.contains("readyok"), "later commands stopped working: {out:?}");
    }

    #[test]
    fn eof_without_quit_exits_cleanly() {
        // No `quit`, just EOF — should return Ok and have answered the command.
        let out = run("isready\n");
        assert!(out.contains("readyok"));
    }

    #[test]
    fn no_legal_moves_emits_null_move() {
        // Fool's-mate position: Black is checkmated, so `go` has no move to make.
        let fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3";
        let out = run(&format!("position fen {fen}\ngo depth 2\nquit\n"));
        assert_eq!(bestmove(&out).as_deref(), Some("0000"), "expected null move: {out:?}");
    }

    /// Extract the move from the single `bestmove <m>` line, if present.
    fn bestmove(out: &str) -> Option<String> {
        out.lines()
            .find_map(|l| l.strip_prefix("bestmove "))
            .map(|m| m.trim().to_string())
    }
}
