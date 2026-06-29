//! gen_data — generate a labelled, quiet-filtered position dataset for Texel
//! tuning (issue #42).
//!
//! Plays seeded self-play games and emits one `FEN RESULT` line per **quiet**
//! position, where `RESULT ∈ {1.0, 0.5, 0.0}` is the game's outcome **from
//! White's point of view** (win / draw / loss). The tuner ([`examples/texel.rs`])
//! fits the eval weights so `sigmoid(K · eval_white)` predicts that label.
//!
//! Each game starts from the standard position, plays a few **random** opening
//! plies (a seeded LCG — reproducible, and far more diverse than the 24-line book)
//! to spread the data across openings, then self-plays the rest with a shallow
//! fixed-depth search. Positions are recorded only after the random opening, only
//! when [`engine::search::is_quiet`] holds (qsearch == static eval, side not in
//! check), so the static eval the tuner optimises is meaningful. Games end on
//! mate/stalemate, the fifty-move rule, threefold repetition, a decisive-score
//! resign adjudication, or a ply cap.
//!
//! Run (the dataset is large — gitignored; commit the *tuned constants*, not this):
//!
//! ```text
//! cargo run --release --example gen_data -- <target_positions> <depth> <seed> > data/texel.txt
//! ```
//!
//! Defaults: 100000 positions, depth 6, seed 1. Reproducible: same args → same
//! file (single-threaded, seeded; no wall-clock or RNG-crate dependence).

use std::collections::HashMap;
use std::env;
use std::io::{self, BufWriter, Write};
use std::str::FromStr;

use engine::board::Board;
use engine::movegen::{generate_legal, in_check};
use engine::perft::STARTPOS;
use engine::search::{is_quiet, search_with_tt, MATE};
use engine::tt::TranspositionTable;
use engine::types::Color;

/// Random opening plies per game before recording / search self-play begins.
const RANDOM_PLIES: usize = 8;
/// Hard cap on game length (plies) — reached games are scored a draw.
const MAX_PLIES: usize = 300;
/// Resign adjudication: if one side is ahead by at least this many centipawns for
/// [`RESIGN_PLIES`] consecutive plies, the game is awarded to it.
const RESIGN_CP: i32 = 1000;
const RESIGN_PLIES: u32 = 6;

/// A tiny seeded LCG (PCG-style multiplier) — deterministic, no `rand` crate.
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Return a well-mixed value (xorshift on the high bits).
        let x = self.0;
        (x ^ (x >> 29)).wrapping_mul(0xbf58476d1ce4e5b9)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// The outcome of a self-play game, from White's point of view.
#[derive(Clone, Copy)]
enum Outcome {
    WhiteWin,
    Draw,
    BlackWin,
}

impl Outcome {
    fn label(self) -> &'static str {
        match self {
            Outcome::WhiteWin => "1.0",
            Outcome::Draw => "0.5",
            Outcome::BlackWin => "0.0",
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let target: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let depth: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let mut written = 0usize;
    let mut games = 0usize;
    // One TT, reused across games (cleared per search) — realistic and cheaper.
    let mut tt = TranspositionTable::new(16);

    while written < target {
        let mut rng = Lcg(seed.wrapping_add(games as u64).wrapping_mul(0x9e3779b97f4a7c15));
        let (positions, outcome) = play_game(&mut rng, depth, &mut tt);
        for fen in positions {
            // Stop exactly at the target so a run is fully determined by its args.
            if written >= target {
                break;
            }
            writeln!(out, "{fen} {}", outcome.label()).expect("write dataset line");
            written += 1;
        }
        games += 1;
        if games.is_multiple_of(200) {
            eprintln!("games {games}, positions {written}/{target}");
        }
    }
    out.flush().expect("flush dataset");
    eprintln!("done: {written} positions from {games} games (depth {depth}, seed {seed})");
}

/// Play one self-play game; return the quiet positions to record (full FENs) and
/// the game outcome (White's POV).
fn play_game(rng: &mut Lcg, depth: u32, tt: &mut TranspositionTable) -> (Vec<String>, Outcome) {
    let mut board = Board::from_str(STARTPOS).expect("startpos FEN");
    let mut counts: HashMap<u64, u8> = HashMap::new();
    let mut positions = Vec::new();

    // Random opening for diversity (not recorded).
    for _ in 0..RANDOM_PLIES {
        let legal = generate_legal(&board);
        if legal.is_empty() {
            return (positions, terminal_outcome(&board));
        }
        let mv = legal[rng.below(legal.len())];
        board.make_move(mv);
    }

    let mut resign_leader: Option<Color> = None;
    let mut resign_count = 0u32;

    for _ in 0..MAX_PLIES {
        let legal = generate_legal(&board);
        if legal.is_empty() {
            return (positions, terminal_outcome(&board));
        }
        if board.halfmove_clock >= 100 {
            return (positions, Outcome::Draw); // fifty-move rule
        }
        let seen = counts.entry(board.hash).or_insert(0);
        *seen += 1;
        if *seen >= 3 {
            return (positions, Outcome::Draw); // threefold repetition
        }

        // Record this position if it's a usable training sample.
        if is_quiet(&board) {
            positions.push(board.to_fen());
        }

        let result = search_with_tt(&board, depth, tt);

        // Decisive-score adjudication, in White's frame (search score is
        // side-to-move-relative).
        let white_score =
            if board.side_to_move == Color::White { result.score } else { -result.score };
        if result.score >= MATE - 1000 {
            // The side to move has a forced mate (mate scores sit near ±MATE).
            return (positions, win_for(board.side_to_move));
        }
        if white_score.abs() >= RESIGN_CP {
            let leader = if white_score > 0 { Color::White } else { Color::Black };
            if resign_leader == Some(leader) {
                resign_count += 1;
            } else {
                resign_leader = Some(leader);
                resign_count = 1;
            }
            if resign_count >= RESIGN_PLIES {
                return (positions, win_for(leader));
            }
        } else {
            resign_leader = None;
            resign_count = 0;
        }

        board.make_move(result.best_move);
    }
    (positions, Outcome::Draw) // ply cap
}

/// Outcome when the side to move has no legal move: checkmate (the side to move
/// loses) or stalemate (draw).
fn terminal_outcome(board: &Board) -> Outcome {
    if in_check(board, board.side_to_move) {
        win_for(board.side_to_move.flip()) // side to move is mated
    } else {
        Outcome::Draw
    }
}

fn win_for(c: Color) -> Outcome {
    match c {
        Color::White => Outcome::WhiteWin,
        Color::Black => Outcome::BlackWin,
    }
}
