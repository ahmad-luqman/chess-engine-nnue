//! genbook — emit the curated opening book as EPD for match testing (issue #31).
//!
//! SPRT needs varied, balanced starting positions, or every `-repeat` game is
//! the same line and the result is noise. This plays a hand-curated set of
//! mainline openings through the engine's own legal-move generator and prints
//! each resulting position as EPD (the first four FEN fields). Going through
//! real movegen guarantees every line is legal and every FEN well-formed; a
//! mistyped move panics loudly, naming the opening and the offending move.
//!
//! Regenerate the committed book with:
//!
//! ```text
//! cargo run --release --example genbook > books/openings.epd
//! ```
//!
//! Moves are in coordinate (UCI) notation — `e2e4`, not `e4` — because that's
//! what the engine parses; the comment on each line names the opening. Every
//! line ends on a piece move, so the played position has en-passant `-` and the
//! book sidesteps ep edge cases. Colour-reversed `-repeat` games in the SPRT run
//! neutralise any small opening imbalance, so exact material parity isn't needed.

use std::str::FromStr;

use engine::board::Board;
use engine::perft::STARTPOS;
use engine::uci::parse_uci_move;

/// `(name, coordinate move-list)`. ~24 balanced mainlines spanning 1.e4 / 1.d4 /
/// 1.c4 / 1.Nf3, each ending on a non-pawn move (en-passant resolves to `-`).
const OPENINGS: &[(&str, &[&str])] = &[
    // ── 1.e4 ────────────────────────────────────────────────────────────────
    ("Ruy Lopez", &["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4"]),
    ("Italian Game", &["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"]),
    ("Scotch Game", &["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4"]),
    ("Petrov Defense", &["e2e4", "e7e5", "g1f3", "g8f6"]),
    ("Sicilian Najdorf", &["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3"]),
    ("Sicilian Taimanov", &["e2e4", "c7c5", "g1f3", "e7e6", "d2d4", "c5d4", "f3d4", "b8c6"]),
    ("French Winawer", &["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4"]),
    ("Caro-Kann Main", &["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "b8d7"]),
    ("Pirc Defense", &["e2e4", "d7d6", "d2d4", "g8f6", "b1c3"]),
    ("Scandinavian", &["e2e4", "d7d5", "e4d5", "d8d5", "b1c3"]),
    // ── 1.d4 ────────────────────────────────────────────────────────────────
    ("Queen's Gambit Declined", &["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"]),
    ("Slav Defense", &["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6"]),
    ("Nimzo-Indian", &["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"]),
    ("King's Indian", &["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"]),
    ("Grünfeld Defense", &["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "c4d5", "f6d5"]),
    ("Queen's Gambit Accepted", &["d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6"]),
    ("Catalan", &["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5", "f1g2"]),
    ("Benoni", &["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3"]),
    ("Dutch Defense", &["d2d4", "f7f5", "g2g3", "g8f6", "f1g2"]),
    // ── 1.c4 (English) ──────────────────────────────────────────────────────
    ("English Symmetrical", &["c2c4", "c7c5", "b1c3", "b8c6", "g1f3", "g8f6"]),
    ("English Reversed Sicilian", &["c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6"]),
    // ── 1.Nf3 ───────────────────────────────────────────────────────────────
    ("Réti Opening", &["g1f3", "d7d5", "c2c4", "e7e6", "g2g3", "g8f6", "f1g2"]),
    ("King's Indian Attack", &["g1f3", "g8f6", "g2g3", "g7g6", "f1g2", "f8g7"]),
    ("Symmetrical English (Nf3)", &["g1f3", "c7c5", "c2c4", "b8c6", "b1c3", "g8f6"]),
];

fn main() {
    for (name, moves) in OPENINGS {
        let mut board = Board::from_str(STARTPOS).expect("startpos FEN is valid");
        for mv_str in *moves {
            let mv = parse_uci_move(&board, mv_str).unwrap_or_else(|| {
                panic!("illegal or mistyped move `{mv_str}` in opening `{name}`")
            });
            board.make_move(mv);
        }
        // EPD is the first four FEN fields: placement, side-to-move, castling,
        // en-passant. Drop the halfmove/fullmove clocks `to_fen` appends.
        let fen = board.to_fen();
        let epd = fen.split_whitespace().take(4).collect::<Vec<_>>().join(" ");
        println!("{epd}");
    }
}
