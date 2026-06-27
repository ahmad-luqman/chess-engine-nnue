//! Move generation: attack sets for every piece.
//!
//! Pieces split into two worlds by geometry:
//!
//! - **Non-sliders** (knight, king, pawn) attack a *fixed* set of squares
//!   relative to where they stand — independent of what else is on the board.
//!   Their attacks are computed once into a `[Bitboard; 64]` lookup table (this
//!   file's first half) and never recomputed.
//! - **Sliders** (rook, bishop, queen) attack *along rays until something blocks
//!   them*, so their attacks depend on the occupancy of the whole board and must
//!   be computed per position (second half — added in issue #13).
//!
//! The one bug that haunts table generation is **file wraparound**: "one file to
//! the left of a1" must be *off the board*, not h-file of the rank below. We
//! guard it by computing target file/rank as signed integers and discarding any
//! target whose file or rank leaves `0..8`.

use crate::bitboard::Bitboard;
use crate::types::{Color, Square};

// ── Non-sliding attack tables (knight, king, pawn) ──────────────────────────
//
// All three are built at compile time by `const fn`s below, so there is zero
// runtime init cost and no need for a lazy-static dependency. Inside a `const
// fn` we can't call `Bitboard::with`/`union` (not const) or use `for` loops (no
// iterators in const context), so the builders work on the raw `u64` with
// `while` loops and bit shifts.

/// Knight attacks from each square: `KNIGHT_ATTACKS[sq]` is every square a
/// knight on `sq` attacks.
pub const KNIGHT_ATTACKS: [Bitboard; 64] = knight_table();

/// King attacks from each square (the up-to-8 adjacent squares).
pub const KING_ATTACKS: [Bitboard; 64] = king_table();

/// Pawn *capture* targets, indexed `[color][sq]`. Pushes are not here: a push
/// depends on the destination being empty (and double pushes on rank), so it is
/// positional, not a fixed offset — the move generator (issue #15) handles
/// pushes directly. This table is exactly the two forward diagonals.
pub const PAWN_ATTACKS: [[Bitboard; 64]; 2] = [pawn_table(1), pawn_table(-1)];

/// Build a non-sliding table from a fixed list of `(file_delta, rank_delta)`
/// jumps, applying file-wrap masking. Shared by the knight and king builders —
/// the only thing that differs between them is the offset list.
const fn jump_table(offsets: &[(i8, i8)]) -> [Bitboard; 64] {
    let mut table = [Bitboard(0); 64];
    let mut sq = 0usize;
    while sq < 64 {
        let file = (sq & 7) as i8;
        let rank = (sq >> 3) as i8;
        let mut bb = 0u64;
        let mut i = 0;
        while i < offsets.len() {
            let (df, dr) = offsets[i];
            let tf = file + df;
            let tr = rank + dr;
            // Keep the target only if it stays on the board. This bounds check
            // IS the anti-wraparound guard: without it, df = -1 on the a-file
            // would land on the previous rank's h-file.
            if tf >= 0 && tf < 8 && tr >= 0 && tr < 8 {
                bb |= 1u64 << (tr as usize * 8 + tf as usize);
            }
            i += 1;
        }
        table[sq] = Bitboard(bb);
        sq += 1;
    }
    table
}

const fn knight_table() -> [Bitboard; 64] {
    // The eight L-shaped jumps: two in one axis, one in the other.
    jump_table(&[
        (1, 2),
        (2, 1),
        (2, -1),
        (1, -2),
        (-1, -2),
        (-2, -1),
        (-2, 1),
        (-1, 2),
    ])
}

const fn king_table() -> [Bitboard; 64] {
    // The eight neighbours: every (df, dr) in {-1,0,1}² except (0,0).
    jump_table(&[
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ])
}

/// Build the pawn capture table for one color. `rank_delta` is +1 for White
/// (pawns move up the board) and -1 for Black; captures are the two squares one
/// file left and right of that forward square.
const fn pawn_table(rank_delta: i8) -> [Bitboard; 64] {
    let mut table = [Bitboard(0); 64];
    let mut sq = 0usize;
    while sq < 64 {
        let file = (sq & 7) as i8;
        let rank = (sq >> 3) as i8;
        let mut bb = 0u64;
        // df steps -1 then +1 (skipping the straight-ahead 0, which is a push).
        let mut df = -1i8;
        while df <= 1 {
            if df != 0 {
                let tf = file + df;
                let tr = rank + rank_delta;
                if tf >= 0 && tf < 8 && tr >= 0 && tr < 8 {
                    bb |= 1u64 << (tr as usize * 8 + tf as usize);
                }
            }
            df += 2;
        }
        table[sq] = Bitboard(bb);
        sq += 1;
    }
    table
}

/// Squares a knight on `sq` attacks.
pub fn knight_attacks(sq: Square) -> Bitboard {
    KNIGHT_ATTACKS[sq.0 as usize]
}

/// Squares a king on `sq` attacks.
pub fn king_attacks(sq: Square) -> Bitboard {
    KING_ATTACKS[sq.0 as usize]
}

/// Squares a pawn of `color` on `sq` attacks (its two capture diagonals).
pub fn pawn_attacks(color: Color, sq: Square) -> Bitboard {
    PAWN_ATTACKS[color.index()][sq.0 as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::str::FromStr;

    fn sq(name: &str) -> Square {
        Square::from_str(name).unwrap()
    }

    /// Collect an attack bitboard into a sorted Vec of square names, for
    /// readable assertions.
    fn squares(mut bb: Bitboard) -> Vec<String> {
        let mut names = Vec::new();
        while let Some(s) = bb.pop_lsb() {
            names.push(s.to_string());
        }
        names.sort();
        names
    }

    #[test]
    fn knight_in_the_center_hits_eight() {
        let mut names = squares(knight_attacks(sq("d4")));
        let mut expected = vec!["b3", "b5", "c2", "c6", "e2", "e6", "f3", "f5"];
        expected.sort();
        names.sort();
        assert_eq!(names, expected);
    }

    #[test]
    fn knight_in_the_corner_does_not_wrap() {
        // The classic wraparound case: a knight on a1 has exactly two moves.
        // If file masking were missing, "two files left" would appear on the
        // g/h files.
        assert_eq!(squares(knight_attacks(sq("a1"))), vec!["b3", "c2"]);
        assert_eq!(knight_attacks(sq("a1")).count(), 2);
        assert_eq!(knight_attacks(sq("h8")).count(), 2);
    }

    #[test]
    fn king_corner_and_center() {
        assert_eq!(squares(king_attacks(sq("a1"))), vec!["a2", "b1", "b2"]);
        assert_eq!(king_attacks(sq("a1")).count(), 3);
        assert_eq!(king_attacks(sq("e4")).count(), 8);
        assert_eq!(king_attacks(sq("h8")).count(), 3);
    }

    #[test]
    fn pawn_captures_point_forward() {
        // White pawns attack up the board, Black down.
        assert_eq!(squares(pawn_attacks(Color::White, sq("e4"))), vec!["d5", "f5"]);
        assert_eq!(squares(pawn_attacks(Color::Black, sq("e5"))), vec!["d4", "f4"]);
        // Edge files attack only inward — no wrap.
        assert_eq!(squares(pawn_attacks(Color::White, sq("a2"))), vec!["b3"]);
        assert_eq!(squares(pawn_attacks(Color::Black, sq("h7"))), vec!["g6"]);
    }

    #[test]
    fn pawn_attacks_are_color_mirrored() {
        // A white pawn on e4 attacks d5/f5; a black pawn on e5 attacks d4/f4 —
        // the same diagonals reflected across the e4/e5 boundary.
        for file in 0..8u8 {
            for rank in 1..7u8 {
                let s = Square::from_file_rank(file, rank);
                let white = pawn_attacks(Color::White, s);
                // The black pawn one rank higher attacks this square's mirror.
                let mirror = Square::from_file_rank(file, rank + 1);
                let black = pawn_attacks(Color::Black, mirror);
                assert_eq!(white.count(), black.count());
            }
        }
    }
}
