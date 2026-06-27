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

// ── Sliding attacks (rook, bishop, queen) ───────────────────────────────────
//
// Unlike the tables above, a slider's reach depends on what blocks it, so these
// are computed per call against an `occupied` bitboard. The method here is the
// straightforward "walk the ray" loop — correct and obvious, but it touches the
// board square by square. Phase 2 replaces it with magic bitboards (one
// multiply + table lookup) once perft proves this version correct; that's the
// deliberate "correct first, fast later" ordering from the iron rules.

/// Rook ray directions as `(file_step, rank_step)`: along ranks and files.
const ROOK_DIRS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// Bishop ray directions: the four diagonals.
const BISHOP_DIRS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

/// Walk each ray out from `sq` until the board edge or the first occupied
/// square, unioning every square reached.
///
/// The first blocker **is included** in the result. That's intentional: this
/// function is pure geometry, so it can't know whether the blocker is friend or
/// foe. Including it means a capture of an enemy blocker is already present;
/// the move generator (issue #15) masks off blockers of the *mover's own*
/// color afterward. Keeping that decision out of here is what lets one function
/// serve both attack detection and capture generation.
fn ray_attacks(sq: Square, occupied: Bitboard, dirs: &[(i8, i8)]) -> Bitboard {
    let mut bb = Bitboard::EMPTY;
    let start_file = sq.file() as i8;
    let start_rank = sq.rank() as i8;
    for &(df, dr) in dirs {
        let mut file = start_file + df;
        let mut rank = start_rank + dr;
        // Recompute file/rank each step and stop when either leaves the board —
        // the same signed-bounds guard the tables use, applied per ray step.
        while (0..8).contains(&file) && (0..8).contains(&rank) {
            let target = Square::from_file_rank(file as u8, rank as u8);
            bb = bb.with(target);
            if occupied.contains(target) {
                break; // blocker reached: include it, then halt this ray.
            }
            file += df;
            rank += dr;
        }
    }
    bb
}

/// Squares a rook on `sq` attacks given the board `occupied` set.
pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    ray_attacks(sq, occupied, &ROOK_DIRS)
}

/// Squares a bishop on `sq` attacks given the board `occupied` set.
pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    ray_attacks(sq, occupied, &BISHOP_DIRS)
}

/// Squares a queen on `sq` attacks — the union of rook and bishop rays, since a
/// queen is exactly a rook plus a bishop.
pub fn queen_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    rook_attacks(sq, occupied).union(bishop_attacks(sq, occupied))
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

    /// Build an occupancy bitboard from a list of square names.
    fn occ(names: &[&str]) -> Bitboard {
        let mut bb = Bitboard::EMPTY;
        for n in names {
            bb = bb.with(sq(n));
        }
        bb
    }

    #[test]
    fn rook_on_empty_board_sweeps_rank_and_file() {
        // a1 with nothing in the way: the whole a-file (7) + rank 1 (7) = 14.
        assert_eq!(rook_attacks(sq("a1"), Bitboard::EMPTY).count(), 14);
        // A center rook also reaches 14 on an empty board.
        assert_eq!(rook_attacks(sq("d4"), Bitboard::EMPTY).count(), 14);
    }

    #[test]
    fn rook_stops_at_blocker_and_includes_it() {
        // Blocker on a4: the rook sees a2, a3, a4 up the file but not a5+.
        let attacks = rook_attacks(sq("a1"), occ(&["a4"]));
        assert!(attacks.contains(sq("a2")));
        assert!(attacks.contains(sq("a3")));
        assert!(attacks.contains(sq("a4")), "blocker square is included (capture)");
        assert!(!attacks.contains(sq("a5")));
    }

    #[test]
    fn rook_boxed_in_on_four_sides() {
        // Blockers one step away in every direction: exactly those four squares.
        let attacks = rook_attacks(sq("d4"), occ(&["d5", "d3", "c4", "e4"]));
        assert_eq!(squares(attacks), vec!["c4", "d3", "d5", "e4"]);
    }

    #[test]
    fn bishop_diagonals_and_blockers() {
        // c1 on an empty board: the two diagonals b2-a3 and d2-h6.
        assert_eq!(
            squares(bishop_attacks(sq("c1"), Bitboard::EMPTY)),
            vec!["a3", "b2", "d2", "e3", "f4", "g5", "h6"]
        );
        // A blocker on e3 halts the long diagonal there.
        let attacks = bishop_attacks(sq("c1"), occ(&["e3"]));
        assert!(attacks.contains(sq("d2")));
        assert!(attacks.contains(sq("e3")));
        assert!(!attacks.contains(sq("f4")));
    }

    #[test]
    fn queen_is_rook_union_bishop() {
        let occupied = occ(&["d6", "f4", "b2"]);
        let expected =
            rook_attacks(sq("d4"), occupied).union(bishop_attacks(sq("d4"), occupied));
        assert_eq!(queen_attacks(sq("d4"), occupied), expected);
        // On an empty board the center queen reaches 14 (rook) + 13 (bishop) = 27.
        assert_eq!(queen_attacks(sq("d4"), Bitboard::EMPTY).count(), 27);
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
