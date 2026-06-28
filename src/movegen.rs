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
use crate::board::{Board, CastlingRights};
use crate::moves::{Move, MoveFlag};
use crate::types::{Color, PieceType, Square};

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
// Unlike the tables above, a slider's reach depends on what blocks it. The
// geometry oracle is the straightforward "walk the ray" loop ([`ray_attacks`]):
// correct and obvious, but it touches the board square by square. The public
// `rook_attacks`/`bishop_attacks` now delegate to **magic bitboards** (one
// multiply + table lookup, see `crate::magic`), which `ray_attacks` still backs
// — it builds and verifies the magic tables. This is the "correct first, fast
// later" ordering from the iron rules: perft proved the ray loop, and magics are
// a drop-in that perft must still match exactly (issue #27).

/// Rook ray directions as `(file_step, rank_step)`: along ranks and files.
pub(crate) const ROOK_DIRS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// Bishop ray directions: the four diagonals.
pub(crate) const BISHOP_DIRS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

/// Walk each ray out from `sq` until the board edge or the first occupied
/// square, unioning every square reached.
///
/// The first blocker **is included** in the result. That's intentional: this
/// function is pure geometry, so it can't know whether the blocker is friend or
/// foe. Including it means a capture of an enemy blocker is already present;
/// the move generator (issue #15) masks off blockers of the *mover's own*
/// color afterward. Keeping that decision out of here is what lets one function
/// serve both attack detection and capture generation.
pub(crate) fn ray_attacks(sq: Square, occupied: Bitboard, dirs: &[(i8, i8)]) -> Bitboard {
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
///
/// Drop-in over magic bitboards; identical result to `ray_attacks(sq, occupied,
/// &ROOK_DIRS)` but a single multiply-shift-index instead of a per-square walk.
pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    crate::magic::rook_attacks(sq, occupied)
}

/// Squares a bishop on `sq` attacks given the board `occupied` set. Drop-in over
/// magic bitboards (see [`rook_attacks`]).
pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    crate::magic::bishop_attacks(sq, occupied)
}

/// Squares a queen on `sq` attacks — the union of rook and bishop rays, since a
/// queen is exactly a rook plus a bishop.
pub fn queen_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    rook_attacks(sq, occupied).union(bishop_attacks(sq, occupied))
}

// ── Legal move generation (issue #15) ───────────────────────────────────────
//
// Strategy: **pseudo-legal generation + copy-make legality filter**. We first
// emit every move that respects piece geometry and own-piece blocking, then keep
// only those that leave the mover's own king unattacked, by applying each to a
// throwaway clone and testing king safety. This is the "correct first" path from
// the iron rules: it costs a board copy per move but needs no pin/check-evasion
// machinery, and it gets three notoriously fiddly cases right *for free* —
// en-passant discovered checks (the "ep-pin"), moving while pinned, and escaping
// (double) check — because the filter simply asks "is my king safe after this?".
//
// The one case the destination-only filter can't see is **castling**, where the
// king slides across squares it doesn't end on. Those are validated explicitly
// during generation (rights, emptiness, and no attack on the king's path) and so
// are appended already-legal, bypassing the filter.
//
// Performance is deliberately not a concern yet: the clone-per-move and the
// `Vec` allocation are what perft (issue #17) will measure and issue #16's
// in-place make/unmake will replace. Magic bitboards come even later.

/// Is `sq` attacked by any piece of color `by`?
///
/// Runs the attack tables "in reverse": a square is attacked by a knight iff a
/// `by`-knight sits on one of the squares a knight on `sq` would reach, and the
/// same symmetry holds for kings and (over the current occupancy) sliders. Pawns
/// are the exception — their attacks aren't symmetric — so we probe with the
/// *opposite* color's pawn table from `sq`.
pub fn is_square_attacked(board: &Board, sq: Square, by: Color) -> bool {
    let by_pieces = board.color(by);

    let pawns = board.pieces(PieceType::Pawn).intersect(by_pieces);
    if !pawn_attacks(by.flip(), sq).intersect(pawns).is_empty() {
        return true;
    }

    let knights = board.pieces(PieceType::Knight).intersect(by_pieces);
    if !knight_attacks(sq).intersect(knights).is_empty() {
        return true;
    }

    let kings = board.pieces(PieceType::King).intersect(by_pieces);
    if !king_attacks(sq).intersect(kings).is_empty() {
        return true;
    }

    let occupied = board.occupied();
    let queens = board.pieces(PieceType::Queen);
    let diagonal = board.pieces(PieceType::Bishop).union(queens).intersect(by_pieces);
    if !bishop_attacks(sq, occupied).intersect(diagonal).is_empty() {
        return true;
    }
    let straight = board.pieces(PieceType::Rook).union(queens).intersect(by_pieces);
    if !rook_attacks(sq, occupied).intersect(straight).is_empty() {
        return true;
    }

    false
}

/// The square the `color` king stands on. Every legal position has exactly one.
///
/// Public because search needs it for mate/stalemate detection, and NNUE
/// (Phase 4) will want king squares for king-relative features.
pub fn king_square(board: &Board, color: Color) -> Square {
    let mut kings = board.pieces(PieceType::King).intersect(board.color(color));
    kings.pop_lsb().expect("every position has a king of each color")
}

/// True if `color`'s king is currently in check — i.e. attacked by the enemy.
///
/// Search uses this to tell checkmate (no legal moves *and* in check) from
/// stalemate (no legal moves, *not* in check).
pub fn in_check(board: &Board, color: Color) -> bool {
    is_square_attacked(board, king_square(board, color), color.flip())
}

/// True if making `mv` on the throwaway `work` board leaves the mover's own king
/// unattacked. `work` is mutated and restored in place via make/unmake, so the
/// whole legality filter needs just one board clone for an entire position.
fn leaves_king_safe(work: &mut Board, us: Color, mv: Move) -> bool {
    let undo = work.make_move(mv);
    let safe = !is_square_attacked(work, king_square(work, us), us.flip());
    work.unmake_move(mv, undo);
    safe
}

/// Append a non-pawn piece's moves from `from` to `targets` (already masked free
/// of own pieces), tagging each as a capture or quiet by what sits on the square.
fn add_target_moves(from: Square, targets: Bitboard, enemy: Bitboard, out: &mut Vec<Move>) {
    let mut t = targets;
    while let Some(to) = t.pop_lsb() {
        let flag = if enemy.contains(to) { MoveFlag::CAPTURE } else { MoveFlag::QUIET };
        out.push(Move::new(from, to, flag));
    }
}

/// Emit the four promotion moves (knight, bishop, rook, queen) for a pawn
/// reaching the back rank, capturing or not.
fn add_promotions(from: Square, to: Square, capture: bool, out: &mut Vec<Move>) {
    let flags = if capture {
        [
            MoveFlag::KNIGHT_PROMO_CAPTURE,
            MoveFlag::BISHOP_PROMO_CAPTURE,
            MoveFlag::ROOK_PROMO_CAPTURE,
            MoveFlag::QUEEN_PROMO_CAPTURE,
        ]
    } else {
        [
            MoveFlag::KNIGHT_PROMO,
            MoveFlag::BISHOP_PROMO,
            MoveFlag::ROOK_PROMO,
            MoveFlag::QUEEN_PROMO,
        ]
    };
    for flag in flags {
        out.push(Move::new(from, to, flag));
    }
}

/// Pseudo-legal pawn moves: single/double pushes, diagonal captures, promotions
/// (4 per reachable back-rank square), and en-passant.
fn generate_pawn_moves(
    board: &Board,
    us: Color,
    own: Bitboard,
    enemy: Bitboard,
    occupied: Bitboard,
    out: &mut Vec<Move>,
) {
    // Direction of "forward" for this color, plus the ranks that matter. A pawn
    // never stands on its own back rank, so `rank + dr` is always 0..=7.
    let (dr, start_rank, last_rank): (i8, u8, u8) = match us {
        Color::White => (1, 1, 7),
        Color::Black => (-1, 6, 0),
    };

    let mut pawns = board.pieces(PieceType::Pawn).intersect(own);
    while let Some(from) = pawns.pop_lsb() {
        let file = from.file();
        let rank = from.rank();

        // ── Pushes ──
        let one = Square::from_file_rank(file, (rank as i8 + dr) as u8);
        if !occupied.contains(one) {
            if one.rank() == last_rank {
                add_promotions(from, one, false, out);
            } else {
                out.push(Move::new(from, one, MoveFlag::QUIET));
                // A double push is only possible from the start rank and only if
                // *both* the one- and two-step squares are empty.
                if rank == start_rank {
                    let two = Square::from_file_rank(file, (one.rank() as i8 + dr) as u8);
                    if !occupied.contains(two) {
                        out.push(Move::new(from, two, MoveFlag::DOUBLE_PAWN_PUSH));
                    }
                }
            }
        }

        // ── Captures (including capture-promotions) ──
        let mut caps = pawn_attacks(us, from).intersect(enemy);
        while let Some(to) = caps.pop_lsb() {
            if to.rank() == last_rank {
                add_promotions(from, to, true, out);
            } else {
                out.push(Move::new(from, to, MoveFlag::CAPTURE));
            }
        }

        // ── En passant ── geometry only; the copy-make filter rejects the
        // discovered-check "ep-pin" case afterward.
        if let Some(ep) = board.ep_square {
            if pawn_attacks(us, from).contains(ep) {
                out.push(Move::new(from, ep, MoveFlag::EN_PASSANT));
            }
        }
    }
}

/// Append the legal castling moves for the side to move. Validated fully here —
/// rights present, the squares between king and rook empty, the king not in
/// check, and no square the king *passes over or lands on* attacked — so these
/// skip the copy-make filter. The square beside the queen's rook (b-file) is
/// allowed to be attacked; only the king's three-square path matters.
fn generate_castling(board: &Board, us: Color, occupied: Bitboard, out: &mut Vec<Move>) {
    let enemy = us.flip();
    let king_from = king_square(board, us);
    // Cannot castle out of check.
    if is_square_attacked(board, king_from, enemy) {
        return;
    }

    let rank = match us {
        Color::White => 0,
        Color::Black => 7,
    };
    let (king_right, queen_right) = match us {
        Color::White => (CastlingRights::WHITE_KING, CastlingRights::WHITE_QUEEN),
        Color::Black => (CastlingRights::BLACK_KING, CastlingRights::BLACK_QUEEN),
    };

    // King side: f and g empty; f and g unattacked (e already checked above).
    if board.castling.has(king_right) {
        let f = Square::from_file_rank(5, rank);
        let g = Square::from_file_rank(6, rank);
        if !occupied.contains(f)
            && !occupied.contains(g)
            && !is_square_attacked(board, f, enemy)
            && !is_square_attacked(board, g, enemy)
        {
            out.push(Move::new(king_from, g, MoveFlag::KING_CASTLE));
        }
    }

    // Queen side: b, c, d empty; c and d unattacked (b may be attacked).
    if board.castling.has(queen_right) {
        let b = Square::from_file_rank(1, rank);
        let c = Square::from_file_rank(2, rank);
        let d = Square::from_file_rank(3, rank);
        if !occupied.contains(b)
            && !occupied.contains(c)
            && !occupied.contains(d)
            && !is_square_attacked(board, c, enemy)
            && !is_square_attacked(board, d, enemy)
        {
            out.push(Move::new(king_from, c, MoveFlag::QUEEN_CASTLE));
        }
    }
}

/// Every legal move for the side to move.
///
/// Pseudo-legal moves for pawns, knights, sliders, and the king are generated,
/// then filtered down to those that leave the king safe; legal castling moves
/// (validated during generation) are appended last.
pub fn generate_legal(board: &Board) -> Vec<Move> {
    let us = board.side_to_move;
    let own = board.color(us);
    let enemy = board.color(us.flip());
    let occupied = board.occupied();

    let mut moves = Vec::new();

    generate_pawn_moves(board, us, own, enemy, occupied, &mut moves);

    let mut knights = board.pieces(PieceType::Knight).intersect(own);
    while let Some(from) = knights.pop_lsb() {
        add_target_moves(from, knight_attacks(from).minus(own), enemy, &mut moves);
    }

    let mut bishops = board.pieces(PieceType::Bishop).intersect(own);
    while let Some(from) = bishops.pop_lsb() {
        add_target_moves(from, bishop_attacks(from, occupied).minus(own), enemy, &mut moves);
    }

    let mut rooks = board.pieces(PieceType::Rook).intersect(own);
    while let Some(from) = rooks.pop_lsb() {
        add_target_moves(from, rook_attacks(from, occupied).minus(own), enemy, &mut moves);
    }

    let mut queens = board.pieces(PieceType::Queen).intersect(own);
    while let Some(from) = queens.pop_lsb() {
        add_target_moves(from, queen_attacks(from, occupied).minus(own), enemy, &mut moves);
    }

    let king = king_square(board, us);
    add_target_moves(king, king_attacks(king).minus(own), enemy, &mut moves);

    // Filter the pseudo-legal moves down to the legal ones, reusing one working
    // copy across all of them (make/unmake in place, no per-move clone)...
    let mut work = board.clone();
    moves.retain(|&mv| leaves_king_safe(&mut work, us, mv));
    // ...then add castling, which was validated as legal during generation.
    generate_castling(board, us, occupied, &mut moves);

    moves
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

    // ── Legal move generation ──────────────────────────────────────────────

    fn board(fen: &str) -> Board {
        Board::from_str(fen).unwrap()
    }

    /// perft(1): the number of legal moves in a position — the cheapest, most
    /// effective single check on the whole generator. Counts verified against
    /// the chessprogramming-wiki "Perft Results" standard positions.
    fn legal_count(fen: &str) -> usize {
        generate_legal(&board(fen)).len()
    }

    #[test]
    fn perft1_matches_standard_positions() {
        // Startpos: the canonical 20 opening moves (16 pawn + 4 knight).
        assert_eq!(
            legal_count("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            20
        );
        // Kiwipete — castling, pins, and captures all live: 48.
        assert_eq!(
            legal_count("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
            48
        );
        // Position 3 — a sparse rook-and-pawn endgame: 14.
        assert_eq!(legal_count("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"), 14);
        // Position 4 and its mirror — only 6 legal moves, and color-symmetric.
        assert_eq!(
            legal_count("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
            6
        );
        assert_eq!(
            legal_count("r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1"),
            6
        );
        // Position 5 — a tactical middlegame: 44.
        assert_eq!(legal_count("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1"), 44);
    }

    #[test]
    fn promotion_makes_four_moves_per_pawn() {
        // A lone white pawn on a7 promotes; exactly four of the legal moves are
        // promotions (one per piece), and they all land on a8.
        let moves = generate_legal(&board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"));
        let promos: Vec<_> = moves.iter().filter(|m| m.is_promotion()).collect();
        assert_eq!(promos.len(), 4);
        assert!(promos.iter().all(|m| m.to() == Square::from_str("a8").unwrap()));
    }

    #[test]
    fn en_passant_is_offered_when_available() {
        // White pawn e5, black pawn d5, ep target d6: e5xd6 e.p. is legal.
        let moves = generate_legal(&board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"));
        let ep: Vec<_> = moves.iter().filter(|m| m.is_en_passant()).collect();
        assert_eq!(ep.len(), 1);
        assert_eq!(ep[0].to(), Square::from_str("d6").unwrap());
    }

    #[test]
    fn en_passant_pin_is_rejected() {
        // The classic ep-pin: capturing e.p. would vacate both pawns from the
        // 4th rank, exposing the black king on a4 to the white rook on h4. The
        // copy-make filter must reject it — zero en-passant moves — while the
        // king still has ordinary moves.
        let moves = generate_legal(&board("8/8/8/8/k2pP2R/8/8/4K3 b - e3 0 1"));
        assert!(moves.iter().all(|m| !m.is_en_passant()));
        assert!(!moves.is_empty());
    }

    #[test]
    fn castling_both_sides_on_empty_back_rank() {
        let moves = generate_legal(&board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"));
        let castles: Vec<_> = moves.iter().filter(|m| m.is_castle()).collect();
        assert_eq!(castles.len(), 2);
    }

    #[test]
    fn castling_blocked_through_attacked_square() {
        // Black rook on g8 attacks g1: king-side castling is forbidden (the king
        // would pass over/into g1), but queen-side remains legal — only one
        // castle move, the queen-side one (king e1 -> c1).
        let moves = generate_legal(&board("4k1r1/8/8/8/8/8/8/R3K2R w KQ - 0 1"));
        let castles: Vec<_> = moves.iter().filter(|m| m.is_castle()).collect();
        assert_eq!(castles.len(), 1);
        assert_eq!(castles[0].to(), Square::from_str("c1").unwrap());
    }

    #[test]
    fn king_in_check_must_respond() {
        // Black king e8 in check from a white rook on e1 down the open e-file.
        // Every generated move must genuinely end the check: re-apply it and
        // confirm the black king is no longer attacked.
        let b = board("4k3/8/8/8/8/8/8/4R1K1 b - - 0 1");
        let moves = generate_legal(&b);
        assert!(!moves.is_empty());
        let mut work = b.clone();
        for &mv in &moves {
            assert!(
                leaves_king_safe(&mut work, Color::Black, mv),
                "{mv} leaves the king in check"
            );
        }
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
