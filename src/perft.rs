//! Perft — the move-generation correctness oracle, and the Phase 0 gate.
//!
//! `perft(depth)` counts the leaf nodes of the legal-move tree to a fixed depth
//! by making and unmaking every move along the way. Its value is that it pins
//! *generation* and *make/unmake* against published reference counts at once: a
//! single illegal, missing, or duplicated move anywhere in the tree shows up as
//! a node-count mismatch. The iron rules forbid any search work until perft
//! matches the standard positions, because a generator bug otherwise corrupts
//! every measurement built on top of it.
//!
//! [`perft_divide`] is the companion debugger: when a total is wrong, it shows
//! the per-root-move subtree counts so a discrepancy can be bisected against a
//! reference engine move by move.

use crate::board::Board;
use crate::movegen::generate_legal;
use crate::moves::Move;

// The standard perft positions (chessprogramming-wiki "Perft Results"). These
// are the move generator's reference fixtures; the unit tests below pin their
// node counts, and `benches/` reuses the exact same FENs so correctness and
// performance measure the identical positions (issue #30).
pub const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
pub const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
pub const POS3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
pub const POS4: &str = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
pub const POS5: &str = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1";

/// Count the leaf nodes exactly `depth` plies below `board`.
///
/// `perft(0) == 1` — the position itself is one node. Depth 1 is bulk-counted as
/// the legal-move count: a leaf needs no make/unmake, which is the standard
/// speedup and changes no totals.
pub fn perft(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = generate_legal(board);
    if depth == 1 {
        return moves.len() as u64;
    }
    let mut nodes = 0;
    for mv in moves {
        let undo = board.make_move(mv);
        nodes += perft(board, depth - 1);
        board.unmake_move(mv, undo);
    }
    nodes
}

/// For each legal move, the perft of the resulting position at `depth - 1`.
///
/// Summing the counts yields `perft(board, depth)`. Compared against a trusted
/// engine's `go perft`, this localizes a wrong total to the offending root move,
/// the first step in tracking down a generation or make/unmake bug.
pub fn perft_divide(board: &mut Board, depth: u32) -> Vec<(Move, u64)> {
    assert!(depth >= 1, "perft_divide needs at least depth 1");
    let mut breakdown = Vec::new();
    for mv in generate_legal(board) {
        let undo = board.make_move(mv);
        let nodes = perft(board, depth - 1);
        board.unmake_move(mv, undo);
        breakdown.push((mv, nodes));
    }
    breakdown
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::str::FromStr;

    fn perft_fen(fen: &str, depth: u32) -> u64 {
        perft(&mut Board::from_str(fen).unwrap(), depth)
    }

    /// The fast gate, run by `cargo test`. Each position is taken to a depth that
    /// stays quick in a debug build; the full depth-5/6 numbers live in the
    /// `#[ignore]`d release test below. Counts are the chessprogramming-wiki
    /// "Perft Results" reference values.
    #[test]
    fn perft_matches_reference_shallow() {
        // (fen, depth, expected) — the full progression up to each fast depth, so
        // a regression at any depth is caught, not just the deepest.
        let cases = [
            (STARTPOS, 1, 20),
            (STARTPOS, 2, 400),
            (STARTPOS, 3, 8902),
            (STARTPOS, 4, 197281),
            (KIWIPETE, 1, 48),
            (KIWIPETE, 2, 2039),
            (KIWIPETE, 3, 97862),
            (POS3, 1, 14),
            (POS3, 2, 191),
            (POS3, 3, 2812),
            (POS3, 4, 43238),
            (POS3, 5, 674624),
            (POS4, 1, 6),
            (POS4, 2, 264),
            (POS4, 3, 9467),
            (POS4, 4, 422333),
            (POS5, 1, 44),
            (POS5, 2, 1486),
            (POS5, 3, 62379),
        ];
        for (fen, depth, expected) in cases {
            assert_eq!(perft_fen(fen, depth), expected, "perft({depth}) for {fen}");
        }
    }

    #[test]
    fn perft_zero_is_one() {
        // The position itself is a single node, with no move made.
        assert_eq!(perft_fen(STARTPOS, 0), 1);
    }

    #[test]
    fn divide_sums_to_perft() {
        // perft_divide must partition perft exactly: the root subtree counts add
        // up to the whole, and there is one entry per legal root move.
        let mut board = Board::from_str(KIWIPETE).unwrap();
        let breakdown = perft_divide(&mut board, 3);
        assert_eq!(breakdown.len(), 48); // Kiwipete has 48 legal moves
        let total: u64 = breakdown.iter().map(|&(_, n)| n).sum();
        assert_eq!(total, 97862);
    }

    /// The real Phase 0 gate: the standard positions to depth 5 (plus startpos to
    /// depth 6). These run tens-to-hundreds of millions of nodes, so they are
    /// `#[ignore]`d and meant for release:
    ///
    /// ```text
    /// cargo test --release -- --ignored perft_matches_reference_deep
    /// ```
    #[test]
    #[ignore = "slow; run in release as the Phase 0 perft gate"]
    fn perft_matches_reference_deep() {
        let cases = [
            (STARTPOS, 5, 4865609),
            (STARTPOS, 6, 119060324),
            (KIWIPETE, 4, 4085603),
            (KIWIPETE, 5, 193690690),
            (POS3, 6, 11030083),
            (POS4, 5, 15833292),
            (POS5, 4, 2103487),
            (POS5, 5, 89941194),
        ];
        for (fen, depth, expected) in cases {
            assert_eq!(perft_fen(fen, depth), expected, "perft({depth}) for {fen}");
        }
    }
}
