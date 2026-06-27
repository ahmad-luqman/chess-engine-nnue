//! Static evaluation.
//!
//! Evaluation answers one question for the search: *how good is this position,
//! right now, without looking further?* It returns a score in **centipawns**
//! (hundredths of a pawn) from the point of view of the **side to move** —
//! positive means the side to move is better. That sign convention is what makes
//! it pair cleanly with negamax, which always reasons from the mover's seat.
//!
//! ## Why an interface
//!
//! The whole arc of this engine is hand-crafted eval (material + piece-square
//! tables) → **NNUE** (a small neural net) in Phase 4. We keep evaluation behind
//! the [`Evaluator`] trait so that swap is a localized change: search holds an
//! evaluator and calls `evaluate` at the leaves; replacing the concrete type is
//! the substitution. We deliberately keep the trait *minimal* — the real work of
//! making NNUE fast (an accumulator updated incrementally inside make/unmake)
//! happens elsewhere; the trait is just the seam where a score is read out.
//!
//! Issue #19 ships material-only scoring. Issue #20 adds piece-square tables to
//! the same [`Material`] evaluator.

use crate::board::Board;
use crate::types::{Color, PieceType};

/// Something that can statically score a position, in centipawns, from the
/// perspective of the side to move (positive = side to move is better).
pub trait Evaluator {
    fn evaluate(&self, board: &Board) -> i32;
}

/// Centipawn value of each piece type for material counting. The king is never
/// counted — both sides always have exactly one, so it cancels, and giving it a
/// finite value would just be noise. These are the classic values; PSTs (issue
/// #20) layer positional adjustments on top.
const PIECE_VALUE: [i32; 6] = [
    100, // Pawn
    320, // Knight
    330, // Bishop
    500, // Rook
    900, // Queen
    0,   // King — not counted
];

/// The hand-crafted evaluator: material balance (issue #19), with piece-square
/// tables to come (issue #20).
///
/// Zero-sized and stateless today. When NNUE replaces it (Phase 4) the evaluator
/// becomes stateful — it owns an accumulator — which is why search holds an
/// `Evaluator` value rather than calling a free function.
#[derive(Clone, Copy, Default)]
pub struct Material;

impl Evaluator for Material {
    fn evaluate(&self, board: &Board) -> i32 {
        // Sum material white-relative (White positive, Black negative), then flip
        // to side-to-move-relative at the end. Computing white-relative first
        // keeps the per-piece-type arithmetic sign-free and easy to read.
        let white = material_for(board, Color::White);
        let black = material_for(board, Color::Black);
        let white_relative = white - black;

        match board.side_to_move {
            Color::White => white_relative,
            Color::Black => -white_relative,
        }
    }
}

/// Total material value (centipawns) of `color`'s pieces.
fn material_for(board: &Board, color: Color) -> i32 {
    let own = board.color(color);
    let mut total = 0;
    for pt in [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
    ] {
        let count = board.pieces(pt).intersect(own).count() as i32;
        total += count * PIECE_VALUE[pt.index()];
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::str::FromStr;

    fn eval(fen: &str) -> i32 {
        Material.evaluate(&Board::from_str(fen).unwrap())
    }

    #[test]
    fn startpos_is_balanced() {
        // Symmetric material → exactly even, regardless of side to move.
        assert_eq!(eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), 0);
        assert_eq!(eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"), 0);
    }

    #[test]
    fn extra_queen_is_worth_about_nine_pawns() {
        // White has all its pieces; Black is missing its queen. White to move
        // sees +900; the same position with Black to move sees -900.
        let fen_w = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let fen_b = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1";
        assert_eq!(eval(fen_w), 900);
        assert_eq!(eval(fen_b), -900);
    }

    #[test]
    fn score_is_side_to_move_relative() {
        // A lone white rook vs bare kings is +500 for White, -500 for Black —
        // the same position scored from whichever seat is to move.
        let white_up = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1";
        let black_to_move = "4k3/8/8/8/8/8/8/R3K3 b - - 0 1";
        assert_eq!(eval(white_up), 500);
        assert_eq!(eval(black_to_move), -500);
    }

    #[test]
    fn color_mirror_is_symmetric() {
        // Mirroring colors and side to move must yield the same stm-relative
        // score. Here: White up a knight to move, vs Black up a knight to move.
        let white_knight_up = "4k3/8/8/8/8/8/8/N3K3 w - - 0 1";
        let black_knight_up = "n3k3/8/8/8/8/8/8/4K3 b - - 0 1";
        assert_eq!(eval(white_knight_up), eval(black_knight_up));
        assert_eq!(eval(white_knight_up), 320);
    }
}
