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
//! Issue #19 shipped material-only scoring. Issue #20 adds **piece-square
//! tables** (PSTs) to the same [`Material`] evaluator: a per-(piece, square)
//! bonus that teaches the engine *where* pieces belong — knights toward the
//! centre, rooks on open ranks, pawns pushing, the king tucked away in the
//! middlegame. Material alone makes every quiet move look identical; PSTs give
//! the search a positional gradient to climb.

use crate::board::Board;
use crate::types::{Color, PieceType, Square};

/// Something that can statically score a position, in centipawns, from the
/// perspective of the side to move (positive = side to move is better).
pub trait Evaluator {
    fn evaluate(&self, board: &Board) -> i32;
}

/// Centipawn value of each piece type for material counting. The king is never
/// counted — both sides always have exactly one, so it cancels, and giving it a
/// finite value would just be noise. These are the classic values; PSTs (issue
/// #20) layer positional adjustments on top.
pub const PIECE_VALUE: [i32; 6] = [
    100, // Pawn
    320, // Knight
    330, // Bishop
    500, // Rook
    900, // Queen
    0,   // King — not counted
];

/// The hand-crafted evaluator: material balance + piece-square tables.
///
/// Zero-sized and stateless today. When NNUE replaces it (Phase 4) the evaluator
/// becomes stateful — it owns an accumulator — which is why search holds an
/// `Evaluator` value rather than calling a free function.
#[derive(Clone, Copy, Default)]
pub struct Material;

impl Evaluator for Material {
    fn evaluate(&self, board: &Board) -> i32 {
        // Sum each side's material + PST score white-relative (White positive,
        // Black negative), then flip to side-to-move-relative at the end.
        // Computing white-relative first keeps the arithmetic sign-free.
        let white_relative = side_score(board, Color::White) - side_score(board, Color::Black);

        match board.side_to_move {
            Color::White => white_relative,
            Color::Black => -white_relative,
        }
    }
}

/// Total static score (centipawns) of `color`'s pieces: material value plus the
/// piece-square bonus for each piece's square.
fn side_score(board: &Board, color: Color) -> i32 {
    let own = board.color(color);
    let mut total = 0;
    for pt in [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King, // zero material, but its PST matters (king safety).
    ] {
        let table = &PIECE_SQUARE_TABLE[pt.index()];
        let value = PIECE_VALUE[pt.index()];
        let mut bb = board.pieces(pt).intersect(own);
        while let Some(sq) = bb.pop_lsb() {
            total += value + table[pst_index(color, sq)];
        }
    }
    total
}

/// Map a piece's square to its index into a piece-square table.
///
/// The tables below are laid out **rank 8 first** (the way a board looks from
/// White's side, and the way every reference prints them), so table index 0 is
/// a8 and 63 is h1. A White piece on `sq` (a1 = 0) therefore reads `sq ^ 56`,
/// which flips the rank to convert a1-origin coordinates into the rank-8-first
/// layout. A Black piece reads `sq` directly: that is the same square a White
/// piece would occupy after a vertical board flip, so Black automatically gets
/// the mirror-image bonuses without a second set of tables.
fn pst_index(color: Color, sq: Square) -> usize {
    let idx = match color {
        Color::White => sq.0 ^ 56,
        Color::Black => sq.0,
    };
    idx as usize
}

/// Piece-square tables, indexed by [`PieceType::index`] then by board square in
/// **rank-8-first** layout (index 0 = a8 … 63 = h1) — see [`pst_index`].
///
/// These are Tomasz Michniewski's "Simplified Evaluation Function" tables, a
/// well-known public-domain starting point. They are written from White's
/// perspective; Black reuses them mirrored (again, see [`pst_index`]). A single
/// king table (middlegame) is used for now — game-phase tapering (an endgame
/// king table that walks the king toward the centre) is a later refinement.
const PIECE_SQUARE_TABLE: [[i32; 64]; 6] = [
    // Pawn — reward central advances; discourage the f2/g2 squares weakening.
    [
          0,   0,   0,   0,   0,   0,   0,   0,
         50,  50,  50,  50,  50,  50,  50,  50,
         10,  10,  20,  30,  30,  20,  10,  10,
          5,   5,  10,  25,  25,  10,   5,   5,
          0,   0,   0,  20,  20,   0,   0,   0,
          5,  -5, -10,   0,   0, -10,  -5,   5,
          5,  10,  10, -20, -20,  10,  10,   5,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    // Knight — strongly central; corners and edges are poor.
    [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20,   0,   0,   0,   0, -20, -40,
        -30,   0,  10,  15,  15,  10,   0, -30,
        -30,   5,  15,  20,  20,  15,   5, -30,
        -30,   0,  15,  20,  20,  15,   0, -30,
        -30,   5,  10,  15,  15,  10,   5, -30,
        -40, -20,   0,   5,   5,   0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    ],
    // Bishop — long diagonals; avoid getting stuck on the back rank.
    [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,  10,  10,   5,   0, -10,
        -10,   5,   5,  10,  10,   5,   5, -10,
        -10,   0,  10,  10,  10,  10,   0, -10,
        -10,  10,  10,  10,  10,  10,  10, -10,
        -10,   5,   0,   0,   0,   0,   5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ],
    // Rook — the 7th rank and central files; small penalty on the a/h edges.
    [
          0,   0,   0,   0,   0,   0,   0,   0,
          5,  10,  10,  10,  10,  10,  10,   5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
          0,   0,   0,   5,   5,   0,   0,   0,
    ],
    // Queen — mild central preference; nothing drastic.
    [
        -20, -10, -10,  -5,  -5, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,   5,   5,   5,   0, -10,
         -5,   0,   5,   5,   5,   5,   0,  -5,
          0,   0,   5,   5,   5,   5,   0,  -5,
        -10,   5,   5,   5,   5,   5,   0, -10,
        -10,   0,   5,   0,   0,   0,   0, -10,
        -20, -10, -10,  -5,  -5, -10, -10, -20,
    ],
    // King (middlegame) — stay home and castled; the centre is dangerous.
    [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
         20,  20,   0,   0,   0,   0,  20,  20,
         20,  30,  10,   0,   0,  10,  30,  20,
    ],
];

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
    fn material_dominates_a_missing_queen() {
        // Startpos but Black is missing its queen. Material (≈900) dwarfs any PST
        // wobble, so White to move is up close to a queen and Black to move is
        // down close to a queen. Range, not exact, so the tables can be retuned.
        let fen_w = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let fen_b = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1";
        assert!((850..=950).contains(&eval(fen_w)), "got {}", eval(fen_w));
        assert_eq!(eval(fen_w), -eval(fen_b));
    }

    #[test]
    fn score_is_side_to_move_relative() {
        // A lone white rook vs bare kings (kings on mirror squares, rook on a1
        // with a zero PST entry) is +500 for White, -500 for Black — the same
        // position scored from whichever seat is to move.
        assert_eq!(eval("4k3/8/8/8/8/8/8/R3K3 w - - 0 1"), 500);
        assert_eq!(eval("4k3/8/8/8/8/8/8/R3K3 b - - 0 1"), -500);
    }

    #[test]
    fn color_mirror_is_symmetric() {
        // Mirroring the whole position (colors + ranks + side to move) must yield
        // the identical side-to-move-relative score. This holds for ANY tables,
        // because Black reads them mirrored — it's the core PST invariant.
        let white_knight_up = "4k3/8/8/8/8/8/8/N3K3 w - - 0 1";
        let black_knight_up = "n3k3/8/8/8/8/8/8/4K3 b - - 0 1";
        assert_eq!(eval(white_knight_up), eval(black_knight_up));
    }

    #[test]
    fn centralized_knight_beats_a_cornered_one() {
        // The whole point of PSTs: a knight on e4 is worth more than one rotting
        // on a1, even though the material is identical.
        let central = eval("4k3/8/8/8/4N3/8/8/4K3 w - - 0 1");
        let cornered = eval("4k3/8/8/8/8/8/8/N3K3 w - - 0 1");
        assert!(central > cornered, "central {central} should beat cornered {cornered}");
    }
}
