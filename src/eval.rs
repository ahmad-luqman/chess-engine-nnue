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
//!
//! ## Tapered evaluation (issue #40)
//!
//! A single king table tells the king to hide on the back rank — correct in the
//! middlegame, wrong in a king-and-pawn ending where it should march to the
//! centre. So every term now carries **two** scores, [`Score`]`{ mg, eg }`, and
//! the final number interpolates between them by a **game phase** read off the
//! remaining non-pawn material: full material → all `mg`, bare board → all `eg`.
//! For now only the king has distinct tables ([`KING_MG`] / [`KING_EG`]); the
//! other pieces share one table (so `mg == eg`) and material is phase-independent.
//! The `(mg, eg)` pair is the unit that hand-crafted terms (#41) and Texel tuning
//! (#42) extend and optimise.

use std::ops::{Add, AddAssign, Neg, Sub};

use crate::board::Board;
use crate::types::{Color, PieceType, Square};

/// A packed middlegame/endgame score pair. Terms accumulate as `Score`s and are
/// collapsed to a single centipawn number by [`Score::taper`] once the game phase
/// is known. A plain struct (not bit-packing) — the optimiser folds the two fields
/// and it keeps #41/#42 readable.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
struct Score {
    mg: i32,
    eg: i32,
}

impl Score {
    /// Interpolate between the middlegame and endgame scores. `phase` runs from
    /// `PHASE_MAX` (24, full material → pure `mg`) down to 0 (bare → pure `eg`);
    /// see [`game_phase`].
    fn taper(self, phase: i32) -> i32 {
        (self.mg * phase + self.eg * (PHASE_MAX - phase)) / PHASE_MAX
    }
}

impl Add for Score {
    type Output = Score;
    fn add(self, rhs: Score) -> Score {
        Score { mg: self.mg + rhs.mg, eg: self.eg + rhs.eg }
    }
}

impl AddAssign for Score {
    fn add_assign(&mut self, rhs: Score) {
        self.mg += rhs.mg;
        self.eg += rhs.eg;
    }
}

impl Sub for Score {
    type Output = Score;
    fn sub(self, rhs: Score) -> Score {
        Score { mg: self.mg - rhs.mg, eg: self.eg - rhs.eg }
    }
}

impl Neg for Score {
    type Output = Score;
    fn neg(self) -> Score {
        Score { mg: -self.mg, eg: -self.eg }
    }
}

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
        // Sum each side's (mg, eg) material + PST score white-relative (White
        // positive, Black negative), then collapse to a single number for the
        // current game phase, then flip to side-to-move-relative. Computing
        // white-relative first keeps the arithmetic sign-free.
        let white_relative = side_score(board, Color::White) - side_score(board, Color::Black);
        let score = white_relative.taper(game_phase(board));

        match board.side_to_move {
            Color::White => score,
            Color::Black => -score,
        }
    }
}

/// Total static [`Score`] of `color`'s pieces: material value plus the piece-square
/// bonus for each piece's square. Non-king pieces share one table, so their `mg`
/// and `eg` are equal; the king reads the middlegame table into `mg` and the
/// centralising endgame table into `eg` (its material is zero).
fn side_score(board: &Board, color: Color) -> Score {
    let own = board.color(color);
    let mut total = Score::default();
    for pt in [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
    ] {
        let table = &PIECE_SQUARE_TABLE[pt.index()];
        let value = PIECE_VALUE[pt.index()];
        let mut bb = board.pieces(pt).intersect(own);
        while let Some(sq) = bb.pop_lsb() {
            let s = value + table[pst_index(color, sq)];
            total += Score { mg: s, eg: s };
        }
    }
    // The king carries no material but its placement flips by phase: tucked away
    // in the middlegame (KING_MG), centralised in the endgame (KING_EG).
    let mut kings = board.pieces(PieceType::King).intersect(own);
    while let Some(sq) = kings.pop_lsb() {
        let idx = pst_index(color, sq);
        total += Score { mg: KING_MG[idx], eg: KING_EG[idx] };
    }
    total
}

/// Full-material game phase, the highest value [`game_phase`] returns.
const PHASE_MAX: i32 = 24;

/// The game phase in `[0, PHASE_MAX]`: a measure of how much non-pawn material is
/// left, summed over **both** sides with the classic weights (Q=4, R=2, B=N=1).
/// The starting position totals 24 (two queens, four rooks, eight minors); a bare
/// king-and-pawn ending totals 0. Clamped because promotions can briefly exceed
/// the starting material. This is the knob that slides the eval from `mg` to `eg`.
fn game_phase(board: &Board) -> i32 {
    let phase = 4 * board.pieces(PieceType::Queen).count() as i32
        + 2 * board.pieces(PieceType::Rook).count() as i32
        + board.pieces(PieceType::Bishop).count() as i32
        + board.pieces(PieceType::Knight).count() as i32;
    phase.min(PHASE_MAX)
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

/// Piece-square tables for the five non-king pieces, indexed by
/// [`PieceType::index`] (Pawn=0 … Queen=4) then by board square in
/// **rank-8-first** layout (index 0 = a8 … 63 = h1) — see [`pst_index`].
///
/// These are Tomasz Michniewski's "Simplified Evaluation Function" tables, a
/// well-known public-domain starting point. They are written from White's
/// perspective; Black reuses them mirrored (again, see [`pst_index`]). The same
/// table serves both game phases for these pieces (so their `mg == eg`); only the
/// king splits into [`KING_MG`] / [`KING_EG`] (issue #40).
///
/// `rustfmt::skip` keeps each table laid out as a readable 8×8 board (rank per
/// line); without it rustfmt collapses them into an unreadable flat run.
#[rustfmt::skip]
const PIECE_SQUARE_TABLE: [[i32; 64]; 5] = [
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
];

/// King table, **middlegame**: stay home and castled; the centre is lethal while
/// queens and rooks are on. Same Michniewski source and rank-8-first layout as
/// [`PIECE_SQUARE_TABLE`]; read into a [`Score`]'s `mg` field.
#[rustfmt::skip]
const KING_MG: [i32; 64] = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
];

/// King table, **endgame**: with the heavy pieces gone the king is a fighting
/// piece — march it to the centre and keep it off the edges and corners. Read
/// into a [`Score`]'s `eg` field; tapering (issue #40) blends it in as material
/// leaves the board. (Michniewski's endgame king table.)
#[rustfmt::skip]
const KING_EG: [i32; 64] = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
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

    // ── Tapered evaluation (issue #40) ──────────────────────────────────────

    #[test]
    fn endgame_king_prefers_the_centre() {
        // The motivating case: in a king-and-pawn ending (phase 0, pure endgame),
        // a central king is better than one stuck in the corner. The two positions
        // are identical but for the white king (e4 vs a1) — the black king (h8) and
        // white pawn (e2) cancel — so the only difference is KING_EG. The old
        // single (middlegame) table gets this backwards: it rewards the corner.
        let central = eval("7k/8/8/8/4K3/8/4P3/8 w - - 0 1");
        let cornered = eval("7k/8/8/8/8/8/4P3/K7 w - - 0 1");
        assert!(central > cornered, "central king {central} should beat cornered {cornered}");
    }

    #[test]
    fn game_phase_runs_from_full_material_to_bare() {
        fn phase(fen: &str) -> i32 {
            game_phase(&Board::from_str(fen).unwrap())
        }
        // Starting position has all the heavy/minor pieces → the maximum phase.
        assert_eq!(phase("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), PHASE_MAX);
        // King-and-pawn ending: no non-pawn material at all → pure endgame.
        assert_eq!(phase("4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1"), 0);
        // The weights: a single queen (4) + single rook (2) per side = 12.
        assert_eq!(phase("4k2r/3q4/8/8/8/8/3Q4/R3K3 w - - 0 1"), 12);
        // Promotions can over-fill the board; the phase must clamp, not overflow.
        assert_eq!(phase("QQQQkQQQ/QQQQQQQQ/8/8/8/8/8/4K3 w - - 0 1"), PHASE_MAX);
    }

    #[test]
    fn taper_interpolates_between_mg_and_eg() {
        let s = Score { mg: 100, eg: -20 };
        assert_eq!(s.taper(PHASE_MAX), 100); // full material → pure middlegame
        assert_eq!(s.taper(0), -20); //         bare board     → pure endgame
        assert_eq!(s.taper(12), (100 * 12 + -20 * 12) / 24); // halfway blend
    }
}
