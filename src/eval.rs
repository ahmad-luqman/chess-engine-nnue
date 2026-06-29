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

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::movegen::{
    bishop_attacks, king_attacks, king_square, knight_attacks, queen_attacks, rook_attacks,
};
use crate::types::{Color, PieceType, Square};

/// A packed middlegame/endgame score pair. Terms accumulate as `Score`s and are
/// collapsed to a single centipawn number by [`Score::taper`] once the game phase
/// is known. A plain struct (not bit-packing) — the optimiser folds the two fields
/// and it keeps #41/#42 readable.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Score {
    pub mg: i32,
    pub eg: i32,
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
    for pt in
        [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen]
    {
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

    // Hand-crafted positional terms (issue #41), each isolated so Texel (#42) can
    // tune its weight independently.
    total += bishop_pair(board, color);
    total += rook_files(board, color);
    total += mobility(board, color);
    total += pawn_structure(board, color);
    total += king_safety(board, color);

    total
}

// ── Hand-crafted terms (issue #41) ──────────────────────────────────────────
//
// Each term returns a tapered `(mg, eg)` [`Score`] *for `color`* (positive = good
// for that side), accumulated into [`side_score`]. Weights here are textbook
// starting points; Texel tuning (#42) optimises them. New terms are added one at a
// time, each SPRT-gated, so a regression is attributable to a single term.

/// Bonus for holding the **bishop pair**: two bishops cover both colour complexes
/// and are worth more than the sum of the parts. Awarded once a side has ≥ 2
/// bishops (the common case is exactly two; more only arise via promotion).
/// (Texel-tuned, #42.)
pub const BISHOP_PAIR: Score = Score { mg: 53, eg: 54 };

fn bishop_pair(board: &Board, color: Color) -> Score {
    let bishops = board.pieces(PieceType::Bishop).intersect(board.color(color));
    if bishops.count() >= 2 {
        BISHOP_PAIR
    } else {
        Score::default()
    }
}

/// Bonus for a rook on an **open** file (no pawn of either colour) or a
/// **semi-open** file (no *friendly* pawn, but an enemy pawn present). Open files
/// are a rook's highways into the enemy position; the bonus is larger in the
/// middlegame, when there is more along the file to pressure.
pub const ROOK_OPEN_FILE: Score = Score { mg: 69, eg: -11 };
pub const ROOK_SEMI_OPEN_FILE: Score = Score { mg: 27, eg: 11 };

fn rook_files(board: &Board, color: Color) -> Score {
    let own_pawns = board.pieces(PieceType::Pawn).intersect(board.color(color));
    let enemy_pawns = board.pieces(PieceType::Pawn).intersect(board.color(color.flip()));
    let mut total = Score::default();
    let mut rooks = board.pieces(PieceType::Rook).intersect(board.color(color));
    while let Some(sq) = rooks.pop_lsb() {
        let file = FILE_MASKS[sq.file() as usize];
        if own_pawns.intersect(file).is_empty() {
            total += if enemy_pawns.intersect(file).is_empty() {
                ROOK_OPEN_FILE
            } else {
                ROOK_SEMI_OPEN_FILE
            };
        }
    }
    total
}

/// Mobility: a small bonus per square a piece attacks that isn't occupied by one
/// of its own pieces. More squares ≈ a more active piece. Weighted per piece type
/// and phase — rooks value lines more in the endgame; the queen is weighted lightly
/// so its huge reach doesn't swamp the term. This is cheap *pseudo*-mobility (it
/// doesn't exclude enemy-controlled squares); eval runs at every qsearch leaf, so
/// the attack lookups are kept to one per piece. Indexed by [`PieceType::index`].
pub const MOBILITY: [Score; 6] = [
    Score { mg: 0, eg: 0 },  // Pawn — not counted
    Score { mg: 2, eg: -8 }, // Knight
    Score { mg: 7, eg: 2 },  // Bishop
    Score { mg: 2, eg: 6 },  // Rook
    Score { mg: 1, eg: 10 }, // Queen
    Score { mg: 0, eg: 0 },  // King — not counted
];

fn mobility(board: &Board, color: Color) -> Score {
    let own = board.color(color);
    let occ = board.occupied();
    let mut total = Score::default();
    for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
        let w = MOBILITY[pt.index()];
        let mut bb = board.pieces(pt).intersect(own);
        while let Some(sq) = bb.pop_lsb() {
            let attacks = match pt {
                PieceType::Knight => knight_attacks(sq),
                PieceType::Bishop => bishop_attacks(sq, occ),
                PieceType::Rook => rook_attacks(sq, occ),
                PieceType::Queen => queen_attacks(sq, occ),
                _ => unreachable!(),
            };
            let n = attacks.minus(own).count() as i32;
            total += Score { mg: w.mg * n, eg: w.eg * n };
        }
    }
    total
}

/// Pawn-structure penalties and bonuses. **Doubled** (two+ pawns on a file) and
/// **isolated** (no friendly pawn on an adjacent file) are weaknesses — they can't
/// defend each other and are easy to blockade — so they score negative.
/// **Passed** pawns (no enemy pawn ahead on the same or adjacent files) are
/// strong, more so the closer to promotion and the deeper into the endgame, so the
/// bonus rises with rank and tapers toward `eg`.
pub const DOUBLED_PAWN: Score = Score { mg: 5, eg: -20 };
pub const ISOLATED_PAWN: Score = Score { mg: -25, eg: -8 };
/// Passed-pawn bonus indexed by the pawn's rank *from its own side* (0 = home rank,
/// 6 = one step from promotion). Index 0 and 7 are unreachable for a pawn. Texel
/// tuning (#42) put almost all the value in the endgame (`eg`) — a passed pawn is
/// a promotion threat once the heavy pieces are gone — and near zero in the
/// middlegame, where it can equally be a target.
pub const PASSED_PAWN_MG: [i32; 8] = [0, 5, -4, -11, 14, 31, 37, 0];
pub const PASSED_PAWN_EG: [i32; 8] = [0, 9, 12, 41, 67, 148, 196, 0];

fn pawn_structure(board: &Board, color: Color) -> Score {
    let own_pawns = board.pieces(PieceType::Pawn).intersect(board.color(color));
    let enemy_pawns = board.pieces(PieceType::Pawn).intersect(board.color(color.flip()));
    let mut total = Score::default();

    // Doubled: every pawn beyond the first on a file is a doubled pawn.
    for file in FILE_MASKS {
        let n = own_pawns.intersect(file).count() as i32;
        if n > 1 {
            total += Score { mg: DOUBLED_PAWN.mg * (n - 1), eg: DOUBLED_PAWN.eg * (n - 1) };
        }
    }

    let mut pawns = own_pawns;
    while let Some(sq) = pawns.pop_lsb() {
        // Isolated: no friendly pawn on either adjacent file (the pawn's own file
        // is excluded from the mask, so it never matches itself).
        if own_pawns.intersect(ADJACENT_FILE_MASKS[sq.file() as usize]).is_empty() {
            total += ISOLATED_PAWN;
        }
        // Passed: no enemy pawn on the same or adjacent files anywhere ahead.
        if enemy_pawns.intersect(PASSED_MASKS[color.index()][sq.0 as usize]).is_empty() {
            let rel = match color {
                Color::White => sq.rank() as usize,
                Color::Black => 7 - sq.rank() as usize,
            };
            total += Score { mg: PASSED_PAWN_MG[rel], eg: PASSED_PAWN_EG[rel] };
        }
    }
    total
}

/// King safety: a **middlegame-only** penalty for enemy pressure on the king's
/// neighbourhood. Two cheap, well-behaved signals (kept conservative — an
/// over-eager king-safety term is the classic source of eval regressions; Texel
/// #42 can sharpen it): **zone pressure** (a weighted count of enemy pieces
/// attacking the king ring) and **shield holes** (files beside the king, king
/// file ± 1, with no friendly pawn — open avenues toward the king).
///
/// `eg == 0`: in the endgame the king is a fighting piece, not a target (the
/// counterpart to the centralising [`KING_EG`] table).
///
/// The per-piece weights are in **centipawns directly** (the old design multiplied
/// raw "attack units" by a separate `KING_DANGER_PER_UNIT` scalar; folding that
/// constant in makes the term *linear* in its weights — `penalty = -Σ weight·count`
/// — which Texel tuning #42 requires, and is exactly score-preserving).
pub const KING_ATTACK_WEIGHT: [i32; 6] = [0, 20, 21, 14, 13, 0]; // P,N,B,R,Q,K (centipawns, Texel-tuned #42)
pub const PAWN_SHIELD_HOLE: i32 = 35;

fn king_safety(board: &Board, color: Color) -> Score {
    let king_sq = king_square(board, color);
    let zone = king_attacks(king_sq).with(king_sq);
    let occ = board.occupied();
    let enemy = color.flip();

    // Zone pressure: each enemy piece that attacks into the king ring adds its
    // weight (in centipawns) to the danger total.
    let mut units = 0;
    for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
        let mut bb = board.pieces(pt).intersect(board.color(enemy));
        while let Some(sq) = bb.pop_lsb() {
            let attacks = match pt {
                PieceType::Knight => knight_attacks(sq),
                PieceType::Bishop => bishop_attacks(sq, occ),
                PieceType::Rook => rook_attacks(sq, occ),
                PieceType::Queen => queen_attacks(sq, occ),
                _ => unreachable!(),
            };
            if !attacks.intersect(zone).is_empty() {
                units += KING_ATTACK_WEIGHT[pt.index()];
            }
        }
    }

    // Shield holes: open files immediately around the king.
    let own_pawns = board.pieces(PieceType::Pawn).intersect(board.color(color));
    let kf = king_sq.file() as usize;
    let mut holes = 0;
    for file in &FILE_MASKS[kf.saturating_sub(1)..=(kf + 1).min(7)] {
        if own_pawns.intersect(*file).is_empty() {
            holes += 1;
        }
    }

    Score { mg: -(units + holes * PAWN_SHIELD_HOLE), eg: 0 }
}

// ── File / pawn masks (issue #41) ───────────────────────────────────────────
//
// Precomputed at compile time via `const fn` (no runtime cost, no init step).
// Square index is `rank * 8 + file` with a1 = 0 (see `src/bitboard.rs`).

/// `FILE_MASKS[f]` is every square on file `f` (a = 0 … h = 7). Used for rook
/// open/semi-open files and for doubled/isolated pawn detection (#41).
pub const FILE_MASKS: [Bitboard; 8] = {
    let mut masks = [Bitboard(0); 8];
    let mut f = 0;
    while f < 8 {
        let mut bb = 0u64;
        let mut r = 0;
        while r < 8 {
            bb |= 1u64 << (r * 8 + f);
            r += 1;
        }
        masks[f] = Bitboard(bb);
        f += 1;
    }
    masks
};

/// `ADJACENT_FILE_MASKS[f]` is the files immediately left and right of `f` (not `f`
/// itself). A pawn on file `f` is **isolated** when no friendly pawn intersects it.
pub const ADJACENT_FILE_MASKS: [Bitboard; 8] = {
    let mut m = [Bitboard(0); 8];
    let mut f = 0;
    while f < 8 {
        let mut bb = 0u64;
        if f > 0 {
            bb |= FILE_MASKS[f - 1].0;
        }
        if f < 7 {
            bb |= FILE_MASKS[f + 1].0;
        }
        m[f] = Bitboard(bb);
        f += 1;
    }
    m
};

/// `PASSED_MASKS[color][sq]` is every square on `sq`'s file and the two adjacent
/// files that lies **strictly ahead** of `sq` from `color`'s direction of travel
/// (White toward rank 8, Black toward rank 1). A pawn is **passed** when no enemy
/// pawn occupies its mask — nothing can block it or capture it on the way to
/// promotion. Indexed `[Color::index()][Square::0]`.
pub const PASSED_MASKS: [[Bitboard; 64]; 2] = {
    let mut masks = [[Bitboard(0); 64]; 2];
    let mut sq = 0;
    while sq < 64 {
        let file = sq % 8;
        let rank = sq / 8;
        let lo = if file == 0 { 0 } else { file - 1 };
        let hi = if file == 7 { 7 } else { file + 1 };
        let mut white = 0u64;
        let mut black = 0u64;
        let mut f = lo;
        while f <= hi {
            // White: every rank above this one. Black: every rank below.
            let mut r = rank + 1;
            while r < 8 {
                white |= 1u64 << (r * 8 + f);
                r += 1;
            }
            let mut r2 = 0;
            while r2 < rank {
                black |= 1u64 << (r2 * 8 + f);
                r2 += 1;
            }
            f += 1;
        }
        masks[0][sq] = Bitboard(white);
        masks[1][sq] = Bitboard(black);
        sq += 1;
    }
    masks
};

/// Full-material game phase, the highest value [`game_phase`] returns.
pub const PHASE_MAX: i32 = 24;

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
/// Originally Tomasz Michniewski's "Simplified Evaluation Function" tables, now
/// **Texel-tuned** (#42) against ~200k self-play quiet positions. They are written
/// from White's perspective; Black reuses them mirrored (again, see [`pst_index`]).
/// The same table serves both game phases for these pieces (so their `mg == eg`);
/// only the king splits into [`KING_MG`] / [`KING_EG`] (issue #40), and the king
/// tables were held fixed during tuning (too few king samples per position).
///
/// `rustfmt::skip` keeps each table laid out as a readable 8×8 board (rank per
/// line); without it rustfmt collapses them into an unreadable flat run.
#[rustfmt::skip]
pub const PIECE_SQUARE_TABLE: [[i32; 64]; 5] = [
    // Pawn
    [
           0,    0,    0,    0,    0,    0,    0,    0,
         102,   93,   65,   48,   50,   53,   60,   87,
          53,   46,   32,    3,    3,   28,   37,   39,
          31,   24,   16,   13,   15,   14,   18,   15,
          14,   10,   11,   14,   14,    5,    4,   -1,
          10,    6,    9,    1,   11,    9,    7,    0,
           7,    6,   -3,  -13,   -3,   19,   13,   -5,
           0,    0,    0,    0,    0,    0,    0,    0,
    ],
    // Knight
    [
         -44,   30,   69,   49,   74,   33,   18,  -50,
          34,   71,  114,  113,  101,   99,   69,   28,
          55,  113,  137,  146,  144,  149,  116,   68,
          79,  117,  140,  151,  134,  151,  111,   91,
          77,  107,  134,  134,  139,  132,  116,   84,
          70,   96,  117,  131,  134,  128,  114,   77,
          57,   59,   92,  110,  111,  105,   72,   69,
          20,   72,   64,   72,   79,   74,   75,   29,
    ],
    // Bishop
    [
          32,   45,   27,   42,   50,   40,   41,   33,
          41,   54,   51,   42,   58,   67,   56,   21,
          49,   64,   67,   63,   59,   69,   72,   55,
          57,   60,   69,   75,   72,   68,   57,   56,
          55,   62,   61,   77,   70,   57,   58,   60,
          52,   68,   73,   65,   71,   75,   69,   54,
          56,   72,   65,   63,   72,   68,   86,   49,
          36,   59,   53,   55,   55,   56,   46,   40,
    ],
    // Rook
    [
         133,  131,  134,  133,  134,  128,  132,  126,
         136,  138,  141,  140,  134,  140,  135,  135,
         127,  131,  126,  127,  119,  125,  129,  119,
         122,  119,  129,  120,  122,  129,  118,  121,
         112,  121,  123,  119,  119,  113,  121,  108,
         106,  114,  112,  114,  115,  115,  116,   99,
         103,  116,  116,  121,  118,  121,  111,   85,
         118,  120,  124,  126,  125,  134,  103,  110,
    ],
    // Queen
    [
         277,  309,  317,  313,  341,  332,  327,  341,
         272,  261,  299,  317,  314,  336,  331,  347,
         286,  283,  292,  312,  333,  347,  349,  352,
         288,  280,  291,  287,  313,  317,  328,  313,
         293,  288,  296,  298,  305,  308,  315,  310,
         290,  303,  305,  306,  308,  310,  321,  314,
         282,  297,  316,  315,  324,  316,  289,  310,
         299,  290,  300,  321,  303,  294,  280,  258,
    ],
];

/// King table, **middlegame**: stay home and castled; the centre is lethal while
/// queens and rooks are on. Same Michniewski source and rank-8-first layout as
/// [`PIECE_SQUARE_TABLE`]; read into a [`Score`]'s `mg` field.
#[rustfmt::skip]
pub const KING_MG: [i32; 64] = [
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
pub const KING_EG: [i32; 64] = [
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
        // Startpos but Black is missing its queen. Material dwarfs any PST wobble,
        // so White to move is up close to a queen. Texel tuning (#42) put the
        // queen's effective value (material + PST) a bit above its fixed 900, so
        // the window is wide — the point is "≈ a queen up", not an exact number.
        let fen_w = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let fen_b = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1";
        assert!((1100..=1300).contains(&eval(fen_w)), "got {}", eval(fen_w));
        assert_eq!(eval(fen_w), -eval(fen_b));
    }

    #[test]
    fn score_is_side_to_move_relative() {
        // The same position scored from each seat must negate — that sign-flip is
        // the invariant. The exact magnitude is incidental (material + PST + the
        // positional terms; the lone rook here also sits on an open file), so we
        // assert symmetry plus a clearly-winning score rather than an exact number.
        let w = eval("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
        let b = eval("4k3/8/8/8/8/8/8/R3K3 b - - 0 1");
        assert_eq!(w, -b, "side-to-move score must negate between seats");
        assert!(w > 400, "white up a rook should be clearly winning, got {w}");
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

    // ── Hand-crafted terms (issue #41) ──────────────────────────────────────

    #[test]
    fn bishop_pair_beats_no_pair_all_else_equal() {
        // Two bishops vs bishop+knight: same material (B≈N here) and mirrored
        // placement, so the *only* difference is the pair bonus. The side holding
        // both bishops must score better. White has two bishops (c1, f1); Black has
        // a knight on b8 instead of one bishop.
        let pair = eval("rn1qkb1r/pppppppp/8/8/8/8/PPPPPPPP/R1BQKB1R w KQkq - 0 1");
        // Mirror: now White also has a knight instead of the c1 bishop → no pair.
        let no_pair = eval("rn1qkb1r/pppppppp/8/8/8/8/PPPPPPPP/RN1QKB1R w KQkq - 0 1");
        assert!(pair > no_pair, "bishop pair {pair} should beat no pair {no_pair}");
    }

    #[test]
    fn bishop_pair_term_is_a_nonnegative_bonus() {
        // Direct sign/structure check: the term is 0 with one bishop and positive
        // with two, never a penalty.
        let one = Board::from_str("4k3/8/8/8/8/8/8/2B1K3 w - - 0 1").unwrap();
        let two = Board::from_str("4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1").unwrap();
        assert_eq!(bishop_pair(&one, Color::White), Score::default());
        assert_eq!(bishop_pair(&two, Color::White), BISHOP_PAIR);
    }

    #[test]
    fn file_masks_cover_each_file() {
        // a-file is squares a1..a8 = bits 0,8,16,…,56.
        assert_eq!(FILE_MASKS[0], Bitboard(0x0101_0101_0101_0101));
        // h-file is the a-file shifted left 7.
        assert_eq!(FILE_MASKS[7], Bitboard(0x8080_8080_8080_8080));
        // Each file has exactly 8 squares and they partition the board.
        let mut all = Bitboard(0);
        for file in FILE_MASKS {
            assert_eq!(file.count(), 8);
            all = all.union(file);
        }
        assert_eq!(all, Bitboard(u64::MAX));
    }

    #[test]
    fn rook_on_open_file_beats_a_blocked_rook() {
        // White rook on the open d-file (no pawns on it) vs a rook on the closed
        // a-file (own pawn on a2). Same material; the open-file bonus must make the
        // open-file placement score higher.
        let open = eval("4k3/8/8/8/8/8/P7/3RK3 w - - 0 1");
        let blocked = eval("4k3/8/8/8/8/8/P7/R3K3 w - - 0 1");
        assert!(open > blocked, "rook on open file {open} should beat blocked {blocked}");
    }

    #[test]
    fn rook_file_bonus_grades_open_over_semi_over_closed() {
        // A single white rook on the d-file under three pawn configurations:
        //   open      — no pawns on d
        //   semi-open — only an enemy (black) pawn on d
        //   closed    — a friendly (white) pawn on d
        let open =
            rook_files(&Board::from_str("4k3/8/8/8/8/8/8/3RK3 w - - 0 1").unwrap(), Color::White);
        let semi =
            rook_files(&Board::from_str("4k3/3p4/8/8/8/8/8/3RK3 w - - 0 1").unwrap(), Color::White);
        let closed =
            rook_files(&Board::from_str("4k3/8/8/8/8/8/3P4/3RK3 w - - 0 1").unwrap(), Color::White);
        assert_eq!(open, ROOK_OPEN_FILE);
        assert_eq!(semi, ROOK_SEMI_OPEN_FILE);
        assert_eq!(closed, Score::default());
    }

    #[test]
    fn mobility_rewards_active_pieces() {
        // A rook in the open centre reaches far more squares than one boxed into a
        // corner behind its own king and pawns. Test mobility() directly so the
        // result is isolated from PST and material.
        let active =
            mobility(&Board::from_str("4k3/8/8/8/3R4/8/8/4K3 w - - 0 1").unwrap(), Color::White);
        let boxed =
            mobility(&Board::from_str("4k3/8/8/8/8/8/PP6/KR6 w - - 0 1").unwrap(), Color::White);
        assert!(active.mg > boxed.mg, "active rook {active:?} should out-move boxed {boxed:?}");
        assert!(active.eg > boxed.eg);
    }

    fn sq(s: &str) -> Square {
        s.parse().unwrap()
    }

    #[test]
    fn passed_and_adjacent_masks_are_exact() {
        // The off-by-one magnet: same + adjacent files, strictly ahead, per colour.
        // White pawn on e4 → d5–d8, e5–e8, f5–f8 (12 squares); nothing on rank ≤ 4
        // and nothing off the d/e/f files.
        let m = PASSED_MASKS[Color::White.index()][sq("e4").0 as usize];
        assert_eq!(m.count(), 12);
        for s in ["d5", "e5", "f5", "d8", "e8", "f8"] {
            assert!(m.contains(sq(s)), "white passed mask for e4 should include {s}");
        }
        for s in ["e4", "e3", "e2", "c5", "g5"] {
            assert!(!m.contains(sq(s)), "white passed mask for e4 should exclude {s}");
        }
        // Black travels the other way: e5 → d1–d4, e1–e4, f1–f4.
        let mb = PASSED_MASKS[Color::Black.index()][sq("e5").0 as usize];
        assert_eq!(mb.count(), 12);
        assert!(mb.contains(sq("e4")) && mb.contains(sq("d1")));
        assert!(!mb.contains(sq("e6")) && !mb.contains(sq("e5")));
        // Adjacent files of e are d and f only (not e).
        assert_eq!(
            ADJACENT_FILE_MASKS[sq("e4").file() as usize],
            FILE_MASKS[3].union(FILE_MASKS[5])
        );
    }

    #[test]
    fn isolated_pawn_is_a_penalty() {
        // White d4, blocked and flanked by black c5/d5/e5: not passed (enemy ahead),
        // not doubled, and isolated (no friendly c/e pawn). So the term is exactly
        // the isolated penalty — a direct sign + detection check.
        let b = Board::from_str("4k3/8/8/2ppp3/3P4/8/8/4K3 w - - 0 1").unwrap();
        let s = pawn_structure(&b, Color::White);
        assert_eq!(s, ISOLATED_PAWN);
        // A penalty (Texel-tuned #42: weak in both phases — harder to defend).
        assert!(s.mg < 0 && s.eg < 0, "isolated pawn must be a penalty, got {s:?}");
    }

    #[test]
    fn doubled_pawns_score_worse_than_a_single_pawn() {
        // a2+a3 (doubled) vs a3 alone, both with a black a7 pawn ahead so *neither*
        // is passed — that isolates the doubled penalty from the passed-pawn bonus
        // (which otherwise differs between the two and confounds the comparison).
        // Both a-pawns are isolated in both positions, so only the doubling differs.
        let doubled = pawn_structure(
            &Board::from_str("k7/p7/8/8/8/P7/P7/K7 w - - 0 1").unwrap(),
            Color::White,
        );
        let single = pawn_structure(
            &Board::from_str("k7/p7/8/8/8/8/P7/K7 w - - 0 1").unwrap(),
            Color::White,
        );
        assert!(
            doubled.mg < single.mg,
            "doubled {doubled:?} should be worse than single {single:?}"
        );
        assert!(doubled.eg < single.eg);
    }

    #[test]
    fn passed_pawn_beats_a_blocked_pawn() {
        // White e5 with a clear path (passed) vs a black pawn on e6 ahead (blocked,
        // not passed). The passed pawn must score higher, more so in the endgame.
        let passed = pawn_structure(
            &Board::from_str("4k3/8/8/4P3/8/8/8/4K3 w - - 0 1").unwrap(),
            Color::White,
        );
        let blocked = pawn_structure(
            &Board::from_str("4k3/4p3/4P3/8/8/8/8/4K3 w - - 0 1").unwrap(),
            Color::White,
        );
        assert!(passed.eg > blocked.eg, "passed {passed:?} should beat blocked {blocked:?}");
    }

    #[test]
    fn exposed_king_is_penalised() {
        // Safe king tucked on g1 behind f2/g2/h2 with no enemy pressure → no
        // penalty. Exposed king on e4 in the open with a black queen + rook bearing
        // down the d/e files and no shield → a clear middlegame penalty.
        let safe = king_safety(
            &Board::from_str("4k3/8/8/8/8/8/5PPP/6K1 w - - 0 1").unwrap(),
            Color::White,
        );
        let exposed = king_safety(
            &Board::from_str("3qr3/8/8/8/4K3/8/8/4k3 w - - 0 1").unwrap(),
            Color::White,
        );
        assert_eq!(
            safe,
            Score::default(),
            "a sheltered king with no pressure should be unpenalised"
        );
        assert!(
            exposed.mg < safe.mg,
            "exposed king {exposed:?} should be worse than safe {safe:?}"
        );
        assert_eq!(exposed.eg, 0, "king safety is middlegame-only");
    }
}
