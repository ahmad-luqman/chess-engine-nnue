//! NNUE inference: quantized perspective accumulator + incremental update (#46).
//!
//! This is the heart of the neural eval — a fast, integer, incrementally-updated
//! forward pass that slots behind the [`Evaluator`](crate::eval::Evaluator) trait
//! in place of the hand-crafted [`Material`](crate::eval::Material) eval.
//!
//! The architecture, quantisation constants, net byte layout, and the exact
//! inference arithmetic are fixed by **ADR 0016** (this module must match it
//! bit-for-bit) and the implementation decisions here by **ADR 0017**. The trainer
//! (`trainer/`, issue #45) produced the embedded net; `trainer/verify` is the
//! bullet-validated reference these tests pin against.
//!
//! Shape: plain `(768 → 256)×2 → 1×8`, SCReLU, `Chess768` perspective inputs.
//! - **Two accumulators**, one per *perspective colour* (White, Black). Each is a
//!   running sum of the active feature columns, kept in `i16`. Storing by colour
//!   (not by side-to-move) is what makes the update side-agnostic and incremental:
//!   a piece toggle adds/subtracts the same column regardless of whose turn it is.
//! - **Incremental update** (the whole point of NNUE): on a move, only a few
//!   `(piece, square)` features change, so we add/subtract a handful of columns
//!   from the [`DirtyPiece`] deltas (#43) rather than recomputing. Search (#48)
//!   drives this via [`Accumulator::apply`]/[`revert`](Accumulator::revert) over
//!   the make/unmake stack; this issue (#46) only proves it correct.
//! - **Forward pass**: concatenate the side-to-move accumulator **first**, the
//!   opponent **second**; apply SCReLU `clamp(x, 0, QA)²` (squared/accumulated in
//!   `i32` — `QA² = 65025` overflows `i16`); run the material-bucketed output
//!   layer; dequantise to centipawns.

use std::sync::OnceLock;

use crate::board::{Board, DirtyPiece};
use crate::eval::Evaluator;
use crate::types::{Color, Piece, Square};

/// Hidden-layer width per perspective.
const HL: usize = 256;
/// Number of material-count output buckets.
const BUCKETS: usize = 8;
/// Accumulator-weight quantisation factor.
const QA: i32 = 255;
/// Output-weight quantisation factor.
const QB: i32 = 64;
/// Eval scale (centipawns) applied during dequantisation.
const SCALE: i32 = 400;
/// `768 = 2 colours × 6 pieces × 64 squares`.
const N_FEATURES: usize = 768;

/// The committed quantised net (#45), embedded into the binary. Layout = ADR 0016:
/// little-endian `i16` — feature weights `[768][256]`, feature bias `[256]`, output
/// weights `[8][512]` (bucket-major, stm half first), output bias `[8]`, then
/// zero padding to a multiple of 64 bytes.
static NET_BYTES: &[u8] = include_bytes!("../nets/chess-engine-nnue-0001.bin");

/// The parsed network weights. Heap-resident (`Vec`s) so neither construction nor
/// the `'static` cache puts ~400 KB on the stack.
pub struct Network {
    /// Feature weights, indexed by feature → a column of `HL` accumulator deltas.
    feature_weights: Vec<[i16; HL]>,
    /// Shared accumulator bias (the from-scratch starting value).
    feature_bias: [i16; HL],
    /// Output weights, indexed by bucket. `0..HL` multiply the stm accumulator,
    /// `HL..2*HL` the opponent's.
    output_weights: Vec<[i16; 2 * HL]>,
    /// Output bias, indexed by bucket.
    output_bias: [i16; BUCKETS],
}

impl Network {
    /// Parse the embedded little-endian `i16` weights in ADR-0016 order.
    fn load() -> Network {
        let bytes = NET_BYTES;
        let need = (N_FEATURES * HL + HL + BUCKETS * 2 * HL + BUCKETS) * 2;
        assert!(bytes.len() >= need, "embedded net too small: {} < {need} bytes", bytes.len());

        let mut idx = 0usize;
        let mut next = || {
            let v = i16::from_le_bytes([bytes[idx], bytes[idx + 1]]);
            idx += 2;
            v
        };

        let mut feature_weights = Vec::with_capacity(N_FEATURES);
        for _ in 0..N_FEATURES {
            let mut col = [0i16; HL];
            for w in col.iter_mut() {
                *w = next();
            }
            feature_weights.push(col);
        }

        let mut feature_bias = [0i16; HL];
        for b in feature_bias.iter_mut() {
            *b = next();
        }

        let mut output_weights = Vec::with_capacity(BUCKETS);
        for _ in 0..BUCKETS {
            let mut row = [0i16; 2 * HL];
            for w in row.iter_mut() {
                *w = next();
            }
            output_weights.push(row);
        }

        let mut output_bias = [0i16; BUCKETS];
        for b in output_bias.iter_mut() {
            *b = next();
        }

        Network { feature_weights, feature_bias, output_weights, output_bias }
    }
}

/// The embedded network, parsed once on first use.
pub fn network() -> &'static Network {
    static NET: OnceLock<Network> = OnceLock::new();
    NET.get_or_init(Network::load)
}

/// The `Chess768` feature index for a piece, viewed from `perspective` (0 = White,
/// 1 = Black). Replicated from bullet's `Chess768::map_features` and cross-checked
/// against it (see `trainer/bullet-train/src/bin/featcheck.rs`): for the non-White
/// perspective the board is vertically mirrored (`sq ^ 56`) and colour is taken
/// relative to the viewer (friendly = 0, enemy = 1).
#[inline]
fn feature_index(perspective: usize, piece: Piece, sq: Square) -> usize {
    let color = piece.color.index();
    let pt = piece.piece_type.index();
    let sq = sq.0 as usize;
    let relative_color = usize::from(color != perspective);
    let relative_sq = if perspective == 0 { sq } else { sq ^ 56 };
    [0, 384][relative_color] + 64 * pt + relative_sq
}

/// SCReLU activation, returning `i32`: `QA² = 65025` overflows `i16`, so the clamp
/// and square must happen in `i32`.
#[inline]
fn screlu(x: i16) -> i32 {
    let v = i32::from(x).clamp(0, QA);
    v * v
}

/// Output bucket for a position with `piece_count` pieces on the board.
/// `MaterialCount<8>` ⇒ `(count − 2) / 4`, clamped to `0..=7`.
#[inline]
fn output_bucket(piece_count: u32) -> usize {
    (((piece_count as i32 - 2) / 4).clamp(0, BUCKETS as i32 - 1)) as usize
}

/// The two perspective accumulators (the first NNUE layer's output), one per
/// colour. Index `0` = White's view, `1` = Black's view. Maintained incrementally
/// across make/unmake, or rebuilt from scratch with [`Accumulator::from_board`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Accumulator {
    vals: [[i16; HL]; 2],
}

impl Accumulator {
    /// A fresh accumulator holding only the feature bias (no pieces added yet).
    pub fn new(net: &Network) -> Accumulator {
        Accumulator { vals: [net.feature_bias; 2] }
    }

    /// Build the accumulator for `board` from scratch: bias + every piece's column.
    pub fn from_board(board: &Board) -> Accumulator {
        let net = network();
        let mut acc = Accumulator::new(net);
        let mut occ = board.occupied();
        while let Some(sq) = occ.pop_lsb() {
            let piece = board.piece_on(sq).expect("occupied square has a piece");
            acc.add(piece, sq, net);
        }
        acc
    }

    /// Add a piece's feature column to both perspectives.
    #[inline]
    pub fn add(&mut self, piece: Piece, sq: Square, net: &Network) {
        for perspective in 0..2 {
            let col = &net.feature_weights[feature_index(perspective, piece, sq)];
            for (a, &w) in self.vals[perspective].iter_mut().zip(col) {
                *a += w;
            }
        }
    }

    /// Remove a piece's feature column from both perspectives.
    #[inline]
    pub fn remove(&mut self, piece: Piece, sq: Square, net: &Network) {
        for perspective in 0..2 {
            let col = &net.feature_weights[feature_index(perspective, piece, sq)];
            for (a, &w) in self.vals[perspective].iter_mut().zip(col) {
                *a -= w;
            }
        }
    }

    /// Apply a move's [`DirtyPiece`] deltas (the make direction): add the columns
    /// the move introduced, remove the ones it cleared.
    pub fn apply(&mut self, dirty: &DirtyPiece, net: &Network) {
        for &(piece, sq) in dirty.added() {
            self.add(piece, sq, net);
        }
        for &(piece, sq) in dirty.removed() {
            self.remove(piece, sq, net);
        }
    }

    /// Reverse a move's [`DirtyPiece`] deltas (the unmake direction).
    pub fn revert(&mut self, dirty: &DirtyPiece, net: &Network) {
        for &(piece, sq) in dirty.added() {
            self.remove(piece, sq, net);
        }
        for &(piece, sq) in dirty.removed() {
            self.add(piece, sq, net);
        }
    }
}

/// Run the output layer over an accumulator and return centipawns from the side-
/// to-move's perspective. `piece_count` selects the material bucket.
///
/// Arithmetic is ADR 0016 verbatim (the integer divisions truncate, so the step
/// order is load-bearing — do not algebraically combine them):
/// `Σ screlu(acc)·w` (in `QA²·QB`) `/QA` `+bias` `·SCALE` `/(QA·QB)`.
pub fn evaluate_accumulator(acc: &Accumulator, stm: Color, piece_count: u32) -> i32 {
    let net = network();
    let us = &acc.vals[stm.index()];
    let them = &acc.vals[stm.flip().index()];
    let bucket = output_bucket(piece_count);
    let weights = &net.output_weights[bucket];

    let mut out: i32 = 0;
    for (i, &a) in us.iter().enumerate() {
        out += screlu(a) * i32::from(weights[i]);
    }
    for (i, &a) in them.iter().enumerate() {
        out += screlu(a) * i32::from(weights[HL + i]);
    }

    out /= QA;
    out += i32::from(net.output_bias[bucket]);
    out *= SCALE;
    out /= QA * QB;
    out
}

/// The NNUE evaluator. Zero-sized: the weights live in the [`network`] singleton.
///
/// Implementing [`Evaluator`] rebuilds the accumulator from scratch on every call —
/// correct but not fast. Search wiring (#48) keeps an [`Accumulator`] updated
/// incrementally across make/unmake and calls [`evaluate_accumulator`] directly.
#[derive(Clone, Copy, Default)]
pub struct Nnue;

impl Evaluator for Nnue {
    fn evaluate(&self, board: &Board) -> i32 {
        let acc = Accumulator::from_board(board);
        evaluate_accumulator(&acc, board.side_to_move, board.occupied().count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::generate_legal;
    use crate::perft::{KIWIPETE, POS4, POS5, STARTPOS};
    use std::str::FromStr;

    fn eval_fen(fen: &str) -> i32 {
        Nnue.evaluate(&Board::from_str(fen).unwrap())
    }

    /// Anchor 1 — pin the bullet-validated `trainer/verify` outputs for the
    /// committed net. Reproducing these exact numbers confirms the *entire* forward
    /// pass at once: feature index (including the `sq ^ 56` flip direction), stm-
    /// first concat, SCReLU, dequant order, per-bucket weight/bias lookup, and net
    /// parsing. The positions deliberately span output buckets 0, 1, 2, 3 and 7 —
    /// startpos and the queen-odds positions all share bucket 7, so without the
    /// low-piece positions the bucket-selection path would be untested.
    ///
    /// These constants are tied to `nets/chess-engine-nnue-0001.bin`. If #48
    /// retrains, regenerate them: `cargo run -r -p verify -- nets/<net>.bin "<FEN>"…`.
    #[test]
    fn matches_verify_oracle() {
        let cases = [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", -318), // bucket 7
            ("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2267), // 7, white +Q
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1", -3002), // 7, white -Q
            ("8/2k5/8/8/3Q4/8/5K2/8 w - - 0 1", 2941),                          // bucket 0
            ("8/2k5/8/8/3Q4/8/5K2/8 b - - 0 1", -2889),                         // bucket 0
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", 49),                       // bucket 1
            ("4k3/pppppppp/8/8/8/8/8/4K3 w - - 0 1", -3462),                    // bucket 2
            ("r3k2r/pp4pp/8/8/8/8/PP4PP/R3K2R w KQkq - 0 1", 441),              // bucket 3
        ];
        for (fen, expected) in cases {
            assert_eq!(eval_fen(fen), expected, "eval mismatch for {fen}");
        }
    }

    /// Net-independent invariant: a colour-mirrored + vertically-flipped position,
    /// with the side to move flipped, must evaluate identically (the perspective
    /// network sees the same relative features). Survives retraining — unlike the
    /// pinned constants — so it guards the feature mapping for any future net.
    #[test]
    fn perspective_mirror_is_symmetric() {
        // An asymmetric position; mirror swaps colour, flips rank (sq ^ 56), and
        // flips the side to move.
        let fens = [
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2k5/8/8/3Q4/8/5K2/8 w - - 0 1",
            "4k3/pppppppp/8/8/8/8/8/4K3 w - - 0 1",
        ];
        for fen in fens {
            let board = Board::from_str(fen).unwrap();
            let mut mirror = Board::empty();
            let mut occ = board.occupied();
            while let Some(sq) = occ.pop_lsb() {
                let p = board.piece_on(sq).unwrap();
                let flipped = Piece { color: p.color.flip(), piece_type: p.piece_type };
                mirror.put_piece(Square(sq.0 ^ 56), flipped);
            }
            mirror.side_to_move = board.side_to_move.flip();

            assert_eq!(
                Nnue.evaluate(&board),
                Nnue.evaluate(&mirror),
                "mirror not symmetric for {fen}"
            );
        }
    }

    /// Net-independent invariant: more material for the side to move ⇒ higher eval.
    #[test]
    fn material_is_monotonic() {
        let start = eval_fen(STARTPOS);
        let up_q = eval_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let down_q = eval_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1");
        assert!(up_q > start + 200, "up a queen ({up_q}) not >> startpos ({start})");
        assert!(down_q < start - 200, "down a queen ({down_q}) not << startpos ({start})");
    }

    /// All evals are finite and of plausible magnitude on a varied FEN set.
    #[test]
    fn evals_are_finite_and_plausible() {
        for fen in [STARTPOS, KIWIPETE, POS4, POS5] {
            let e = eval_fen(fen);
            assert!(e.abs() < 30_000, "implausible eval {e} for {fen}");
        }
    }

    /// Anchor 2 — the canonical DirtyPiece guard. Walk the legal-move tree keeping
    /// an incremental accumulator (apply at make, revert at unmake) and assert it is
    /// **bit-exact** equal to the from-scratch accumulator at every node — comparing
    /// the full raw `2 × 256` `i16` state, not just the eval. A sign or index error
    /// in the incremental path (the classic silent Elo bug) breaks this immediately.
    fn walk(board: &mut Board, depth: u32, acc: &mut Accumulator, net: &Network, seen: &mut (bool, bool)) {
        assert_eq!(*acc, Accumulator::from_board(board), "incremental != from-scratch");
        if depth == 0 {
            return;
        }
        for mv in generate_legal(board) {
            seen.0 |= mv.is_en_passant();
            seen.1 |= mv.is_promotion();
            let undo = board.make_move(mv);
            acc.apply(&undo.dirty, net);
            walk(board, depth - 1, acc, net, seen);
            acc.revert(&undo.dirty, net);
            board.unmake_move(mv, undo);
        }
    }

    #[test]
    fn incremental_equals_from_scratch_over_perft_walk() {
        let net = network();
        let mut seen = (false, false); // (en-passant, promotion)
        // Position diversity beats depth: Kiwipete/POS4/POS5 exercise castling,
        // en-passant, and promotion — the DirtyPiece classes startpos never hits.
        for fen in [STARTPOS, KIWIPETE, POS4, POS5] {
            let mut board = Board::from_str(fen).unwrap();
            let mut acc = Accumulator::from_board(&board);
            walk(&mut board, 3, &mut acc, net, &mut seen);
        }
        assert!(seen.0, "walk never reached an en-passant capture — coverage gap");
        assert!(seen.1, "walk never reached a promotion — coverage gap");
    }
}
