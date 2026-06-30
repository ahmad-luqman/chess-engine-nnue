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
            acc_add(&mut self.vals[perspective], col);
        }
    }

    /// Remove a piece's feature column from both perspectives.
    #[inline]
    pub fn remove(&mut self, piece: Piece, sq: Square, net: &Network) {
        for perspective in 0..2 {
            let col = &net.feature_weights[feature_index(perspective, piece, sq)];
            acc_sub(&mut self.vals[perspective], col);
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

    // The hot reduction `Σ screlu(acc)·w` (SIMD-accelerated when available); the
    // truncating dequant below stays scalar because its step order is load-bearing.
    let mut out = forward(us, them, weights);

    out /= QA;
    out += i32::from(net.output_bias[bucket]);
    out *= SCALE;
    out /= QA * QB;
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD hot paths (#47)
//
// Three implementations of each of the two NNUE hot loops — accumulator
// column add/sub and the `Σ screlu·w` output reduction:
//
//   * `*_scalar` — the always-compiled reference. Bit-exact, portable, and the
//     oracle the `simd_matches_scalar_over_perft_walk` test compares against.
//   * `simd::*_avx2` — x86-64, selected at runtime via `is_x86_feature_detected!`.
//   * `simd::*_neon` — aarch64, where NEON is baseline (no runtime check needed).
//
// The `acc_add`/`acc_sub`/`forward` dispatchers pick a kernel at the cfg/runtime
// level so the rest of the module is SIMD-agnostic. All three kernels are
// bit-identical: the accumulator ops are plain `i16` lane add/sub, and the
// forward reduction differs only in the *order* it sums the 512 `screlu·w` terms,
// which is associative in `i32` (no lane overflows — the whole sum already fits).
// ─────────────────────────────────────────────────────────────────────────────

/// Scalar reference: add a feature column into one perspective's accumulator.
///
/// Never feature-gated out — it is the portable fallback *and* the bit-identity
/// oracle for the SIMD kernels, so it can be unused in a SIMD-only library build.
#[inline]
#[allow(dead_code)]
fn acc_add_scalar(acc: &mut [i16; HL], col: &[i16; HL]) {
    for (a, &w) in acc.iter_mut().zip(col) {
        *a += w;
    }
}

/// Scalar reference: subtract a feature column from one perspective's accumulator.
#[inline]
#[allow(dead_code)]
fn acc_sub_scalar(acc: &mut [i16; HL], col: &[i16; HL]) {
    for (a, &w) in acc.iter_mut().zip(col) {
        *a -= w;
    }
}

/// Scalar reference for the output reduction `Σ screlu(acc)·w` (in `QA²·QB`).
///
/// The `us` half multiplies `weights[0..HL]`, the `them` half `weights[HL..2HL]`;
/// the running sum stays `i32` (a single `screlu·w` term can approach `i32::MAX`).
#[inline]
#[allow(dead_code)]
fn forward_scalar(us: &[i16; HL], them: &[i16; HL], weights: &[i16; 2 * HL]) -> i32 {
    let mut out: i32 = 0;
    for (i, &a) in us.iter().enumerate() {
        out += screlu(a) * i32::from(weights[i]);
    }
    for (i, &a) in them.iter().enumerate() {
        out += screlu(a) * i32::from(weights[HL + i]);
    }
    out
}

/// Add a feature column into one perspective's accumulator (SIMD when available).
#[inline]
fn acc_add(acc: &mut [i16; HL], col: &[i16; HL]) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: the runtime check above confirms AVX2 is present.
            return unsafe { simd::acc_add_avx2(acc, col) };
        }
        return acc_add_scalar(acc, col);
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    // SAFETY: NEON is part of the aarch64 baseline, so it is always available.
    return unsafe { simd::acc_add_neon(acc, col) };
    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    acc_add_scalar(acc, col)
}

/// Subtract a feature column from one perspective's accumulator (SIMD when available).
#[inline]
fn acc_sub(acc: &mut [i16; HL], col: &[i16; HL]) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: the runtime check above confirms AVX2 is present.
            return unsafe { simd::acc_sub_avx2(acc, col) };
        }
        return acc_sub_scalar(acc, col);
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    // SAFETY: NEON is part of the aarch64 baseline, so it is always available.
    return unsafe { simd::acc_sub_neon(acc, col) };
    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    acc_sub_scalar(acc, col)
}

/// The output reduction `Σ screlu(acc)·w` (SIMD when available).
#[inline]
fn forward(us: &[i16; HL], them: &[i16; HL], weights: &[i16; 2 * HL]) -> i32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: the runtime check above confirms AVX2 is present.
            return unsafe { simd::forward_avx2(us, them, weights) };
        }
        return forward_scalar(us, them, weights);
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    // SAFETY: NEON is part of the aarch64 baseline, so it is always available.
    return unsafe { simd::forward_neon(us, them, weights) };
    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    forward_scalar(us, them, weights)
}

/// Hand-written SIMD kernels. Every `unsafe fn` here is bit-identical to its
/// `*_scalar` counterpart; `unsafe` is confined to the intrinsics, and the safety
/// contract is the same for all of them (documented per fn).
#[cfg(feature = "simd")]
mod simd {
    use super::{HL, QA};

    /// AVX2 accumulator add: 16 `i16` lanes per 256-bit vector, `HL` divisible by 16.
    ///
    /// # Safety
    /// Caller must ensure the AVX2 target feature is available (checked at runtime
    /// in [`super::acc_add`]). The fixed-size array args make the unaligned loads
    /// and `HL`-strided indexing in-bounds.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn acc_add_avx2(acc: &mut [i16; HL], col: &[i16; HL]) {
        use core::arch::x86_64::*;
        let mut i = 0;
        while i < HL {
            let a = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
            let w = _mm256_loadu_si256(col.as_ptr().add(i) as *const __m256i);
            _mm256_storeu_si256(acc.as_mut_ptr().add(i) as *mut __m256i, _mm256_add_epi16(a, w));
            i += 16;
        }
    }

    /// AVX2 accumulator subtract. See [`acc_add_avx2`] for the safety contract.
    ///
    /// # Safety
    /// Same as [`acc_add_avx2`]: AVX2 must be available.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn acc_sub_avx2(acc: &mut [i16; HL], col: &[i16; HL]) {
        use core::arch::x86_64::*;
        let mut i = 0;
        while i < HL {
            let a = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
            let w = _mm256_loadu_si256(col.as_ptr().add(i) as *const __m256i);
            _mm256_storeu_si256(acc.as_mut_ptr().add(i) as *mut __m256i, _mm256_sub_epi16(a, w));
            i += 16;
        }
    }

    /// AVX2 output reduction. Each lane group widens 8 `i16` → `i32` (`cvtepi16_epi32`),
    /// clamps to `[0, QA]`, squares, multiplies by the weight, and accumulates. Two
    /// independent accumulators (16 `i16` per iteration) hide the `mullo` latency; both
    /// are summed, then the eight `i32` lanes are horizontally reduced. Bit-identical to
    /// the scalar reduction (associative `i32`, no lane overflow).
    ///
    /// # Safety
    /// AVX2 must be available (checked in [`super::forward`]).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn forward_avx2(us: &[i16; HL], them: &[i16; HL], weights: &[i16; 2 * HL]) -> i32 {
        use core::arch::x86_64::*;

        // One lane group: widen 8 i16 → i32, clamp [0, QA], square, ×weight, accumulate.
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn lane(acc: __m256i, vals: *const i16, wts: *const i16, qa: __m256i) -> __m256i {
            let zero = _mm256_setzero_si256();
            let a = _mm256_cvtepi16_epi32(_mm_loadu_si128(vals as *const __m128i));
            let c = _mm256_min_epi32(_mm256_max_epi32(a, zero), qa); // clamp [0, QA]
            let sq = _mm256_mullo_epi32(c, c); // screlu
            let w = _mm256_cvtepi16_epi32(_mm_loadu_si128(wts as *const __m128i));
            _mm256_add_epi32(acc, _mm256_mullo_epi32(sq, w))
        }

        // Two independent accumulators (16 i16 per iteration) to hide mul latency.
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn dot(
            a0: __m256i,
            a1: __m256i,
            vals: *const i16,
            wts: *const i16,
        ) -> (__m256i, __m256i) {
            let qa = _mm256_set1_epi32(QA);
            let (mut a0, mut a1) = (a0, a1);
            let mut i = 0;
            while i < HL {
                a0 = lane(a0, vals.add(i), wts.add(i), qa);
                a1 = lane(a1, vals.add(i + 8), wts.add(i + 8), qa);
                i += 16;
            }
            (a0, a1)
        }

        let z = _mm256_setzero_si256();
        let (a0, a1) = dot(z, z, us.as_ptr(), weights.as_ptr());
        let (a0, a1) = dot(a0, a1, them.as_ptr(), weights.as_ptr().add(HL));
        let acc = _mm256_add_epi32(a0, a1);

        // Horizontal sum of the eight i32 lanes.
        let lo = _mm256_castsi256_si128(acc);
        let hi = _mm256_extracti128_si256(acc, 1);
        let s = _mm_add_epi32(lo, hi);
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
        _mm_cvtsi128_si32(s)
    }

    /// NEON accumulator add: 8 `i16` lanes per 128-bit vector, `HL` divisible by 8.
    ///
    /// # Safety
    /// NEON is part of the aarch64 baseline, so this is always callable on aarch64.
    /// The fixed-size array args keep the loads/stores in-bounds.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn acc_add_neon(acc: &mut [i16; HL], col: &[i16; HL]) {
        use core::arch::aarch64::*;
        let mut i = 0;
        while i < HL {
            let a = vld1q_s16(acc.as_ptr().add(i));
            let w = vld1q_s16(col.as_ptr().add(i));
            vst1q_s16(acc.as_mut_ptr().add(i), vaddq_s16(a, w));
            i += 8;
        }
    }

    /// NEON accumulator subtract. See [`acc_add_neon`] for the safety contract.
    ///
    /// # Safety
    /// Same as [`acc_add_neon`].
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn acc_sub_neon(acc: &mut [i16; HL], col: &[i16; HL]) {
        use core::arch::aarch64::*;
        let mut i = 0;
        while i < HL {
            let a = vld1q_s16(acc.as_ptr().add(i));
            let w = vld1q_s16(col.as_ptr().add(i));
            vst1q_s16(acc.as_mut_ptr().add(i), vsubq_s16(a, w));
            i += 8;
        }
    }

    /// NEON output reduction. Clamps to `[0, QA]` in `i16`, then widening-square
    /// (`vmull_s16`, exact since `c² ≤ QA² = 65025` fits `i32`) and fused multiply-
    /// accumulate (`vmlaq_s32`) by the widened weight. Four independent `i32`
    /// accumulators (16 `i16` per iteration) hide the MLA latency — without this ILP
    /// the kernel loses to LLVM's autovectorised scalar on Apple Silicon; with it the
    /// forward pass is a few percent faster. Bit-identical (associative `i32`).
    ///
    /// # Safety
    /// NEON is part of the aarch64 baseline, so this is always callable on aarch64.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn forward_neon(us: &[i16; HL], them: &[i16; HL], weights: &[i16; 2 * HL]) -> i32 {
        use core::arch::aarch64::*;

        // One 8-wide block: clamp in i16, widening-square (`vmull_s16`, exact since
        // `c² ≤ 65025`), then `+= c²·w` via fused MLA on the widened weight.
        #[inline]
        #[target_feature(enable = "neon")]
        unsafe fn block(
            lo: int32x4_t,
            hi: int32x4_t,
            vals: *const i16,
            wts: *const i16,
        ) -> (int32x4_t, int32x4_t) {
            let zero = vdupq_n_s16(0);
            let qa = vdupq_n_s16(QA as i16);
            let c = vminq_s16(vmaxq_s16(vld1q_s16(vals), zero), qa);
            let w = vld1q_s16(wts);
            let lo = vmlaq_s32(
                lo,
                vmull_s16(vget_low_s16(c), vget_low_s16(c)),
                vmovl_s16(vget_low_s16(w)),
            );
            let hi = vmlaq_s32(hi, vmull_high_s16(c, c), vmovl_high_s16(w));
            (lo, hi)
        }

        // Four independent i32 accumulators (16 i16 per iteration) to hide MLA latency.
        let z = vdupq_n_s32(0);
        let (mut a0, mut a1, mut b0, mut b1) = (z, z, z, z);
        for (vals, wts) in
            [(us.as_ptr(), weights.as_ptr()), (them.as_ptr(), weights.as_ptr().add(HL))]
        {
            let mut i = 0;
            while i < HL {
                (a0, a1) = block(a0, a1, vals.add(i), wts.add(i));
                (b0, b1) = block(b0, b1, vals.add(i + 8), wts.add(i + 8));
                i += 16;
            }
        }
        vaddvq_s32(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(b0, b1)))
    }
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
    fn walk(
        board: &mut Board,
        depth: u32,
        acc: &mut Accumulator,
        net: &Network,
        seen: &mut (bool, bool),
    ) {
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

    /// The SIMD verification (#47). The other tests cannot catch a systematic SIMD
    /// bug: `incremental_equals_from_scratch_over_perft_walk` compares two paths
    /// that *both* go through the dispatched (SIMD) `add`, and `matches_verify_oracle`
    /// pins only eight positions. This walks the full perft-walk tree and at every
    /// node asserts the dispatched SIMD path is **bit-exact** equal to the always-
    /// compiled `*_scalar` reference, for BOTH hot loops:
    ///   * the raw `2 × 256` i16 accumulator (`Accumulator::add` vs `acc_add_scalar`), and
    ///   * the `Σ screlu·w` reduction (`forward` vs `forward_scalar`).
    ///
    /// Only meaningful when the `simd` feature is on (otherwise both sides are the
    /// scalar path). On real x86 this is where the AVX2 kernel is bit-verified —
    /// it cannot be *run* on the aarch64 dev host, only in CI. NB: a *panic* here in
    /// a debug build (rather than an assert failure) would mean an i32 lane overflow
    /// in the reduction → widen the lanes to i64, not a logic bug.
    #[cfg(feature = "simd")]
    fn walk_simd_vs_scalar(board: &mut Board, depth: u32, net: &Network) {
        // Accumulator: SIMD-built (from_board) vs scalar-built, full raw i16 state.
        let simd_acc = Accumulator::from_board(board);
        let mut scalar_vals = [net.feature_bias; 2];
        let mut occ = board.occupied();
        while let Some(sq) = occ.pop_lsb() {
            let piece = board.piece_on(sq).expect("occupied square has a piece");
            for (p, vals) in scalar_vals.iter_mut().enumerate() {
                acc_add_scalar(vals, &net.feature_weights[feature_index(p, piece, sq)]);
            }
        }
        assert_eq!(simd_acc.vals, scalar_vals, "SIMD accumulator != scalar reference");

        // Reduction: SIMD `forward` vs `forward_scalar` for this position's bucket.
        let bucket = output_bucket(board.occupied().count());
        let weights = &net.output_weights[bucket];
        let us = &simd_acc.vals[board.side_to_move.index()];
        let them = &simd_acc.vals[board.side_to_move.flip().index()];
        assert_eq!(
            forward(us, them, weights),
            forward_scalar(us, them, weights),
            "SIMD forward != scalar"
        );

        if depth == 0 {
            return;
        }
        for mv in generate_legal(board) {
            let undo = board.make_move(mv);
            walk_simd_vs_scalar(board, depth - 1, net);
            board.unmake_move(mv, undo);
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_matches_scalar_over_perft_walk() {
        let net = network();
        for fen in [STARTPOS, KIWIPETE, POS4, POS5] {
            let mut board = Board::from_str(fen).unwrap();
            walk_simd_vs_scalar(&mut board, 3, net);
        }
    }
}
