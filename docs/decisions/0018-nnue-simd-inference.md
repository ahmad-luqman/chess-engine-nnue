# ADR 0018 — SIMD-accelerated NNUE inference (AVX2 / NEON)

- **Status**: accepted
- **Date**: 2026-06-30

## Context

The scalar NNUE forward pass (#46, ADR 0017) is correct but slow, and the eval
runs millions of times per search. The two hot loops in `src/nnue.rs` —
`Accumulator::add`/`remove` (a 256-lane `i16` column add/sub) and the
`Σ screlu(acc)·w` output reduction in `evaluate_accumulator` — are clean
fixed-width integer reductions, ideal SIMD targets. This is the engine's first use
of `core::arch`, so the change also has to set the target-feature / build story
without sacrificing a portable default build (issue #47).

Hard constraint: **speed only, no eval change.** The result must be bit-identical
to the scalar path, because the dequant in `evaluate_accumulator` uses
order-sensitive truncating integer divisions (ADR 0016) and the engine's strength
is measured against the prior version.

## Options considered

1. **Compile-time `target-feature` only** (e.g. `RUSTFLAGS=-C target-cpu=native`) —
   simplest code (let LLVM autovectorize), but the default build is then either
   non-portable or stuck on the SSE2 x86-64 baseline. Rejected as the *only*
   mechanism: breaks "default build is portable".
2. **`madd`-based SCReLU** (`_mm256_madd_epi16`) — fewer instructions, but the
   `clamp·weight` intermediate exceeds `i16`, so it cannot compute `clamp²·w`
   directly without extra juggling, and the reordering makes bit-identity harder to
   argue. Rejected for clarity/correctness.
3. **Widen-to-`i32` kernels behind runtime/cfg dispatch** (chosen) — load `i16`,
   widen to `i32`, clamp, square, multiply, accumulate in `i32` lanes, horizontal
   sum. Each SIMD op maps 1:1 onto a scalar op, so bit-identity is by construction.

## Decision

Widen-to-`i32` hand-written kernels (option 3), gated so the default build stays
portable:

- **AVX2** on x86-64, selected at runtime via `is_x86_feature_detected!("avx2")`,
  scalar fallback otherwise. **NEON** on aarch64, used unconditionally (NEON is the
  aarch64 baseline — no runtime check). A `*_scalar` reference for every kernel is
  **always compiled**: it is the portable fallback *and* the bit-identity oracle.
- A `simd` Cargo feature (default on); `--no-default-features` forces the scalar
  path, which CI exercises (`cargo test --no-default-features`) so it can't rot.
- **Unaligned loads** (`loadu` / `vld1q`); `HL = 256` and `2·HL = 512` tile cleanly
  by the lane counts, so there is no remainder and **no change to the `Network`
  layout or parsing**.
- `unsafe` is confined to the intrinsics, with a `# Safety` note per `unsafe fn`
  (the AVX2 fns require the runtime-checked feature; NEON is baseline).
- **AVX-512 deferred** — marginal on top of AVX2 for this width, narrow hardware
  base; revisit if profiling justifies it.
- `.cargo/config.toml` ships a **commented** `target-cpu=native` example only; a
  committed active default would apply to CI/releases and break portability.

Why bit-identity holds: the accumulator ops are plain `i16` lane add/sub
(identical to scalar). The reduction differs only in the *order* it sums the 512
`screlu·w` terms, which is associative in `i32`; no lane overflows because the
whole sum already fits `i32` (worst case ≈ 2.1e9 < `i32::MAX`), so any grouping
gives the same total.

## Consequences

- **Verification:** a new test `simd_matches_scalar_over_perft_walk` asserts the
  dispatched SIMD path equals the always-compiled scalar reference — bit-exact for
  both the raw `2 × 256` `i16` accumulator and the reduction — over the perft-walk
  tree. This is the real guard: the existing `incremental_equals_from_scratch`
  test compares two paths that both go through the dispatched `add`, so it cannot
  catch a systematic SIMD bug; `matches_verify_oracle` pins only 8 positions.
- **AVX2 is bit-verified only in CI.** The dev host is aarch64, so the AVX2 module
  does not compile locally and runs nowhere locally (Rosetta reports no AVX2). It
  is compile-checked with `cargo check --target x86_64-apple-darwin` and
  bit-verified by the GitHub x86 runner.
- A new `nnue` criterion group (`forward/*`, `accumulator/*`) measures the win;
  compare default vs `--no-default-features`. The `i16` accumulator add may already
  be autovectorized under fat-LTO, so its hand-SIMD gain can be small — the forward
  reduction is the higher-value target.
- If the reduction ever *panics* in a debug build (rather than asserting unequal),
  that signals an `i32` lane overflow → widen the lanes to `i64`. Not expected for
  any valid quantised net.
- Establishes the `core::arch` + `target_feature` + per-arch-dispatch pattern for
  any future SIMD in the engine.
