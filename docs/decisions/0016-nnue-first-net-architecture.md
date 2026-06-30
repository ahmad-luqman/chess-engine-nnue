# ADR 0016 — First NNUE: architecture, quantisation & net file layout

- **Status**: accepted
- **Date**: 2026-06-30

## Context

Phase 4 replaces the hand-crafted eval with an NNUE. Issue #45 trains and exports
the **first** net; issue #46 implements in-engine inference; #48 wires it in. The
trainer is `bullet` (ADR 0003). Before training we must fix the net's
architecture, its quantisation constants, and — most importantly — the exact byte
layout and inference arithmetic of the exported net, because that layout is a
**contract** the inference code (#46) must reproduce bit-for-bit. A mismatch does
not crash; it silently evaluates garbage.

`docs/04-nnue.md` originally described a HalfKP/HalfKA, king-relative architecture.
The 2026 consensus for a *first* net (bullet docs, Viridithas/cosmo) is the
opposite: start with the simplest thing that captures most of the gain.

## Options considered

1. **HalfKP / HalfKAv2 (king-relative, ~40k inputs)** — strongest ceiling, but
   needs king-bucket refreshes + finny tables, far more data to exploit, and far
   more code. Wrong first step; easy to *lose* Elo vs a simpler net at low data.
2. **Plain `Chess768` perspective net `(768 → 256)×2 → 1×8`, SCReLU** — king
   agnostic, no accumulator refresh on king moves, trivial to make incremental,
   trains well on modest data. The recommended starting point.

## Decision

Train a **plain 768-input perspective net, `(768 → 256)×2 → 1×8`, SCReLU**, in
bullet. Single most important reason: it is the simplest architecture that beats a
tuned HCE, and it keeps the incremental-update and inference code small enough to
get correct on the first try. Strength refinements (bigger HL, buckets, HalfKA)
are later, SPRT-gated iterations.

### Architecture & training constants

- **Inputs**: `Chess768`, 768 = `2 colours × 6 pieces × 64 squares`.
- **Hidden**: 256 per perspective; two accumulators (stm + opponent) concatenated
  → 512. Activation **SCReLU**: `f(x) = clamp(x, 0, 1)²` (quantised: `clamp(x,0,QA)²`).
- **Output buckets**: `MaterialCount<8>` → `bucket = (piece_count − 2) / 4`
  (integer division; `piece_count` = number of pieces on the board, clamp to 0..7).
- **Quantisation**: `QA = 255` (accumulator weights), `QB = 64` (output weights),
  `SCALE = 400` (eval scale, centipawns).
- **Loss / labels**: `output.sigmoid()` vs `target`, squared error;
  `target = wdl·game_result + (1−wdl)·sigmoid(cp_score / SCALE)`, `ConstantWDL = 0.75`.
  Both `cp_score` and `game_result` are **white-relative** in the data (bullet
  re-orients to stm internally).
- **Optimiser/schedule**: AdamW, batch 16384, StepLR `{start 0.001, γ 0.1, step 18}`.

### Net file layout (`quantised.bin`) — the #46 contract

Little-endian `i16` throughout, written in bullet `save_format` order; matrices are
**column-major** with shape `output × input`. The layout below is what `verify`
loads and what #46 must replicate.

| Block | Count (i16) | Shape / meaning | Quant factor |
|-------|-------------|-----------------|--------------|
| `feature_weights` | `768 × 256` | `[feature][hidden]` (column-major of `256×768`) | `QA` |
| `feature_bias`    | `256`       | shared by both accumulators | `QA` |
| `output_weights`  | `8 × 512`   | `[bucket][512]`; `0..256` = **stm**, `256..512` = ntm. `l1w` is `.transpose()`d at save so each bucket's 512 weights are contiguous | `QB` |
| `output_bias`     | `8`         | `[bucket]` | `QA·QB` |

Total `200968` i16 = `401936` bytes, zero-padded to a multiple of 64 → `401984` B.

### Inference arithmetic (must match exactly)

1. **Feature index** (`Chess768`): for a piece of `colour`, type `pt∈0..5`
   (P,N,B,R,Q,K), square `sq` (a1=0..h8=63), evaluated for `perspective`:
   `rel_colour = (colour != perspective)` (0=friendly,1=enemy);
   `rel_sq = perspective==white ? sq : sq^56`;
   `index = [0,384][rel_colour] + 64·pt + rel_sq`.
   The stm accumulator uses `perspective = stm`; the opponent uses `perspective = !stm`.
2. **Concatenate stm accumulator FIRST**, opponent second.
3. **SCReLU in i32**: `clamp(x,0,QA)²` reaches `QA² = 65025` — accumulate in `i32`,
   never `i16`.
4. **Dequantise (this order)**: `out = Σ screlu(acc)·w` (in `QA²·QB`); `out /= QA`;
   `out += output_bias[bucket]`; `out *= SCALE`; `out /= QA·QB`. Result is
   centipawns, **side-to-move relative** (matches `Evaluator::evaluate`).

These three conventions (stm-first concat, i32 SCReLU, exact dequant order) are the
classic silently-Elo-losing bugs; `trainer/verify/src/main.rs` is the reference
implementation and asserts material monotonicity + perspective-mirror equality.

## Consequences

- Makes #46 a near-mechanical port of `trainer/verify` into `src/` behind the
  existing `Evaluator` trait; incremental updates hook the #43 `DirtyPiece` deltas.
- The DirtyPiece NPS cost (~4%, already paid, see memory) is consumed here, not added.
- Forecloses nothing: bigger HL, more buckets, or HalfKA are future SPRT-gated steps;
  each would supersede the constants above, not the pipeline.
- `docs/04-nnue.md` is updated to match (it previously described HalfKP).
