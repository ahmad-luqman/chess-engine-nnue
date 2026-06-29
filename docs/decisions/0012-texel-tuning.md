# ADR 0012 — Texel tuning of the eval weights

- **Status**: accepted
- **Date**: 2026-06-29

## Context

The `(mg, eg)` weights from tapered eval ([ADR 0011 era], #40) and the hand-crafted
terms (#41) were hand-guesses. **Texel tuning** fits them by minimising the logistic
error of the static eval against game-outcome-labelled quiet positions — the
standard last squeeze from a hand-crafted eval before NNUE (Phase 4).

The eval is (after one reparam) **linear in its weights**, which opens up the fast,
robust feature-extraction + gradient approach: reduce each position once to the
white-relative `(mg, eg)` coefficient of every weight, then gradient-descend the
weight vector. The engine keeps baked `const` weights; an offline tuner
(`examples/texel.rs`) fits new values and we paste them back.

## Options considered

1. **Black-box local search over a runtime `Params` eval** — no feature math, but
   slow for ~500 params and needs the engine eval refactored to read params.
2. **Feature extraction + gradient descent, consistency-checked** — a `trace`
   mirrors the eval and emits coefficients; an exact integer check asserts
   `trace · default == Material::evaluate` on every position, so a mis-mirror fails
   loudly. Fast, and the engine keeps fast baked consts.
3. **Skip tuning** — ship #41's hand weights. Leaves measurable Elo unclaimed.

## Decision

**Option 2.** Key choices, several learned the hard way:

- **King safety is *not* tuned to convergence on quiet data, and the king PST is
  frozen.** Texel filters to *quiet* positions, but king danger only manifests in
  *sharp* ones, so those features have little signal; and only ~2 kings/position
  means king-square params are data-starved (Adam drove them to ±1000). The king
  PST stays at its sound #40 values; the (linear) king-attack weights are tuned but
  watched for sanity.
- **`KING_DANGER_PER_UNIT` was folded into per-piece centipawn weights** so king
  safety is linear in its weights (score-preserving reparam, verified neutral).
- **`PIECE_VALUE` is held fixed** (pawn = 100 anchors `K`, and it's shared with move
  ordering); the PST absorbs material drift — the tuner found effective
  values ≈ P 100 / N 440 / B 385 / R 620 / Q 1210, anchored off the pawn.
- **Dataset quality is the whole game.** A first attempt on a small self-play set
  (depth-6, ~4k games) *regressed −93 Elo* despite a lower dataset MSE — noisy,
  correlated labels. Switching to the public **zurichess `quiet-labeled.epd`**
  (~725k real-game quiet positions) gave a clean fit (train/val MSE within 0.0002)
  and the shipped gain. The self-play generator (`examples/gen_data.rs`) is kept
  but the public set is the recommended source.

## Consequences

- **An SPRT gain over #41** (the only gate that counts; the dataset-MSE drop alone
  is not trustworthy, as the −93 regression proved).
- **Eval weights are now data-derived**, and re-tuning is one command against a
  dataset file. Reproducible: same dataset → same emitted constants.
- **Some tests became eval-coupled** and were updated; the PVS-invariance test now
  asserts the chosen *move* only, and a qsearch test was reframed to an
  eval-independent `is_quiet` check, so future retunes don't churn them.
- **Eval internals (`Score`, weight consts, masks) are now `pub`** so the offline
  tuner reads the exact defaults — a wider API surface, justified by the tuner.
- **King safety and the king PST remain hand-set**, flagged for a future pass with
  sharper-position data or a dedicated king-safety tuning scheme.
