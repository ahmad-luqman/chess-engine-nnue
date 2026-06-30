# ADR 0017 — NNUE inference: incremental perspective accumulator (scalar)

- **Status**: accepted
- **Date**: 2026-06-30

## Context

[ADR 0016](0016-nnue-first-net-architecture.md) fixed the *what* of the first net:
the `(768 → 256)×2 → 1×8` SCReLU architecture, the `QA/QB/SCALE` quantisation, the
`quantised.bin` byte layout, and the exact dequantisation arithmetic. This ADR
records the *how* of evaluating it inside the engine (issue #46): the data
structures and update strategy in `src/nnue.rs`. It does **not** restate 0016 —
the feature index, concat order, SCReLU, and dequant live there and `src/nnue.rs`
matches them bit-for-bit (cross-validated against `trainer/verify`, itself
validated against bullet's own `Chess768`). Search wiring and SPRT are #48/#49.

## Options considered

**Accumulator keying — by perspective colour vs by side-to-move**
1. **Two accumulators keyed by colour** (White-view, Black-view); at eval, read the
   side-to-move's as "us" and the other as "them". A piece toggle updates both
   views with the same column regardless of whose turn it is, so make/unmake never
   has to swap or rebuild. Costs one extra `i16[256]` of state.
2. **Two accumulators keyed by stm/opponent.** Saves nothing meaningful and forces
   a swap (or a refresh) every ply, complicating the incremental path.

**Update strategy — incremental vs recompute**
1. **Incremental from `DirtyPiece`** (#43): add the few changed columns at make,
   subtract them at unmake. The defining NNUE optimisation.
2. **Recompute from scratch each node.** Simple, but throws away the whole point.

**Trait seam**
- The minimal [`Evaluator`](../../src/eval.rs) trait takes `&Board` and returns a
  score, with no per-move state. NNUE's fast path is inherently stateful (it owns
  an accumulator updated across the search stack), so it cannot live *entirely*
  behind that call.

## Decision

- **Accumulators keyed by perspective colour** (`Accumulator { vals: [[i16; 256]; 2] }`,
  index 0 = White, 1 = Black). Side-agnostic updates are worth one extra vector.
- **Incremental update driven by `DirtyPiece`**: `Accumulator::apply` adds the
  added features and removes the removed ones at make; `revert` does the inverse at
  unmake — the Stockfish `StateInfo` model, riding the existing `Undo.dirty`.
- **Two entry points**: `evaluate_accumulator(acc, stm, piece_count)` is the fast,
  state-carrying path #48 will call from search; the `Nnue: Evaluator` impl rebuilds
  the accumulator from scratch per call — correct, used by tests and any
  non-search caller, and explicitly *not* the hot path.
- **Scalar only.** SIMD is #47; this lands a correct reference first.

Single most important reason: colour-keyed accumulators + `DirtyPiece` make the
incremental update a few-column add/subtract with no per-ply bookkeeping, and keep
the from-scratch builder trivially equal to it — which is exactly what the
correctness guard below checks.

## Consequences

- **Correctness is pinned two independent ways.** (1) The forward pass is anchored
  to `trainer/verify`'s bullet-validated golden evals for the committed net, on
  positions spanning output buckets 0–3 and 7 (startpos and the queen-odds
  positions all share bucket 7, so the low-piece positions are what actually
  exercise bucket selection). (2) A perft walk over startpos/Kiwipete/POS4/POS5
  asserts the incremental accumulator is **bit-exact** equal to from-scratch at
  every node — comparing the raw `2×256` `i16` state, not just the eval — and
  asserts the walk reaches en-passant and promotion nodes. This is the canonical
  guard against a `DirtyPiece` sign/index bug that silently costs hundreds of Elo.
- The golden-eval constants are tied to `nets/chess-engine-nnue-0001.bin`; if #48
  retrains they must be regenerated from `verify`. The mirror-symmetry and
  material-monotonicity invariants are net-independent and survive retraining.
- #48 builds its make/unmake accumulator stack on `apply`/`revert`/`from_board`,
  then flips search from `Material` to `Nnue` under SPRT. The ~4% NPS that
  `DirtyPiece` already costs (see project memory) is consumed here, not added anew.
- Forecloses nothing: #47 swaps the scalar loops for SIMD behind the same API.
