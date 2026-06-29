# ADR 0011 — Null-move pruning (NMP)

- **Status**: accepted
- **Date**: 2026-06-29

## Context

PVS ([ADR 0009](0009-principal-variation-search.md), #34) and LMR
([ADR 0010](0010-late-move-reductions.md), #37) made the search *deeper*. **Null-move
pruning (#36) makes it *narrower*** — it cuts whole nodes. The bet: if we let the
side to move *pass* (a null move) and a shallower search of the resulting position
still fails high (`>= beta`), then the real moves — which can only be better than
passing — would fail high too, so we can return the cutoff without searching any of
them. NMP is one of the largest single pruning gains in a classical engine.

Its one real failure mode is **zugzwang**: positions (almost always king-and-pawn
endings) where every legal move worsens your position, so passing would be *better*
than the real moves — exactly the assumption NMP inverts. The open questions were
how to guard zugzwang, how much to reduce, and how to keep the cutoff from
fabricating mate scores.

## Options considered

1. **Unconditional NMP with a fixed `R`** — simplest, but prunes in pawn endings
   (zugzwang blunders) and wastes null searches in positions we're already losing.
2. **Guarded NMP: zugzwang guard + adaptive `R` + `eval >= beta` gate** — only
   attempt a null move when the side to move has non-pawn material (the zugzwang
   guard), `beta` is a real (non-mate) bound, the node isn't in check, and the
   static eval already clears `beta` (so pruning is plausible). Reduce by an
   adaptive `R` (2, rising to 3 at depth ≥ 6). Forbid two null moves in a row.
3. **Option 2 + a high-depth verification search** — re-search without NMP before
   trusting a high-depth cutoff, catching the rare zugzwang the material guard
   misses. Strictly safer, but more code and a second search; deferred until the
   material guard is shown insufficient in play.

## Decision

**Option 2.** NMP lives in `negamax` (`src/search.rs`), after the
terminal/fifty-move/`depth == 0` checks (so the node is known non-terminal) and
before move ordering (so a cutoff skips the ordering work). It is gated on: a
`null_ok` flag, `depth >= 3`, not in check, `beta < MATE_BOUND`,
`has_non_pawn_material(side)`, and `eval(board) >= beta`. On a fail-high it returns
`beta` (fail-hard, consistent with the rest of the search).

Two correctness choices matter:

- **`null_ok` is a `negamax` parameter, not a `SearchContext` flag.** It is `false`
  only for the immediate null child and `true` for every real-move recursion, which
  expresses "never two null moves in a row" precisely — with no restore-on-every-path
  hazard a shared flag would carry.
- **Returning `beta` under `beta < MATE_BOUND` cannot fabricate a mate.** A null
  search can report an inflated mate score (the side that passed gets mated faster);
  returning the *bound* rather than the null score, and only when `beta` isn't itself
  a mate value, keeps mate distances honest.

The null move itself is a new `Board::make_null_move` / `unmake_null_move` pair
(`src/board.rs`): flip the side to move and clear the en-passant square, mirroring
both into the incremental Zobrist key exactly as `make_move` does (a `debug_assert`
against a from-scratch recompute guards the key math).

## Consequences

- **Large node reduction at equal depth** — startpos depth 10 drops 723k → 510k
  nodes (−30%), depth 9 415k → 185k (−55%) vs the pre-NMP build. The **SPRT vs the
  current release is the acceptance gate** for #36 (iron rule #3).
- **No fixed-depth invariance test** (as with LMR): NMP deliberately changes the
  tree. Correctness is behavioural — a tactical suite asserts no dropped win, an
  active-rate test asserts cutoffs fire, and a **pawn-only endgame asserts zero null
  attempts**, a direct check of the zugzwang guard. The forward Zobrist key is
  validated by comparing a made null move against a freshly-parsed post-null
  position (a round-trip can't — `unmake` restores a snapshot).
- **One feature bundles three knobs** — the `eval >= beta` gate and the adaptive
  `R` schedule ride in the first SPRT'd version. If the signal is flat, those are
  the first two to toggle.
- **Refinements deferred:** storing the null cutoff in the TT, the high-depth
  verification search (Option 3), and skipping NMP at PV nodes; `R` and the
  eval-margin are candidates for the Texel-tuning pass (#42).
