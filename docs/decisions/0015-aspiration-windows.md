# ADR 0015 — Aspiration windows

- **Status**: accepted
- **Date**: 2026-06-30

## Context

Iterative deepening re-searches the root at each depth, and until now every
iteration opened the full `(-INF, INF)` window. But the previous iteration's score
is a strong prior: the score at depth `d+1` is usually within a pawn or two of the
score at depth `d`. An **aspiration window** (#35) exploits that — open a narrow
window around the prior and only pay for a wider search when the true score
actually falls outside it.

The narrow window isn't just a smaller root span; it tightens `beta` (and `alpha`)
all the way down the tree, so the forward prunes added in #38 (reverse futility,
NMP, futility) fire far more often. The cost is the occasional **re-search**: when
the score lands outside the window the root search "fails" low or high and must be
repeated wider. The design question is purely how to widen, and how to stay correct
around mate scores and the shallow depths that have no usable prior.

## Options considered

1. **No aspiration — full window every iteration.** Simple and exact, but leaves
   the prior unused and the tree wider than it needs to be.
2. **Fixed narrow window, re-search at full width on any fail.** Captures most of
   the benefit but over-pays on a fail: a score just past the window jumps straight
   to `(-INF, INF)` instead of a slightly wider bracket.
3. **Narrow window that widens incrementally on the failing side (chosen).** Seed
   `prev ± δ`; on a fail-low drop `alpha`, on a fail-high raise `beta`, doubling `δ`
   each time, until the score is bracketed (eventually reaching `±INF`). Re-searches
   only the side that failed, and only as far as needed.

## Decision

**Option 3**, in `src/search.rs`. The root search was refactored so aspiration is a
thin loop on top of a windowed core:

- **`search_root(board, ctx, depth, alpha, beta)`** is the fixed-depth root search,
  now **fail-soft**: it returns whether the result fell at/below `alpha` (fail-low)
  or at/above `beta` (fail-high), takes a real root beta cutoff, and stores the TT
  bound (`Exact` / `Upper` / `Lower`) its window-relative score implies.
- **`run_root`** is now just `search_root(.., -INF, INF)` — a full-window, always-
  exact search. The fixed-depth [`search`] entry and every test keep using it
  unchanged, so their results stay byte-identical.
- **`aspiration_search`** wraps `search_root` for the timed iterative-deepening
  loop: seed `alpha = prev - δ`, `beta = prev + δ` with `δ = ASPIRATION_DELTA` (25
  cp); on `score <= alpha` lower `alpha` by `δ`, on `score >= beta` raise `beta` by
  `δ`, doubling `δ` each time; return as soon as the score is inside the window.

Two correctness gates:

- **Depths 1–2 search the full window** — there's no reliable prior yet (depth 1 is
  seeded from the initial score 0).
- **A mate-bound prior searches the full window** — "mate in N" can't be bracketed
  by a centipawn margin, so a `|prev| >= MATE_BOUND` prior skips aspiration entirely
  rather than fail repeatedly trying to widen onto a mate.

Termination is guaranteed: each fail widens the failing side toward `±INF`, and once
a side reaches infinity it can no longer fail that way, so the loop converges to a
full window in the worst case.

## Consequences

- **SPRT vs the post-#38 release is the acceptance gate** (iron rule #3):
  **+53.45 ± 16.34 Elo**, LLR 2.95 (pass), 1140 games.
- **No fixed-depth invariance regression.** `run_root` is full-window and exact, so
  the fixed-depth `search` and the PVS move-invariance golden are untouched.
  Aspiration's own correctness is tested directly: a window seeded with the true
  score returns the *same* score and move as a full-window search, and a window
  seeded with a deliberately wrong prior widens and recovers the true score (with
  the `asp_researches` counter firing).
- **Aspiration compounds with #38.** The narrow `beta` is exactly what makes RFP and
  futility prune harder, so this gain rides partly on those already being in.
- **`ASPIRATION_DELTA` and the doubling schedule are tuning targets.** Too tight and
  re-searches dominate; too wide and the window buys nothing. The first SPRT'd
  value is conservative; a Texel/SPSA pass can revisit it.
- **Phase 3's selective-search track is complete.** With PVS, LMR, NMP, SEE +
  check extensions, forward pruning, and now aspiration all in, epic #4 closes and
  Phase 4 (NNUE) is next — the `Evaluator` trait and frozen king tables leave the
  seam ready.
