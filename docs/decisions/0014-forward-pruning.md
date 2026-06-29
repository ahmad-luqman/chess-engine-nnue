# ADR 0014 — Near-leaf forward pruning: reverse futility, futility (razoring rejected)

- **Status**: accepted
- **Date**: 2026-06-29

## Context

NMP (#36) prunes whole nodes by *passing*; LMR (#37) reduces late quiet moves.
Issue #38 adds the **static, near-leaf forward prunes** that need no search at all
— they read the static eval and a depth-scaled margin and decide a node (or a
move) is hopeless. Three were on the table:

- **Reverse futility / static null move** — if `eval` clears `beta` by a margin,
  assume fail-high and return.
- **Futility** — if a quiet move's `eval + margin` can't reach `alpha`, skip it.
- **Razoring** — if `eval` is a margin *below* `alpha` at low depth, verify with a
  qsearch and drop to it if the node really is fail-low.

The shared hazard is **silent tactical blindness**: a margin that's too aggressive
prunes a real resource, and nothing flags it except a lost game (or a tactical-test
regression). So each is gated to non-PV, non-check nodes, kept away from mate-bound
windows, and — following the #39 discipline — **SPRT'd independently** so a flat or
broken one can't hide behind a strong one.

## Options considered

1. **Bundle all three, one SPRT.** Rejected — a combined result can't attribute the
   gain, and (as it turned out) can't isolate a *regression*.
2. **RFP, then futility, then razoring — separate SPRTs, each kept only if it earns
   its place.** Chosen.
3. **Skip forward pruning, lean on LMR/NMP.** Leaves a large, well-understood gain
   on the table at this strength level.

## Decision

**Ship RFP and futility; reject razoring.** All in `negamax` (`src/search.rs`),
guarded by a shared per-node `static_eval` (computed once when not in check) and
`is_pv = beta - alpha > 1`.

- **Reverse futility** — before NMP: `!is_pv && !in_check && depth <= RFP_MAX_DEPTH
  (6) && beta < MATE_BOUND && static_eval - RFP_MARGIN(85)*depth >= beta` returns
  `static_eval`. It passes no move and runs no search, so it stays shallow with a
  generous margin. **SPRT vs the post-#39 release: +76.92 ± 20.49 Elo**, LLR 2.95,
  886 games.
- **Futility** — inside the move loop, after `make_move`: a move with `i >= 1`
  (never the first, so at least one move is always searched), non-PV, not in check,
  `alpha < MATE_BOUND`, not tactical, not giving check, whose `static_eval +
  FUT_MARGIN(150)*depth <= alpha`, is skipped (`continue`). The give-check probe is
  last in the chain (it scans the post-make board). **SPRT vs the RFP build:
  +110.53 ± 23.50 Elo**, LLR 2.95, 614 games.

**Razoring was implemented and rejected.** Even with the qsearch *verification* the
issue calls for, a depth ≤ 3 razoring prune **dropped a forced mate-in-2** in the
test suite: the mating line is a queen sacrifice, so after the sac the static eval
reads "lost", razoring drops the node to a qsearch that only resolves captures —
never the quiet mate — and the node is pruned as fail-low. The regression was a
hard test failure, not a soft SPRT question, so razoring does not ship. RFP and
futility already capture the bulk of #38's value, so chasing a safe razoring variant
(e.g. depth 1 only) wasn't worth the tactical exposure.

## Consequences

- **Two SPRT-verified gains stack** to roughly +187 Elo over the post-#39 release —
  the largest jump of the selective-search phase. Each is the acceptance gate for
  its own change (iron rule #3); razoring's gate was the no-regression mate test,
  which it failed.
- **Correctness is behavioural.** `rfp_cutoffs` / `fut_prunes` active-rate tests
  assert the prunes fire in bulk; the existing forced-mate and decisively-winning
  tactical tests assert no dropped win. Futility's `i >= 1` gate is what makes
  "never the only move" a structural guarantee rather than a margin gamble.
- **The margins are the obvious tuning targets.** `RFP_MARGIN`, `RFP_MAX_DEPTH`,
  `FUT_MARGIN`, `FUT_MAX_DEPTH` rode in on their first SPRT'd values; they're prime
  candidates for the next Texel/SPSA pass. Over-pruning would surface as tactical
  misses, so they were set conservatively.
- **Razoring stays a lesson, not a feature.** The verified-qsearch form is still
  blind to sacrificial mates; if revisited, it needs a much lower depth cap and its
  own mate-suite gate before any SPRT is trusted.
- **#38 closes; #35 (aspiration windows) is the last open Phase 3 search item.**
