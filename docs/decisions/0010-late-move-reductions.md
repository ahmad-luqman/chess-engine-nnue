# ADR 0010 — Late move reductions (LMR)

- **Status**: accepted
- **Date**: 2026-06-29

## Context

PVS ([ADR 0009](0009-principal-variation-search.md), #34) added the
reduce-then-re-search scaffolding but measured ~Elo-neutral on its own — it only
narrowed windows. **Late move reductions (#37) are what turn that scaffolding into
strength.** With good move ordering, moves late in the list are unlikely to be
best, so we search them at *reduced* depth; a reduced search that beats alpha is
re-searched at full depth. LMR is the single biggest depth multiplier in a modern
alpha-beta searcher.

The open questions were which moves to reduce, by how much, and how to keep
reductions from dropping tactics.

## Options considered

1. **Reduce everything late, fixed amount** — simple, but over-reduces forcing
   moves (checks, captures) and tactical lines, costing accuracy that the
   re-search can't always recover within the iteration.
2. **Depth/move-count log table + exemptions** — reduce only *quiet, late,
   non-checking* moves, by a `floor(0.75 + ln(d)·ln(i)/2)` table that grows with
   depth and lateness; exempt the PV/TT move (it's ordered first), killers,
   captures/promotions, moves giving check, and any move while in check. Verify
   every fail-high at full depth before trusting it.
3. **SEE-gated reductions** — also reduce/extend based on static exchange value.
   Better, but SEE is #39 (not yet built), so this is a later refinement.

## Decision

**Option 2.** LMR lives inside the existing PVS scout arm of `negamax`
(`src/search.rs`) as a three-tier search: reduced null-window scout → full-depth
null-window re-search (if the reduced scout beat alpha) → full-window PVS
re-search (if it's a real new best). When the reduction `r == 0` — any
non-eligible move — the path is byte-identical to plain PVS, so only eligible late
quiets change.

Eligibility: `depth >= 3`, move index `>= 3`, `!is_tactical`, node not in check,
not a killer, and the move does not give check. The `gives_check` test is a
non-incremental `in_check` scan, so it is placed last in the `&&` chain (only run
once the cheap tests pass); the node-in-check scan is itself gated on `depth >= 3`,
sparing shallow nodes. The reduction table is clamped so the reduced depth stays
`>= 1` (never straight into qsearch, never a `u32` underflow), and is built once at
startup via `search::init()` — never on a search's clock (the [magic-init lesson](
0009-principal-variation-search.md) made concrete: a lazily-built table billed to
the first move truncates it at a short time control).

## Consequences

- **~2 plies deeper in fixed time** vs v0.5.0 (warm depth-in-time: 5.5→7.1 @100 ms,
  6.3→8.2 @200 ms, 7.1→9.3 @500 ms across a 10-position set). This is the payoff
  that justifies PVS retroactively; the **combined PVS+LMR SPRT vs v0.5.0 is the
  acceptance gate for both #34 and #37**.
- **No fixed-depth invariance test** (unlike PVS): LMR deliberately changes the
  tree, so correctness is guarded behaviourally — a tactical suite (mates + a
  decisively winning position, captured from the pre-LMR engine at the depth it
  solved them) asserts reductions don't drop a win, plus an active-rate test that
  reductions fire with a bounded re-search rate.
- **One tunable knob.** The reduction lives in a single table/function, so the
  formula and exemptions are SPRT-tunable without touching the search structure.
- **Refinements deferred to their issues:** SEE-gated reductions (#39), and
  history-magnitude-scaled reductions (#25 data) — both reduce *less* for moves
  that look good and *more* for moves that look bad. Left out here to keep the
  first version's SPRT signal clean.
