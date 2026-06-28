# ADR 0009 — Principal Variation Search (null-window scouts)

- **Status**: accepted
- **Date**: 2026-06-28

## Context

Phase 3 opens the *selective-search* layer (issue #34). The first technique,
Principal Variation Search (PVS), is the scaffolding the rest build on: LMR (#37)
and null-move pruning (#36) both reduce a search and then *re-search on
fail-high*, which is exactly the null-window-scout-plus-re-search shape PVS
introduces.

Plain alpha-beta searches every move with the full `(alpha, beta)` window. Once
move ordering (#25) plus the TT move make the first move almost always best, the
remaining moves only need to be *proven worse* than the current PV. A null window
(`beta = alpha + 1`) does that far more cheaply — its tighter bound prunes more —
and we pay for a full-width re-search only when a scout surprisingly fails high.

Two things made this non-trivial here:

- The existing `negamax` is **fail-hard** (it returns clamped `alpha`/`beta`, not
  the true score). The re-search condition has to be correct under fail-hard.
- Fixed-depth scores are **not** TT-invariant. The TT probe already documents this
  ("a deeper entry probed at a shallower node returns the deeper score — a known,
  accepted fixed-depth instability"), and PVS makes it sharper: scouts store
  bounds (from null windows) a full-window search would never produce.

## Options considered

1. **Convert to fail-soft, then add PVS** — fail-soft returns sharper bounds that
   make re-searches slightly cheaper and the TT a touch more informative. Cons: a
   larger, riskier change touching every return in `negamax`/`qsearch`, harder to
   prove result-invariant against the current engine, and not required for PVS to
   work.
2. **Keep fail-hard, add PVS minimally** — search move 0 full-window; scout moves
   1..n with `(alpha, alpha+1)`; re-search at full width only when the scout
   returns `s` with `alpha < s < beta`. The `s < beta` guard matters: when our own
   window is already null (a scout one level up), the scout *is* the full window,
   so no re-search is owed. Cons: leaves the fail-hard bound coarseness in place.

## Decision

**Option 2.** PVS sits inside the existing fail-hard loops in `run_root` and
`negamax`; move ordering and TT store/probe are untouched. The re-search fires on
`s > alpha && s < beta` (at the root, which takes no beta cutoff, simply
`s > alpha`). Fail-soft is deferred — it can come later as its own SPRT-gated
change if the sharper bounds prove worth it.

**Verification is TT-disabled on purpose.** Result-invariance is asserted against
golden scores captured from the pre-PVS engine *with the TT disabled*
(`TranspositionTable::disabled()`), where PVS-vs-plain-alpha-beta is genuinely
byte-identical — same alpha progression, same tie-breaks. Asserting against
TT-enabled numbers would be flaky and could push a "fix" that introduces a real
bug. Best-move equality is asserted only on positions with a unique answer; quiet,
many-ties positions (e.g. the start position) assert score only.

## Consequences

- **Foundation in place** for LMR/NMP: both can now reduce-then-re-search using the
  same scout structure.
- **Re-search rate is the health signal.** With TT + ordering the scout proves
  moves worse almost without exception — measured ~0.00% (tens of re-searches per
  ~10⁶ scouts) on a normal middlegame run. A high rate would mean ordering is
  broken or PVS is miswired; a unit test guards it stays well under 20%.
- **Node counts at shallow fixed depth are mixed**, especially with the TT on
  (scouts store null-window bounds that interact with later probes) and TT-off on
  quiet, weakly-ordered positions (more moves nudge alpha, each costing a
  re-search). This is expected PVS behaviour; the acceptance gate is the SPRT vs
  v0.5.0 (non-regression-or-gain), not the node count.
- **Fail-hard remains.** If a later technique wants sharper TT bounds, fail-soft
  becomes a clean follow-up ADR rather than being entangled with PVS here.
