# ADR 0013 — Static Exchange Evaluation (SEE) + check extensions

- **Status**: accepted
- **Date**: 2026-06-29

## Context

Move ordering (#25) ranks captures by **MVV-LVA** — most valuable victim, least
valuable attacker. That heuristic can't see past the *first* capture: it rates
`RxP` highly even when the pawn is defended and the rook is lost on the recapture.
**Static Exchange Evaluation (#39)** fixes this by resolving the *whole* capture
sequence on the target square statically — both sides keep recapturing with their
least valuable attacker — and returning the net material the initiating side ends
up with. Knowing a capture loses material lets us order it last and skip it in
quiescence.

Issue #39 bundles a second, unrelated selective-search idea: **check extensions**.
The fixed depth limit can cut a forcing line off mid-check, handing a position
where the king is under attack to the quiescence search — which stands pat on the
static eval and so badly misjudges it. Spending one extra ply whenever the side to
move is in check keeps forcing lines intact to their resolution.

The two are independent strength changes, so each is gated by its **own** SPRT
(iron rule #3) — a combined run couldn't tell which half worked.

## Options considered

1. **SEE everywhere vs. one site.** Use SEE for move ordering, for qsearch
   pruning, or both. Ordering pays per capture at every node (cost); qsearch
   pruning removes provably-losing captures from the hottest part of the tree.
   Chose **both**, with the ordering cost as the first thing to revert if the
   SPRT came back flat.
2. **SEE attacker bookkeeping: incremental vs. recompute.** The Chess Programming
   Wiki's iterative SEE maintains an attacker set and XORs in X-ray pieces as the
   swap proceeds. Simpler and just as correct here: **recompute `attackers_to`
   against the shrinking occupancy each round** — sliders recomputed against the
   new mask reveal X-ray batteries for free. Clarity over a micro-optimisation in
   non-perft code.
3. **Check-extension gating: reuse the `depth >= 3` check vs. compute
   unconditionally.** The existing `in_check_node` is only computed at `depth >= 3`
   (where NMP/LMR fire). Reusing it would never extend at the frontier (depths
   1–2) — exactly where a cut into qsearch hurts most. Chose to compute the check
   status **unconditionally** and extend at every depth.
4. **Extension cap: explicit counter vs. ply ceiling.** An extension makes
   `depth - 1 + 1 = depth`, so a perpetual cross-check never lets depth reach 0.
   A per-path extension counter is one option; a single **selective-depth ceiling**
   (`ply >= MAX_PLY` returns the static eval) is simpler and bounds *all* runaway
   recursion, not just check chains. Chose the ceiling.

## Decision

All in `negamax` / `move_score` / `qsearch` (`src/search.rs`).

**SEE** — `see(board, mv) -> i32` is the classic swap-off: `gain[0]` is the victim
value; each round the side to recapture picks its least valuable attacker (ordered
by `PieceType` *index*, since the king's `PIECE_VALUE` is 0 and must rank as
recapturer of last resort), and the gains are folded back with the stand-pat rule
(a side recaptures only when it improves its result). `attackers_to` is recomputed
against the shrinking `occupied` mask each round (X-ray reveal). En-passant removes
the victim from its own square (not the landing square); a king may only recapture
into an undefended square. Two use sites:

- **Ordering**: a non-promotion capture with `SEE < 0` returns its (negative) SEE
  score, sorting it below every quiet move (history is `>= 0`) and ordering losing
  captures least-loss first. `SEE >= 0` keeps MVV-LVA; the TT move is never demoted.
- **Quiescence**: captures with `SEE < 0` are skipped outright (counted in
  `see_prunes`).

**v1 scope:** promotions *inside* the swap sequence are ignored (a promoting pawn
counts as a pawn). Both sites use only the **sign** of SEE, so this is safe; the
magnitude matters only for the future futility margins (#38).

**Check extensions** — the check status is computed unconditionally; `ext` is 1 in
check, 0 otherwise; `child_depth = depth - 1 + ext` feeds every recursion site. In
check, LMR is suppressed (its `!in_check_node` gate), so `ext` and any reduction
`r` are never both set and `child_depth - r` can't underflow. The
**selective-depth ceiling** `if ply >= MAX_PLY { return eval }` at the top of
`negamax` is the backstop that bounds recursion and *is* the extension cap.

## Consequences

- **Two separate SPRTs are the acceptance gate.** SEE (ordering + qsearch) vs. the
  pre-SEE build: **+47.95 ± 16.05 Elo**, LLR 2.97 (pass), 1400 games. Check
  extensions vs. the post-SEE build: **+26.73 ± 11.41 Elo**, LLR 2.95 (pass), 2370
  games.
- **SEE correctness is behavioural and fixture-driven.** A SEE bug doesn't crash —
  it mis-orders/mis-prunes and surfaces only as a flat SPRT. Hand-verified
  exchange sequences (hanging queen, rook-for-defended-pawn, equal trade, X-ray
  battery, king-can't-recapture-into-check, en passant) are the real gate, plus a
  `see_prunes` active-rate test.
- **The check extension only ever searches *deeper*,** so it can't drop a tactic;
  a `check_extensions` active-rate test and a no-regression forced-mate test cover
  it. The one fixed-depth golden that moved is a **PVS move-invariance** case where
  SEE now surfaces a winning capture-with-check ahead of the old quiet tie-break —
  sound, since qsearch SEE pruning only drops losing captures and so can't depress
  the side-to-move's score.
- **SEE is a force-multiplier for the rest of Phase 3.** Futility/razoring (#38)
  will gate its pruning on capture safety, and LMR (#37) can later avoid reducing
  SEE-winning captures.
- **Deferred:** promotions inside the SEE sequence; an incremental attacker set if
  ordering cost ever shows in a profile; gating LMR on SEE; and richer extensions
  (singular, recapture) beyond the single check ply.
