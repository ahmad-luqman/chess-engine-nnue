# ADR 0007 — Repetition history on the search context, not the board

- **Status**: accepted
- **Date**: 2026-06-28

## Context

Draw detection (issue #28) needs the sequence of Zobrist keys the game has passed
through, so repetitions can be recognized. The question is *where that key stack
lives*. The issue sketched maintaining it on `Board` — push in `make_move`, pop
in `unmake_move`.

`Board`, however, is a deliberately *pure value*: it derives `Clone`/`Eq`, carries
no hidden history, and is `clone`d at the start of every search. `make_move`
returns an `Undo` rather than mutating any internal stack, precisely so the board
stays a plain snapshot.

## Options considered

1. **Key stack on `Board`** — matches the issue sketch; `make`/`unmake` keep it
   current automatically. Cost: every `Board::clone` (once per search, and the
   board is cloned freely elsewhere) copies a growing `Vec`; `Eq` between two
   equal positions reached differently would now differ on history; perft and
   other `Board` users pay for a field they don't use.
2. **Key stack on the search context** — the search pushes the current key before
   descending and pops on return, seeded from the game history passed in by the
   UCI layer. Cost: the push/pop is in the search loop rather than in
   `make`/`unmake`, so it must be kept in lockstep with them by hand.

## Decision

**Option 2.** The repetition stack is a `Vec<u64>` on the search context, seeded
from the game-history keys the UCI layer records as it applies `position … moves`,
and maintained by push/pop around `make_move`/`unmake_move` in `run_root` and
`negamax`. `Board` is untouched and stays a pure value.

## Consequences

- `Board` clones stay cheap and its `Eq` keeps meaning "same position", which
  several tests rely on (e.g. make/unmake round-trips).
- The push/pop must bracket every `make`/`unmake` in the search; this is local to
  two functions and covered by the repetition test.
- Quiescence needs no repetition tracking: it plays only captures/promotions,
  which are irreversible, so no repetition can arise within it.
- When NNUE arrives (Phase 4) its accumulator stack will live on the evaluator /
  context for the same reason — keeping per-search, path-dependent state off the
  pure `Board` is the consistent pattern.
