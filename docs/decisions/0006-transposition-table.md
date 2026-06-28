# ADR 0006 — Transposition table

- **Status**: accepted
- **Date**: 2026-06-28

## Context

The same position recurs constantly during search via different move orders
(*transpositions*) and across iterative-deepening iterations. Re-searching each
occurrence is wasted work. A transposition table (TT) caches each searched
position's result, keyed by its Zobrist hash (ADR 0005), so a later arrival can
reuse it.

Several design points are non-obvious and worth recording.

## Options considered

- **Replacement scheme**: always-replace vs. depth-preferred vs. bucketed
  (N-way). Depth-preferred (keep the deeper search, always evict across
  generations) is the simple, strong default; buckets are a later refinement.
- **Entry size / key storage**: store the full 64-bit key (simple, robust
  collision check) vs. a 16-bit key fragment (denser, needs care). We store the
  full key — clarity first; density is a later optimization.
- **Fail-hard vs. fail-soft** when adding TT bounds: the Phase 1 search is
  fail-hard. We keep fail-hard so adding the TT does not, by itself, change any
  search value — the TT-disabled path stays byte-identical to v0.1.0.
- **Persisting the table** across iterations/moves: a fresh table per `go` throws
  away exactly the cross-move information that makes the TT valuable; we keep one
  table, aged per search, cleared on `ucinewgame`.

## Decision

A flat, power-of-two-sized `Vec<Entry>` indexed by `hash & (len-1)`, with
**depth-preferred replacement** that always evicts entries from older searches
(an `age`/generation byte). Each `Entry` stores the **full key**, best move,
score, depth, and a bound (`Exact`/`Lower`/`Upper`). The search keeps the search
**fail-hard**: a TT probe cut returns the same `alpha`/`beta` the node would have.
Mate scores are stored **relative to the node** (adjusted by `ply` on store and
probe) so a cached mate keeps the right distance when probed elsewhere. The table
is owned by the UCI layer, borrowed by the search, **cleared on `ucinewgame`**
and resized by **`setoption name Hash value <MB>`** (default 16 MB).

## Consequences

- Correctness is anchored on the existing exact-score search tests (mate scores,
  hanging-piece tactics, stalemate) — they must pass unchanged with the TT on —
  plus best-move invariance (TT vs. no-TT) and a warm-reuse node-count drop.
- **Fixed-depth scores are no longer perfectly invariant.** With depth-preferred
  probing, a position stored at depth *d* and re-probed at a shallower node
  returns the deeper score (a "depth-leak"). This is standard and benign — it
  only makes a leaf more accurate — but it means we test *best-move* invariance,
  not bit-identical scores. (The issue's "same score with/without TT" criterion
  is therefore met in spirit, not literally; documented here deliberately.)
- The dominant strength gain comes from searching the **TT/PV move first**, which
  is move ordering (#25). The TT alone (cutoffs + ID reuse) is a smaller, and at
  fast time controls noisier, gain — especially while the engine still lacks
  quiescence (#26) and draw detection (#28) and so converts won positions poorly.
  This is expected from the #24/#25 split, not a defect in the TT.
- `best_move` is stored now and consumed by move ordering in #25.
