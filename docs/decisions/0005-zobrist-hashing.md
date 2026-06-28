# ADR 0005 — Zobrist hashing

- **Status**: accepted
- **Date**: 2026-06-28

## Context

Phase 2 needs a cheap, collision-resistant 64-bit identity for a position: the
transposition table (#24) keys on it, and repetition detection (#28) compares
it across the game history. Recomputing such a key by scanning all 64 squares at
every node would dominate search cost, so the key must be maintainable
*incrementally* as moves are made and unmade.

## Options considered

1. **Zobrist hashing** — assign a random 64-bit constant to each board feature
   ((color, piece, square), side-to-move, castling rights, ep file); the key is
   their XOR. A move toggles a few features, so the update is O(features changed).
   The universal choice in alpha-beta engines. Cost: a table of random constants;
   care needed to toggle *exactly* the features that change.
2. **Hash the FEN / full board** with a general hash (e.g. FxHash) — trivial to
   write, no incremental story. Cost: O(64) per node, far too slow for search.

Sub-decisions within option 1:
- **Constant generation**: a `const fn` splitmix64 stream from a fixed seed
  (compile-time table, reproducible hashes) vs. runtime RNG init.
- **Unmake**: replay the move's XOR deltas in reverse vs. snapshot the pre-move
  key in `Undo` and restore it.
- **En-passant**: hash the ep file whenever an ep square exists vs. only when an
  enemy pawn can *actually* capture en passant.

## Decision

**Zobrist hashing**, with: compile-time `const` constants from a fixed-seed
splitmix64 (`src/zobrist.rs`); **incremental update in `make_move`**; and
**O(1) unmake by restoring a pre-move snapshot stored in `Undo`** (XOR is
invertible, but restoring a snapshot is simpler and equally cheap, and `Undo`
already carries exactly this kind of un-recomputable state).

The ep file is hashed **only when the capture is genuinely available** (a
side-to-move pawn attacks the ep square). This is required for correctness, not
an optimization: `1.e4 e5 2.Nf3` and `1.Nf3 e5 2.e4` are the same position, but
the second leaves a non-capturable ep square on e3 — they share a key only under
the capturable-only rule.

## Consequences

- TT (#24) and repetition/fifty-move detection (#28) can build directly on
  `Board::hash`.
- A `debug_assert` in `make_move` checks the incremental key against a
  from-scratch `compute()` at every node, so perft (run under `cargo test`)
  doubles as exhaustive hash verification; the assert vanishes in `--release`.
- Reproducible hashes (fixed seed) keep any future on-disk structure (opening
  books, tuning caches) stable across runs.
- A subtle trap is documented in code and tests: the round-trip and
  incremental≡from-scratch checks both pass even with the naive ep rule, because
  both code paths agree with each other — only the transposition test discriminates.
