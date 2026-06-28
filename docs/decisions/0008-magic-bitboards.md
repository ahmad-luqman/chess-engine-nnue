# ADR 0008 — Magic bitboards (found at runtime, not hard-coded)

- **Status**: accepted
- **Date**: 2026-06-28

## Context

Sliding-piece attacks were ray loops (correct, but O(squares-per-ray)). Issue #27
replaces them with magic bitboards — a single multiply-shift-index lookup — the
last Phase 2 movegen speedup, deferred from Phase 0 by design (issue #13). Two
sub-decisions matter: how to obtain the magic constants, and how to keep the swap
provably a no-op.

## Options considered

1. **Hard-code published magics** — copy a known-good set of 128 magic numbers
   (+ masks/shifts) as constants. Fast to write. Cost: a single mistyped digit
   corrupts attacks *silently* — movegen still returns *a* bitboard, just the
   wrong one, caught only by perft, and only on the affected square/occupancy.
2. **Find magics at startup with a verified search** — a fixed-seed PRNG proposes
   sparse candidates; each is accepted only after it maps every occupancy subset
   of the mask to the attack set computed by the ray-walk oracle. Deterministic
   (fixed seed), self-verifying, a few ms once. Cost: ~40 lines of finder, a
   `OnceLock`-gated build.
3. **`const fn` compile-time tables** — zero startup cost, but building 800 KB of
   tables (carry-rippler enumeration + ray geometry) in `const fn` is awkward and
   hard to read.

## Decision

**Option 2.** Find magics at first use with a fixed-seed xorshift PRNG, verifying
each against the ray-walk oracle (`movegen::ray_attacks`) before acceptance; cache
the tables behind a `OnceLock` (`src/magic.rs`). The public
`rook_attacks`/`bishop_attacks` signatures are unchanged, so every caller
(`is_square_attacked`, `generate_legal`) is an untouched drop-in.

## Consequences

- **Correct by construction**: a wrong magic can't be accepted — the finder
  rejects any candidate that disagrees with the oracle. The ray loop is kept
  precisely for this (and as the test oracle), not dead code.
- **Perft is the gate**: node counts for the standard positions are unchanged,
  confirming the swap is purely internal. A `magic == ray` equivalence test over
  random occupancies on every square backs it up directly.
- Startup pays a few ms once; the per-call `OnceLock` load is negligible against
  the lookup win. Perft throughput rises ~10% (the copy-make legality filter,
  not slider cost, dominates perft, so the search-side win is what the SPRT
  measures).
- If startup cost ever matters (it doesn't today), the found magics could be
  frozen into constants later — but then we'd keep the verifier as a test.
