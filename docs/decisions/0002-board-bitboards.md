# ADR 0002 — Board representation: bitboards

- **Status**: accepted
- **Date**: 2026-06-27

## Context

The board representation underpins move generation, make/unmake, and ultimately
NNUE feature extraction. It is the hardest thing to change later, so it's chosen
deliberately up front.

## Options considered

1. **Bitboards** (`u64` per piece type/color) — move gen via bit ops; `popcount`
   and `trailing_zeros` are single instructions; the standard in all strong
   engines. Cost: less intuitive at first; sliding pieces need magic bitboards
   for full speed.
2. **Mailbox / 0x88** (array of squares) — intuitive, simpler. Cost: materially
   slower; not used by competitive engines.

## Decision

**Bitboards**, with a redundant **mailbox `piece_on[64]` array** alongside for
convenient lookups. Sliding-piece attacks start with simple ray loops (correct,
Phase 0) and move to **magic bitboards** later (fast, Phase 2–3).

## Consequences

- Move generation is bit manipulation; perft is the correctness gate.
- NNUE feature updates map naturally onto per-piece bitboard changes.
- A clear later optimization milestone exists: ray loops → magic bitboards.
- Zobrist hashing (for TT/repetition) layers on top cleanly.
