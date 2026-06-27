# src/ — engine crate

Rust source for the engine. See root `CLAUDE.md` for build/test commands and the
iron rules; see `docs/01-research-landscape.md` for the techniques behind each
module.

## Module build order

Build and test each before starting the next. Earlier modules are the foundation
the later ones assume correct.

```
types      core domain types: Color, PieceType, Square        [exists]
bitboard   Bitboard(u64) set + bit primitives                 [in progress: issue #8]
board      position state: piece bitboards, occupancy, mailbox, stm/castle/ep/clock
movegen    attack tables, sliding attacks, legal move gen, make/unmake
perft      node-count correctness oracle — THE Phase 0 gate
search     negamax + alpha-beta, iterative deepening, TT, pruning
eval       hand-crafted (material+PST) → swappable → NNUE
uci        UCI protocol loop + time management
```

## Conventions specific to this crate

- **Square encoding: a1 = 0 … h8 = 63.** Bitboard bit `i` is `Square(i)`. Every
  module depends on this; do not introduce a second convention.
- Prefer **immutable bitboard ops** (`with`/`without` returning a copy) for
  composability; use `&mut` only on genuine hot paths where it's measured to help.
- Hot paths: keep them allocation-free. Profile (`cargo flamegraph`/`samply`)
  before optimizing; the optimizer + LTO already do a lot.
- Every module carries `#[cfg(test)]` unit tests. `movegen` correctness is
  asserted through `perft`, not just unit tests.

## NNUE-readiness (design now, implement in Phase 4)

- `make`/`unmake` should make it cheap to know exactly which `(piece, square)`
  features changed — the NNUE accumulator update hooks there.
- Track king squares cleanly (king-relative features need them).
