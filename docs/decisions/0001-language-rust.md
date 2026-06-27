# ADR 0001 — Language: Rust

- **Status**: accepted
- **Date**: 2026-06-27

## Context

The engine must be both a learning vehicle *now* and a competitive engine
*later*, without a forced rewrite between those stages. Engine development is
performance engineering: search runs millions of nodes per second, so the
language must be in the top performance tier. It also benefits from strong
correctness tooling because move-generation and search bugs are subtle.

## Options considered

1. **Rust** — C++-class performance; memory safety (no segfaults/UB chasing while
   learning); excellent built-in tooling (`cargo`, `clippy`, `criterion`, tests).
   Modern engine peers to learn from (Viridithas, Carp). The leading NNUE trainer
   (`bullet`) is Rust. Cost: borrow-checker learning curve.
2. **C++** — lingua franca (Stockfish, Dragon); largest reference corpus. Cost:
   manual memory management, more footguns, heavier build setup.
3. **C** — minimal and fast, good tutorials (VICE). Cost: most manual; DIY
   everything.
4. **Python first** — fastest path to *understanding*, but ~50–100× too slow to be
   competitive; would require a rewrite.

## Decision

**Rust.** It uniquely satisfies "learn safely now" and "compete seriously later"
in one codebase, with tooling that reduces the class of bugs hardest to debug
while learning.

## Consequences

- Reference material is often C++ (Stockfish) — we translate idioms to Rust.
- Borrow checker shapes some data structures (e.g. the undo stack, TT access).
- NNUE path aligns with `bullet` (Rust). See ADR 0003.
- Performance ceiling is not a concern; top-tier strength is achievable.
