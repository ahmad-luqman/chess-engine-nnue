# ADR 0004 — Match running & SPRT: cutechess-cli

- **Status**: accepted
- **Date**: 2026-06-28

## Context

Iron rule #3 says *every strength change is SPRT-tested against the prior
version*. "Looks like an improvement" is usually noise: a change can win a
handful of games by luck and still be neutral or negative. We need a tool that
plays many engine-vs-engine games at a fast time control, ideally from a varied
set of opening positions, and tells us — with statistical rigour — whether the
new build is genuinely stronger.

The engine is **headless** and speaks UCI (issue #18), so any standard UCI match
runner can drive it. We just need to pick one and write down how we use it.

## Options considered

1. **cutechess-cli** — the de-facto standard match runner in the
   Stockfish/Dragon lineage this engine follows. Built-in SPRT, PGN output,
   opening books (PGN/EPD), concurrency, tournament formats. Battle-tested and
   widely documented. <https://github.com/cutechess/cutechess>
2. **fastchess** — a newer, faster, drop-in-ish alternative with the same SPRT
   workflow; what several top engines now use for very high game throughput.
   <https://github.com/Disservin/fastchess>
3. **Custom harness** — write our own. Maximal control, but re-implementing
   SPRT, time control, and PGN handling is exactly the undifferentiated work the
   above tools already do correctly.

## Decision

Use **cutechess-cli** as the match runner and SPRT tool. It is the lineage
standard, pairs cleanly with a UCI engine, and its documentation is everywhere.
Keep **fastchess** bookmarked as a faster drop-in for when game throughput
becomes the bottleneck (long SPRT runs in later phases) — the invocations are
close enough that switching later is cheap.

## Consequences

- A local install of a match runner (and a small opening book) is a developer
  prerequisite for measuring strength — see [07-testing.md](../07-testing.md)
  for install + invocations. In practice **fastchess** is installed on the dev
  machine (cutechess needs Qt and isn't packaged for macOS); the flags are
  near-identical, so this is the sanctioned drop-in, not a reversal of the
  decision above.
- The working method becomes concrete: change → `cargo test`/perft (correctness)
  → SPRT vs the previous tagged build (strength) → keep only if it passes.
- We must keep a **tagged baseline binary** around as the opponent ("tag releases
  so we always have opponents," per the roadmap). Each accepted gainer becomes
  the next baseline.
- Strength numbers are release-only (iron rule #2); never SPRT a debug build.
