# CLAUDE.md — chess-engine-nnue

Guidance for Claude Code (and humans) working in this repo.

## What this is

A chess engine written in **Rust**, built from bitboards up toward an **NNUE**
(neural) evaluation, in the Stockfish/Dragon lineage (alpha-beta + NNUE, not
Leela-style MCTS). It begins as a learning project and grows into a competitive
engine. Full context lives in [`docs/`](docs/README.md) — read it before large
changes.

## CRITICAL: No AI Attribution in Commits

**NEVER add AI attribution to commit messages** - no Co-Authored-By, no Claude mentions, no Generated-by tags.

```bash
# WRONG
git commit -m "feat: Add auth\n\nCo-Authored-By: Claude <noreply@anthropic.com>"

# CORRECT
git commit -m "feat: Add auth\n\nImplements JWT tokens with refresh."
```

## The one mental model

A strong engine is **headless** — it speaks **UCI** over stdin/stdout; a separate
GUI/tournament manager drives it. We never build a GUI. Engine-vs-engine matches
(+ SPRT) are how we measure progress.

## Build, run, test

```bash
cargo build --release     # the binary IS the product; always bench in release
cargo run                 # debug stub (10-50x slower; correctness only)
cargo test                # unit + perft tests
cargo test bitboard       # a single module's tests
cargo clippy --all-targets
```

## Iron rules

1. **No search before perft passes.** Move generation must match published perft
   node counts (start, Kiwipete, positions 3/4/5) to depth 5-6 before any search
   work begins.
2. **Measure in `--release`.** Debug numbers are meaningless for an engine.
3. **Every strength change is SPRT-tested** (from Phase 1 on) against the prior
   version. Looks-like-an-improvement is usually noise.
4. **Decisions get an ADR.** Non-obvious choices go in `docs/decisions/` (copy
   `0000-template.md`).
5. **Keep eval swappable.** Evaluation lives behind an interface so HCE → NNUE is
   a clean substitution (see `docs/04-nnue.md`).

## Layout

| Path | What |
|------|------|
| `src/` | engine crate (see `src/CLAUDE.md` for module order) |
| `docs/` | research, roadmap, NNUE plan, ADRs (see `docs/CLAUDE.md`) |
| `Cargo.toml` | release profile uses fat LTO + 1 codegen unit |

## Roadmap & tracking

Phased plan in [`docs/03-roadmap.md`](docs/03-roadmap.md). Work is tracked as
GitHub issues: epics (one per phase) with native sub-issues, labelled `phase-*`
and `type:*`. Current phase: **1 — search + eval + UCI** (Phase 0 complete:
perft passes; see [`docs/06-move-generation.md`](docs/06-move-generation.md)).

## Conventions

- Square convention: **a1 = 0, h8 = 63** (file-major, rank-ascending). Bitboard
  bit `i` ⇔ `Square(i)`. Keep this consistent everywhere.
- Commit style: short imperative subject, optional body explaining *why*.
- Branch for non-trivial work; open a PR. Reference issues (`Closes #N`).
