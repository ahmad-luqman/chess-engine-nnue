# Project Documentation

This folder is the engine's long-term memory: research, design decisions, and
the roadmap. It is meant to outlive any single coding session.

## Index

| Doc | Purpose |
|-----|---------|
| [00-overview.md](00-overview.md) | What we're building and the mental model |
| [01-research-landscape.md](01-research-landscape.md) | How real engines work; techniques survey |
| [02-tech-stack.md](02-tech-stack.md) | Languages, tools, libraries, testing infra |
| [03-roadmap.md](03-roadmap.md) | Phased plan with Elo milestones |
| [04-nnue.md](04-nnue.md) | The NNUE endgame: how neural eval works + training pipeline |
| [05-board-representation.md](05-board-representation.md) | Deep dive: bitboards + mailbox, and why the layout helps |
| [06-move-generation.md](06-move-generation.md) | Deep dive: move encoding, legal movegen, make/unmake, perft |
| [decisions/](decisions/) | Architecture Decision Records (ADRs) — the *why* |
| [resources.md](resources.md) | Curated learning links |

## How to use this

- Before a big change, skim the relevant doc.
- When you make a non-obvious choice, add an ADR (copy `decisions/0000-template.md`).
- Keep the roadmap's "current phase" marker honest.
