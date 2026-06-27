# docs/ — project knowledge base

The engine's long-term memory. Meant to outlive any single coding session.

## What goes where

| File | Purpose |
|------|---------|
| `00-overview.md` | mental model + the six subsystems + iron rules |
| `01-research-landscape.md` | survey of techniques real engines use |
| `02-tech-stack.md` | language, tooling, testing & training infra |
| `03-roadmap.md` | phased plan + Elo milestones; tracks current phase |
| `04-nnue.md` | NNUE architecture + training pipeline |
| `decisions/` | ADRs — the *why* behind non-obvious choices |
| `resources.md` | curated external links |

## Rules for editing docs

- **New non-obvious decision → new ADR.** Copy `decisions/0000-template.md`,
  number it sequentially, set status. Never silently reverse a past ADR; add a
  new one marked "supersedes ADR-XXXX".
- **Keep the roadmap's current-phase marker honest** — update it when a phase
  completes.
- Docs describe *intent and reasoning*; code describes *mechanism*. If they
  disagree, the code is the truth and the doc is a bug — fix the doc.
- Keep entries terse and link-rich rather than exhaustive; the Chess Programming
  Wiki is the deep reference (see `resources.md`).
