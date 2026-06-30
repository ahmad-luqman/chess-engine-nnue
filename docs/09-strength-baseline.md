# 09 — Strength baseline (measured absolute Elo)

A record of the engine's **measured absolute strength**, so progress has an
external reference point (not just version-vs-version SPRT). Update this when a
release moves the number materially. Method and tooling live in
[07-testing.md](07-testing.md); this file is the results log.

## Latest measurement

| Field | Value |
|-------|-------|
| Date | 2026-06-30 |
| Engine version | `v0.5.0-32-g82bf900` (Phase 3 complete; NNUE not yet wired) |
| Eval | hand-crafted, Texel-tuned (tapered material + PST + terms); NNUE not yet in |
| Hardware | Apple M4 Max (macOS) |
| Match runner | fastchess `tc=10+0.1`, 50 rounds/anchor (×2 games); book `books/openings.epd` |

| Pool | Estimate | How |
|------|----------|-----|
| **CCRL (engine) Elo** | **~2449** (table ~2403) | gauntlet vs Cinnamon 2.2a=2071 & 2.3=2212 (x86_64 via Rosetta); headline is the **Ordo full-PGN cross-check** pinned to cinnamon2.3 |
| Stockfish `UCI_Elo` | **> 2320** | `sf` rungs; sweeps ≤1720 (100%), beats sf2320 ~80% — point estimate noisy and SF's scale runs hot |
| Maia (Lichess human) | **≳ 1900** (ceiling-limited) | sweeps every net maia-1100…1900; Ordo caps at ~1895 because maia-1900 is the top anchor |

**The engine has outgrown the current anchor set** — it sweeps everything up
through ~2200–2300, so only cinnamon2.3 (2212) produces enough losses to anchor a
real number. Treat **~2400–2450 CCRL** as the figure for engine-vs-engine
comparison; quote the Maia number only as a "≳1900 vs humans" floor. Pools are
different rating scales and must not be averaged.

This is **~+460 over the Phase-2 v0.5.0 baseline (~1989)** — entirely from the
Phase-3 selective-search + tuned eval (#34–#42). The later DirtyPiece plumbing
(#43, −4% NPS) and NNUE datagen (#44) cost no measurable strength, as intended.

### Caveats

- **Anchors no longer bracket from above** (except cinnamon2.3, barely), so the
  CCRL number is mildly extrapolated *upward* and the per-anchor table estimates are
  wide. Add stronger CCRL anchors (~2400–2700) to pin it tightly next time.
- Trust the **Ordo full-PGN cross-check over the per-anchor table** once the engine
  dominates: the table reads each match's *final* fastchess block, but near-sweeps
  still yield `± nan` there (no point estimate), whereas Ordo solves from all games.
- Elo is **pool-relative** (CCRL / CEGT / SSDF / Lichess / FIDE are not
  interchangeable); time control and hardware are part of any rating.

## How to reproduce

```
scripts/setup-anchors.sh                                          # provision anchors (one-time)
MODE=file ANCHORS=scripts/anchors-ccrl.txt scripts/gauntlet.sh    # CCRL  (tc/rounds = gauntlet defaults)
MODE=sf                                     scripts/gauntlet.sh    # Stockfish UCI_Elo rungs
MODE=file ANCHORS=scripts/anchors-maia.txt scripts/gauntlet.sh    # Maia
```

## History

| Date | Version | CCRL est. | Note |
|------|---------|-----------|------|
| 2026-06-28 | v0.5.0 | ~1989 ± 80 | First absolute measurement (Phase 2 complete) |
| 2026-06-30 | `v0.5.0-32-g82bf900` | ~2449 (Ordo) | Phase 3 complete; +~460. Engine has outgrown the anchor set (sweeps ≤~2300) |
