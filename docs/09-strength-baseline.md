# 09 — Strength baseline (measured absolute Elo)

A record of the engine's **measured absolute strength**, so progress has an
external reference point (not just version-vs-version SPRT). Update this when a
release moves the number materially. Method and tooling live in
[07-testing.md](07-testing.md); this file is the results log.

## Latest measurement

| Field | Value |
|-------|-------|
| Date | 2026-06-28 |
| Engine version | v0.5.0 (Phase 2 complete; commit `71db04b`) |
| Eval | hand-crafted (material + PST); NNUE not yet in |
| Hardware | Apple M4 Max (macOS) |
| Match runner | fastchess; opening book `books/openings.epd` |

| Pool | Estimate | Time control | How |
|------|----------|--------------|-----|
| **CCRL (engine) Elo** | **~1989 ± 80** | 8+0.08 | gauntlet vs Cinnamon 2.2a=2071 & 2.3=2212 (x86_64 via Rosetta); Ordo cross-check ≈ 2003 |
| Stockfish `UCI_Elo` | ~1927 | 2+0.02 | `sf` mode (throttled-SF rungs; approximate) |
| Maia (Lichess human) | ~1885–1950 | 5+0.05 | lc0 + maia nets at nodes=1 (engine beats maia-1900) |

All three independently land **~1900–2000**, which is strong corroboration. Quote
the **CCRL ~1989** for engine-vs-engine comparisons; the Maia number for "vs
humans". The two pools are different rating scales and must not be averaged.

### Caveats

- Both CCRL anchors sit *above* the engine (2071, 2212), so ~1989 is mildly
  extrapolated downward — consistent across both anchors and Ordo, but a sub-2000
  ARM/Rosetta-runnable UCI anchor would bracket it better (genuinely weak portable
  UCI engines are scarce; see [07-testing.md](07-testing.md)).
- Elo is **pool-relative** (CCRL / CEGT / SSDF / Lichess / FIDE are not
  interchangeable); time control and hardware are part of any rating.
- Error bars are wide (~±80) at the sample sizes used; raise `ROUNDS` to tighten.

## How to reproduce

```
scripts/setup-anchors.sh                                          # provision anchors (one-time)
MODE=file ANCHORS=scripts/anchors-ccrl.txt ROUNDS=100 scripts/gauntlet.sh 8+0.08   # CCRL
MODE=file ANCHORS=scripts/anchors-maia.txt ROUNDS=100 scripts/gauntlet.sh 5+0.05   # Maia
```

## History

| Date | Version | CCRL est. | Note |
|------|---------|-----------|------|
| 2026-06-28 | v0.5.0 | ~1989 ± 80 | First absolute measurement (Phase 2 complete) |
