# 03 — Roadmap

Phased plan. Each phase ends in a *playable, testable* milestone. Elo figures
are rough order-of-magnitude targets, not promises.

> **Current phase: 0 — Board representation**

| Phase | Goal | Deliverable | Rough Elo |
|-------|------|-------------|-----------|
| **0** | Board + legal move gen + **perft passing** | Provably-correct movegen | — |
| **1** | Negamax + alpha-beta + material eval + UCI | Plays legal, non-losing chess in a GUI | ~1000–1500 |
| **2** | Iterative deepening, TT, move ordering, quiescence | A real engine; beats most humans | ~2000–2300 |
| **3** | Pruning (NMP/LMR/futility) + tuned hand-crafted eval | Strong classical engine | ~2600–2900 |
| **4** | Replace eval with **NNUE** | A modern engine | ~3200+ |
| **5** | Lazy SMP, search tuning, SPRT grind, own NNUE training | "Serious engine" | ↑ |

## Phase 0 — detailed checklist (current)

- [ ] Core types: `Color`, `PieceType`, `Piece`, `Square`, `Bitboard`
- [ ] Board struct: piece bitboards + occupancy + mailbox + state
      (side to move, castling rights, ep square, halfmove clock)
- [ ] FEN parsing (set up arbitrary positions)
- [ ] Attack tables: knight, king (precomputed); pawns
- [ ] Sliding attacks: start simple (ray loops), magics later
- [ ] Move encoding (pack from/to/flags into a compact int)
- [ ] Legal move generation
- [ ] make / unmake with undo stack
- [ ] **Perft** — match published node counts for standard positions
- [ ] Perft test suite wired into `cargo test`

### Phase 0 exit criterion
`perft` matches the known values for at least these positions, to depth 5–6:
start position, Kiwipete, and the standard "position 3/4/5" perft test set.

## Working method (applies every phase from 1 on)

1. Make one change.
2. Run perft/tests (correctness).
3. SPRT vs the previous version (strength).
4. Keep only changes that pass. Tag releases so we always have opponents.
