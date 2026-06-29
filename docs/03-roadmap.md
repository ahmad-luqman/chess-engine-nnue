# 03 — Roadmap

Phased plan. Each phase ends in a *playable, testable* milestone. Elo figures
are rough order-of-magnitude targets, not promises.

> **Current phase: 4 — NNUE evaluation.** Phases 0–3 complete. Phase 3 shipped the
> selective-search core and a tuned hand-crafted eval (issues #34–#42), each
> SPRT-verified — see [08-search-internals.md](08-search-internals.md) and the ADRs.
> Search side: PVS, LMR, null-move pruning, SEE + check extensions (**+47.95**,
> **+26.73**), reverse-futility (**+76.92**) + futility (**+110.53**) pruning, and
> aspiration windows (**+53.45**). Eval side: tapered eval, hand-crafted terms, and
> Texel tuning. Phase 2 (v0.5.0) shipped Zobrist hashing, a transposition table,
> move ordering, quiescence, draw detection, and magic bitboards (issues #23–#28).

| Phase | Goal | Deliverable | Rough Elo |
|-------|------|-------------|-----------|
| **0** | Board + legal move gen + **perft passing** | Provably-correct movegen | — |
| **1** | Negamax + alpha-beta + material eval + UCI | Plays legal, non-losing chess in a GUI | ~1000–1500 |
| **2** | Iterative deepening, TT, move ordering, quiescence | A real engine; beats most humans | ~2000–2300 |
| **3** | Pruning (NMP/LMR/futility) + tuned hand-crafted eval | Strong classical engine | ~2600–2900 |
| **4** | Replace eval with **NNUE** | A modern engine | ~3200+ |
| **5** | Lazy SMP, search tuning, SPRT grind, own NNUE training | "Serious engine" | ↑ |

## Phase 0 — detailed checklist (complete)

- [x] Core types: `Color`, `PieceType`, `Piece`, `Square`, `Bitboard`
- [x] Board struct: piece bitboards + occupancy + mailbox + state
      (side to move, castling rights, ep square, halfmove clock)
- [x] FEN parsing (set up arbitrary positions)
- [x] Attack tables: knight, king (precomputed); pawns
- [x] Sliding attacks: start simple (ray loops), magics later
- [x] Move encoding (pack from/to/flags into a compact int)
- [x] Legal move generation
- [x] make / unmake with undo stack
- [x] **Perft** — match published node counts for standard positions
- [x] Perft test suite wired into `cargo test`

### Phase 0 exit criterion
`perft` matches the known values for at least these positions, to depth 5–6:
start position, Kiwipete, and the standard "position 3/4/5" perft test set.

## Working method (applies every phase from 1 on)

1. Make one change.
2. Run perft/tests (correctness).
3. SPRT vs the previous version (strength).
4. Keep only changes that pass. Tag releases so we always have opponents.
