# 01 — Research: How real engines work

A survey of the techniques used by top engines, organized by subsystem, so we
know what we're building toward. Depth-of-detail here is intentionally a
reference, not a tutorial — see the Chess Programming Wiki for full treatments.

## Board representation

- **Bitboards** — one `u64` per piece type/color; the board is a set of bitmasks.
  A set bit = a piece on that square. Move generation becomes bit manipulation
  (shifts, ANDs, ORs). Universal in strong engines. Fast `popcount`/`trailing_zeros`
  map to single CPU instructions.
- **Mailbox / 0x88** — array-of-squares representations. Simpler to learn, slower.
  We use bitboards but a mailbox `piece_on(square)` array alongside is common and
  convenient.
- **Make/unmake** vs **copy/restore** — we use make/unmake with an undo stack
  (store captured piece, castling rights, en-passant square, halfmove clock).

### Sliding-piece attacks
- **Magic bitboards** — precomputed perfect-hash lookup for rook/bishop attacks
  given a blocker occupancy. The standard fast technique. ("Fancy" and "black"
  magic are variants.)
- **PEXT bitboards** — use the BMI2 `pext` instruction instead of magics (faster
  on Intel, historically slow on some AMD). Optional optimization.
- Simpler stand-ins to start: **Kogge–Stone** fills or plain ray loops. Correct
  but slower; fine for Phase 0, replace later.

### Hashing
- **Zobrist hashing** — XOR a random 64-bit key per (piece, square), side to move,
  castling rights, ep file. Incrementally updated each move. Used by the
  transposition table and for repetition detection.

## Move generation

- Generate pseudo-legal moves, then filter for legality (king not in check), OR
  generate fully legal directly using pin/check masks (faster, harder).
- Tricky cases that perft will catch: **en passant** (incl. the rare ep-pin
  discovery), **castling** through/into check, **promotions** (4 per pawn),
  **double check** (only king moves).

## Search (alpha-beta family — Stockfish/Dragon)

Foundations:
- **Negamax** — single-perspective minimax (`score = -search(-β, -α)`).
- **Alpha-beta pruning** — skip branches that can't affect the result. The whole
  game. Effectiveness depends entirely on move ordering.
- **Iterative deepening (ID)** — search depth 1, 2, 3, … reusing prior results
  for move ordering and time control.

Caching & ordering:
- **Transposition table (TT)** — large hash map of position → (score, depth, best
  move, bound type). Zobrist-keyed. Huge speedup.
- **Move ordering** — TT move first, then captures by **MVV-LVA**, **killer moves**,
  **history heuristic**. Good ordering is what makes alpha-beta cut.

Selectivity (the modern strength comes from here):
- **Quiescence search** — at depth 0, keep searching captures until "quiet" so the
  eval isn't called mid-trade (the horizon effect).
- **Null-move pruning (NMP)** — give the opponent a free move; if still winning,
  prune.
- **Late move reductions (LMR)** — search later (likely worse) moves shallower.
- **Futility / reverse-futility / razoring** — skip moves unlikely to raise alpha.
- **Aspiration windows** — search ID with a narrow window around the last score.
- **Check / singular extensions** — search deeper in forcing lines.

## Evaluation

- **Hand-crafted (HCE)** — material + piece-square tables + pawn structure, king
  safety, mobility, etc. Often **tapered** between middlegame/endgame. This is
  Phase 1–3.
- **Texel tuning** — fit HCE weights by logistic regression against game results.
- **NNUE** — small neural net, CPU-evaluated, **incrementally updated**. Replaces
  HCE in Phase 4. See [04-nnue.md](04-nnue.md).

## Leela / AlphaZero branch (not our path, for context)

- Deep conv/transformer net outputs a policy (move priors) + value (win prob).
- **MCTS** (PUCT) instead of alpha-beta. GPU-bound.
- Trained by **reinforcement learning self-play** from zero knowledge.
- We borrow ideas (self-play data generation) but not the architecture.

## Time management & threading (later)

- Allocate time per move from clock + increment; stop early on stable PV.
- **Lazy SMP** — the dominant multithreading scheme: many threads search the same
  tree sharing a TT. Simple and effective. (Phase 5.)
