# 00 — Overview

## Goal

Build a chess engine that starts as a **learning exercise** and matures into a
**serious, competitive engine** over time. The repo name (`chess-engine-nnue`)
encodes the endgame: an engine with an NNUE (neural) evaluation, in the lineage
of Stockfish / Komodo Dragon.

## The one mental model that matters

A strong chess engine is **not a GUI**. It is a headless program that speaks a
text protocol over stdin/stdout. A separate GUI or tournament manager drives it.

```
┌──────────────┐   UCI text    ┌──────────────┐
│  GUI / Arena │ ───────────►  │  Our engine   │
│ (Cute Chess) │ ◄───────────  │  (a binary)   │
└──────────────┘   over stdio  └──────────────┘
```

The protocol is **UCI (Universal Chess Interface)**. Because every serious
engine speaks it, ours is a drop-in part: it can play in any GUI, and engines
can play each other automatically — which is the basis of all serious testing.

## The six subsystems

Build in this order. Each is testable before the next exists.

1. **Board representation** — store position, make/unmake moves (bitboards).
2. **Move generation** — list all legal moves.
3. **Perft** — count leaf nodes; the *correctness oracle* for 1–2.
4. **Search** — look ahead, pick a move (alpha-beta).
5. **Evaluation** — score a quiet position (hand-crafted → NNUE).
6. **UCI loop** — talk to GUIs/tournaments; manage time.

## Iron rule

**No search before perft passes.** Perft counts legal move sequences to a depth;
the correct values are published. If ours don't match exactly, there is a
move-generation bug — found now, cheaply, instead of as phantom blunders later.

## Two engine philosophies (context for our choice)

- **Stockfish / Dragon**: alpha-beta search + **NNUE** eval on CPU. ← our path
- **Leela (Lc0)**: deep neural net + **MCTS**, GPU, AlphaZero-style self-play.

We take the Stockfish-style path: bitboards → alpha-beta → hand-crafted eval →
replace eval with NNUE. It is the more tractable route to high strength on a CPU
and matches what most modern strong open-source engines do.
