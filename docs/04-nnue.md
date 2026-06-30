# 04 — NNUE: the evaluation endgame

NNUE = **Efficiently Updatable Neural Network**. A small net, evaluated on the
**CPU**, that replaces the hand-crafted evaluation. It is the repo's namesake and
a Phase-4 concern — documented now so the earlier architecture leaves room for it.

## Why "efficiently updatable"

The net's first (largest) layer depends only on which pieces are on which squares.
Its input is a huge sparse one-hot vector of `(piece, square, king-square)`
features. The first layer's output (the **accumulator**) is a sum of the columns
for the active features.

When you make a move, only a few features change → you **add/subtract a handful of
columns** from the accumulator instead of recomputing it. That incremental update
is what makes a neural eval fast enough to run millions of times per second in an
alpha-beta search. This is the whole trick.

## Our first architecture: plain `768`-input perspective net

The first net is deliberately **not** HalfKP/HalfKA. The 2026 consensus (bullet
docs, Viridithas) is to start with the simplest net that beats a tuned HCE and
add complexity only when SPRT says it gains — a HalfKP net needs far more data and
king-bucket machinery, and a beginner usually *loses* Elo with it. So:

```
768 one-hot features ──► [768 → 256 linear] ──► accumulator (per side)
   (piece, square)          (incrementally updated)
                                  │
                            SCReLU  (clamp(x,0,1)²)
                                  │
   concat(stm, opponent) ──► [512 → 1, ×8 buckets] ──► scalar eval (centipawns)
```

- **Inputs**: `768 = 2 colours × 6 pieces × 64 squares`, index `64·piece + square`.
  King-agnostic — no accumulator refresh on king moves.
- **Perspective**: two accumulators (side-to-move and opponent), concatenated
  **stm first**.
- **Output buckets**: 8, selected by piece count (`(count − 2) / 4`).
- **Quantization**: weights are `int16` (`QA=255`, `QB=64`, `SCALE=400`); eval uses
  integer arithmetic (SCReLU squared/accumulated in `int32`). No floats at
  inference. This is why it's CPU-fast.

The exact architecture, quantisation, net file layout, and inference arithmetic are
specified in [ADR 0016](decisions/0016-nnue-first-net-architecture.md) — that ADR
is the contract the inference code (#46) must match. HalfKP/HalfKA remains a
later, SPRT-gated option.

## Inference (`src/nnue.rs`, #46)

The forward pass lives in `src/nnue.rs`, behind the swappable `Evaluator` trait.
Two accumulators (one per perspective colour) are kept in `i16` and updated
**incrementally** from the `DirtyPiece` deltas (#43) on make/unmake — adding or
subtracting a few weight columns instead of recomputing. The committed net is
embedded with `include_bytes!`. Design decisions (colour-keyed accumulators,
`DirtyPiece`-driven update, the from-scratch trait seam, and the perft-walk
correctness guard) are in [ADR 0017](decisions/0017-nnue-inference-incremental-accumulator.md).
Search still uses the hand-crafted eval; flipping it to NNUE under SPRT is #48.

## The training pipeline (the loop)

```
   ┌─────────────────────────────────────────────────────┐
   │ 1. Engine plays millions of fast self-play games     │
   │ 2. Record positions + labels (search score and/or    │
   │    game result)                                      │
   │ 3. Trainer (bullet) learns a net from (pos → label)  │
   │ 4. Quantize + embed net in engine                    │
   │ 5. SPRT-test new net vs old; keep if stronger        │
   │ 6. Stronger engine generates better data → repeat    │
   └─────────────────────────────────────────────────────┘
```

Bootstrapping note: Stockfish's first NNUE nets were trained on labels from
Stockfish's *classical* (hand-crafted) eval, then iteratively improved with
self-play. Our Phase 1–3 hand-crafted eval is therefore not throwaway — it's the
**label source** that bootstraps the first net.

## Tooling

- **bullet** — Rust NNUE trainer (CUDA, ROCm, **and Metal** backends — so it
  trains locally on Apple Silicon as well as on a CUDA box). Default for new
  engines; matches a Rust-engine workflow. <https://github.com/jw1912/bullet>
- **nnue-pytorch** — Stockfish's PyTorch trainer; best-documented reference for
  the *concepts* and SF's exact format. <https://github.com/official-stockfish/nnue-pytorch>

Our trainer setup and the **reproducible training recipe** (data → train → export
→ verify, for both Metal and CUDA) live in [`trainer/README.md`](../trainer/README.md).
`examples/gen_data.rs` already emits bullet's text format for self-play data (#44);
the bootstrap net is trained on public Stockfish/Leela binpacks.

## What this means for earlier phases (design with NNUE in mind)

- Keep **evaluation behind a trait/interface** so HCE and NNUE are swappable.
- Make **make/unmake** expose exactly which `(piece, square)` features changed —
  the NNUE accumulator update will hook here.
- Track **king squares** cleanly (king-relative features need them).
- Have a fast **self-play / datagen** mode early; it's reused to make NNUE data.
