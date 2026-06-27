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

## Typical architecture (HalfKP / HalfKA-style)

```
sparse features ──► [big linear layer] ──► accumulator (per side)
   (king-relative)      (incrementally updated)
                              │
                        clipped ReLU
                              │
                     [small dense layers] ──► scalar eval (centipawns)
```

- **Perspective**: two accumulators (side-to-move and opponent), concatenated.
- **Quantization**: weights are `int8`/`int16`; eval uses integer SIMD (AVX2/AVX-512
  / NEON). No floats at inference. This is why it's CPU-fast.

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

- **bullet** — Rust, CUDA NNUE trainer. Default for new engines; matches a
  Rust-engine workflow. <https://github.com/jw1912/bullet>
- **nnue-pytorch** — Stockfish's PyTorch trainer; best-documented reference for
  the *concepts* and SF's exact format. <https://github.com/official-stockfish/nnue-pytorch>

## What this means for earlier phases (design with NNUE in mind)

- Keep **evaluation behind a trait/interface** so HCE and NNUE are swappable.
- Make **make/unmake** expose exactly which `(piece, square)` features changed —
  the NNUE accumulator update will hook here.
- Track **king squares** cleanly (king-relative features need them).
- Have a fast **self-play / datagen** mode early; it's reused to make NNUE data.
