# ADR 0003 — NNUE trainer: bullet (with nnue-pytorch as reference)

- **Status**: accepted (forward-looking; applies in Phase 4)
- **Date**: 2026-06-27

## Context

Phase 4 replaces the hand-crafted eval with an NNUE network. Training the net
requires a separate trainer that consumes (position → label) data and emits a
quantized net. The user specifically referenced Stockfish's `nnue-pytorch`.

## Options considered

1. **bullet** (Rust, CUDA) — purpose-built to train nets for *your own* engine's
   architecture/format; used by most current top open-source engines; aligns with
   our Rust workflow. <https://github.com/jw1912/bullet>
2. **nnue-pytorch** (PyTorch) — Stockfish's official trainer; superb as a
   *reference* for how NNUE training works, but tightly coupled to Stockfish's
   specific net architecture and data format.
   <https://github.com/official-stockfish/nnue-pytorch>

## Decision

Use **bullet** as the trainer for our net. Keep **nnue-pytorch** bookmarked as the
canonical *documentation/reference* for NNUE training concepts and quantization.

## Consequences

- Our engine must emit training data in a format bullet ingests (plan datagen
  early; see [04-nnue.md](../04-nnue.md)).
- The engine's eval lives behind a swappable interface so HCE → NNUE is a clean
  substitution.
- Revisit if we adopt a net architecture better supported by another trainer.
