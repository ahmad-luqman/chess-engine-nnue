# NNUE trainer (issue #45)

The `bullet`-based training pipeline for chess-engine-nnue's **first NNUE**, plus
a GPU-free reference inference / sanity check. This is a **separate workspace** from
the engine on purpose: the engine crate has zero runtime dependencies, and bullet
(heavy, GPU-backed) must never leak into it.

```
trainer/
  bullet-train/   bin `train`  — the bullet trainer (needs a GPU backend feature)
  verify/         bin `verify` — pure-std reference inference + sanity check (no GPU)
```

- **Architecture**: `(768 → 256)×2 → 1×8`, SCReLU, `Chess768` perspective inputs
  (king-agnostic, NOT HalfKP). `QA=255`, `QB=64`, `SCALE=400`, `ConstantWDL 0.75`.
- **Net file layout + inference arithmetic**: specified in
  [`docs/decisions/0016-nnue-first-net-architecture.md`](../docs/decisions/0016-nnue-first-net-architecture.md).
  `verify/src/main.rs` is the reference implementation that engine wiring (#46) copies.
- bullet is pinned to a fixed commit in `bullet-train/Cargo.toml` for reproducibility.

## Prerequisites

- Rust (stable).
- A bullet backend toolchain for **`train`** only:
  - **Metal** (Apple Silicon): nothing extra; build with `--features metal`.
  - **CUDA** (NVIDIA): install the CUDA Toolkit, set `CUDA_PATH`; build with
    `--features cuda`.
- `verify` needs **no** GPU and no backend feature.

## Recipe

### 1. Get training data (public binpacks — the bootstrap source)

The first net is bootstrapped on public Stockfish/Leela binpacks (SF binpack
format, read directly by `SfBinpackLoader` with a standard quiet-position filter —
no conversion needed). A small official sample (~60 MB, used for the committed
proof-of-life net):

```sh
mkdir -p bullet-train/data
curl -L -o bullet-train/data/test77.binpack \
  https://huggingface.co/datasets/official-stockfish/master-smallnet-binpacks/resolve/main/test77-jan2022-2tb7p.high-simple-eval-1k.min-v2.binpack
```

For a strong first net, use a full monthly set instead (multi-GB, `.zst` — decompress
with `zstd -d`), e.g. from <https://robotmoon.com/nnue-training-data/> or
`huggingface.co/datasets/linrock/test80-2024`. Point `NNUE_DATA` at any `.binpack`.

### 2. Train

`train` is configured by environment variables (defaults = the full schedule):

| Env var | Default | Meaning |
|---------|---------|---------|
| `NNUE_DATA` | `data/bootstrap.binpack` | path to a `.binpack` (run from `bullet-train/`) |
| `NNUE_NET_ID` | `chess-engine-nnue-0001` | checkpoint name |
| `NNUE_SUPERBATCHES` | `40` | end superbatch |
| `NNUE_BATCHES_PER_SB` | `6104` | batches per superbatch (~100M positions) |
| `NNUE_THREADS` | `4` | data-loader threads |
| `NNUE_SAVE_RATE` | `10` | checkpoint every N superbatches |

```sh
cd bullet-train
# Apple Silicon (Metal):
NNUE_DATA=data/test77.binpack cargo run -r -p bullet-train --features metal
# NVIDIA (CUDA):
NNUE_DATA=data/test80.binpack cargo run -r -p bullet-train --features cuda
```

Checkpoints land in `bullet-train/checkpoints/<net_id>-<superbatch>/`; the
quantised net is `quantised.bin` there. bullet skips writing `quantised.bin` only
if quantisation overflows (it won't with `QA=255` and width 256).

> Proof-of-life net committed as `nets/chess-engine-nnue-0001.bin` was trained on
> the ~60 MB `test77` sample, `NNUE_SUPERBATCHES=30 NNUE_BATCHES_PER_SB=1000` on an
> Apple M4 Max (~30 s, Metal). Strength is **not** the goal here (#45) — a
> non-degenerate, correctly-mapped net is. Calibration/strength is #48/#49.

### 3. Verify (no GPU — run anywhere)

```sh
cargo run -r -p verify -- bullet-train/checkpoints/chess-engine-nnue-0001-30/quantised.bin
```

Asserts the net loads and that evals are finite, material-monotonic (up a queen ≫
startpos ≫ down a queen), and **perspective-consistent** (a colour-mirrored
position evaluates identically — this catches feature-flip / concat-order bugs).

Mirror/material checks are flip-*direction*-blind, so the feature mapping is also
cross-validated against bullet's own `Chess768` on black-to-move positions:

```sh
cargo run -r -p bullet-train --features metal --bin featcheck
```

### 4. Commit the net

```sh
cp bullet-train/checkpoints/chess-engine-nnue-0001-30/quantised.bin \
   ../nets/chess-engine-nnue-0001.bin
```

`nets/chess-engine-nnue-0001.bin` is allowlisted in the repo `.gitignore`.

## Scaling up: training on our own self-play data

Once the loop is turning, swap public binpacks for our own data. `examples/gen_data.rs`
(issue #44) already emits bullet's **text** format (`FEN | white_cp | white_result`).
Convert it to bulletformat, shuffle, interleave, then train with
`DirectSequentialDataLoader` (build `bullet-utils` from the pinned bullet checkout —
it needs no GPU):

```sh
# from the engine repo root:
cargo run --release --example gen_data -- 20000000 8 1 bullet > data/nnue_s1.txt
# ... repeat with different seeds for diversity ...
bullet-utils convert  --from text --input data/nnue_s1.txt --output data/nnue_s1.bin
bullet-utils shuffle  --input data/nnue_s1.bin --output data/nnue_s1.shuf.bin --mem-used-mb 4096
bullet-utils interleave data/nnue_s1.shuf.bin data/nnue_s2.shuf.bin --output data/train.bin
```

Then point the trainer at the bulletformat file — switch the loader in
`bullet-train/src/main.rs` from `SfBinpackLoader` to
`DirectSequentialDataLoader::new(&["data/train.bin"])`. (Both score and result in
the text are **white-relative**; bullet re-orients to side-to-move internally.)
