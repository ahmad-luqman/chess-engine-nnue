# 02 — Tech Stack & Tooling

## Language: Rust (decided — see ADR 0001)

- Performance in C++'s class, memory safety, first-class tooling.
- Modern engine peers: Viridithas, Carp, Akimbo (study these).
- `bullet` (the leading modern NNUE trainer) is itself Rust.

## Build & dev tooling

| Tool | Use |
|------|-----|
| `cargo` | build / run / test / bench |
| `cargo build --release` | the ONLY build that matters for speed (LTO on) |
| `clippy` | linting (`cargo clippy`) |
| `criterion` | micro-benchmarks (perft, movegen, eval) |
| `cargo flamegraph` / `samply` | profiling — engine dev IS perf engineering |
| `perf` (Linux) / Instruments (macOS) | CPU profiling |

> Debug builds are 10–50× slower. Always measure in `--release`.

## Correctness testing

| Tool | Use |
|------|-----|
| **Perft suites** | move-gen correctness oracle (known node counts) |
| Built-in `#[test]` | unit tests for each subsystem |
| Tactical suites (WAC, Arasan21, ERET) | sanity: does it find known best moves |

## Strength testing (the heart of serious development)

| Tool | Use |
|------|-----|
| **cutechess-cli** | run automated engine-vs-engine matches |
| **fastchess** | faster modern alternative to cutechess-cli (popular now) |
| **SPRT** | Sequential Probability Ratio Test — decide "is this stronger?" in the fewest games. Non-negotiable for serious dev. |
| **OpenBench** | self-hostable distributed test framework (web UI + worker pool). Aspirational; what top open-source engines use. |
| **Ordo / BayesElo** | compute Elo from match results |

### Why SPRT matters
Most changes that *look* like improvements are noise. SPRT runs games until it
can statistically conclude the change is better (H1) or not (H0), typically with
bounds like `[0, 5]` Elo, `alpha=beta=0.05`. It stops as early as the data allows.
Without it, you will chase ghosts.

## GUIs (for playing/watching, not testing)

- **Cute Chess** (GUI) — also ships `cutechess-cli`.
- **Arena**, **BanksiaGUI**, **En Croissant** — alternatives.

## NNUE training (Phase 4)

| Tool | Use |
|------|-----|
| **bullet** | Rust, CUDA-accelerated NNUE trainer. Default choice for new engines. |
| **nnue-pytorch** | Stockfish's official PyTorch trainer. Best *documentation* of how NNUE training works; tightly coupled to SF's net format. |
| Self-play data | Our engine generates training positions + scores by playing millions of fast games. |

## Opening books & endgame tablebases (later)

- **Polyglot** (`.bin`) opening books for variety in testing.
- **Syzygy** tablebases for perfect endgame play (≤7 pieces). Probed in search.
- Standard test opening sets (e.g. `Pohl`, `UHO`, `8moves_v3`) for balanced
  engine-vs-engine games.

## Reference engines to read (Rust, clean, modern)

- **Viridithas** — well-documented, strong, NNUE, great learning target.
- **Carp**, **Akimbo**, **Svart** — smaller, readable.
- **Stockfish** (C++) — the gold standard, but advanced.
