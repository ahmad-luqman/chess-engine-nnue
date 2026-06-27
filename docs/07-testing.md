# 07 — Testing & strength measurement

Two kinds of testing keep this engine honest, and they answer different
questions:

- **Correctness** — *does it follow the rules and the spec?* `cargo test` (unit
  tests + the perft gate) and `cargo clippy`. Run these on every change.
- **Strength** — *is this version actually stronger than the last?* Engine-vs-
  engine matches with **SPRT**. Required for every change that's meant to gain
  Elo, from Phase 1 on (iron rule #3). This doc is mostly about this half.

> Always measure strength with a **release** build (iron rule #2). Debug is
> 10–50× slower and its results are meaningless.

## The engine as a UCI program

The release binary speaks UCI on stdin/stdout — that's all a match runner needs:

```
cargo build --release         # produces ./target/release/engine
./target/release/engine       # then type: uci / position / go / quit
```

Quick manual smoke test (the engine should answer and then move):

```
printf 'uci\nisready\nposition startpos\ngo movetime 500\nquit\n' \
  | ./target/release/engine
```

## Match runner: cutechess-cli

We use **cutechess-cli** for matches and SPRT (see
[ADR 0004](decisions/0004-cutechess-testing.md)). It is not bundled with this
repo; install it locally.

### Install (macOS)

cutechess needs Qt, so the simplest routes are:

- **Build from source** — clone <https://github.com/cutechess/cutechess>, then
  build per its README (needs Qt + CMake). The CLI binary is `cutechess-cli`.
- **fastchess** (easier to build, no Qt; near-identical flags) —
  <https://github.com/Disservin/fastchess>. A fine substitute everywhere below;
  swap the binary name and adjust flag spellings per its docs.

Verify whichever you installed is on `PATH`:

```
cutechess-cli --version      # or: fastchess --version
```

### A sanity game (two builds, a few games)

Keep a known-good **baseline** binary as the opponent (a tagged release — see
below). Then:

```
cutechess-cli \
  -engine cmd=./target/release/engine name=new \
  -engine cmd=./baseline/engine        name=base \
  -each proto=uci tc=10+0.1 \
  -games 2 -rounds 1 \
  -pgnout sanity.pgn
```

`tc=10+0.1` is 10 seconds + 0.1s/move. Open `sanity.pgn` and confirm the games
are legal and complete — that alone is the **Phase 1 (#22) exit check**: the
engine plays a full legal game via UCI against another engine.

### The real test: SPRT

SPRT plays games until it can conclude (with bounded error) whether the new build
gained Elo, then stops — far more efficient than a fixed N games.

```
cutechess-cli \
  -engine cmd=./target/release/engine name=new \
  -engine cmd=./baseline/engine        name=base \
  -each proto=uci tc=8+0.08 \
  -openings file=book.epd format=epd order=random \
  -repeat -games 2 -rounds 5000 -concurrency 4 \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
  -pgnout sprt.pgn
```

- `-repeat -games 2` plays each opening twice with colours reversed (fairness).
- `-sprt elo0=0 elo1=5` tests "no gain" vs "≥5 Elo gain" at 5% error each way —
  a typical bar for a small improvement. The run ends when a hypothesis is
  accepted.
- `-openings` needs a small opening book (an `.epd`/`.pgn` of start positions) so
  games aren't all the same line; grab any standard book (e.g. a Pohl/8-moves
  EPD) and point `file=` at it.
- `-concurrency` to taste (≈ physical cores); strength is independent of it.

## The working method (every change, Phase 1+)

1. Make one change.
2. `cargo test` + `cargo clippy` — correctness.
3. SPRT vs the previous tagged build — strength.
4. Keep the change only if SPRT passes; otherwise revert.
5. When a gainer lands, **tag a release** and rebuild the baseline from it, so
   there's always an opponent to measure the next change against.

```
git tag -a v0.1.0 -m "first playable: search + eval + UCI"
# build & stash the baseline binary the next SPRT will play against
cargo build --release && mkdir -p baseline && cp target/release/engine baseline/
```

## See also

- [03-roadmap.md](03-roadmap.md) — the phased plan and the working method.
- [ADR 0004](decisions/0004-cutechess-testing.md) — why cutechess-cli.
- Chess Programming Wiki: *Match*, *SPRT* (linked from [resources.md](resources.md)).
