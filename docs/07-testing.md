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

Two routes, both SPRT-capable with near-identical flags:

- **fastchess** — easier on macOS (C++/Makefile, **no Qt**); this is what's
  installed on this dev machine. Clone <https://github.com/Disservin/fastchess>,
  `make -j`, and put the `fastchess` binary on `PATH` (e.g. symlink it into
  `/opt/homebrew/bin`). The commands below use fastchess syntax.
- **cutechess-cli** — the lineage standard (see
  [ADR 0004](decisions/0004-cutechess-testing.md)), but needs Qt: clone
  <https://github.com/cutechess/cutechess> and build per its README. Flags match
  except `-pgnout file=…` becomes `-pgnout …`.

Verify it's on `PATH`:

```
fastchess --version          # or: cutechess-cli --version
```

### A sanity game (two builds, a few games)

Keep a known-good **baseline** binary as the opponent (a tagged release — see
below). Then:

```
fastchess \
  -engine cmd=./target/release/engine name=new \
  -engine cmd=./baseline/engine        name=base \
  -each proto=uci tc=10+0.1 \
  -games 2 -rounds 1 \
  -pgnout file=sanity.pgn
```

`tc=10+0.1` is 10 seconds + 0.1s/move. Open `sanity.pgn` and confirm the games
are legal and complete — that alone is the **Phase 1 (#22) exit check**: the
engine plays a full legal game via UCI against another engine. (First run, the
engine vs itself, did exactly this: 87 plies ending in checkmate.)

### The real test: SPRT

SPRT plays games until it can conclude (with bounded error) whether the new build
gained Elo, then stops — far more efficient than a fixed N games.

**One command** ([`scripts/sprt.sh`](../scripts/sprt.sh), issue #33) wraps the whole
recipe — build the candidate, obtain the baseline opponent, ensure the book, run
the match, and print the LLR / Elo / accept-reject verdict:

```
scripts/sprt.sh v0.5.0              # candidate vs the v0.5.0 baseline at tc=5+0.05
scripts/sprt.sh v0.5.0 8+0.08       # ... at a slower time control
scripts/sprt.sh baseline/engine     # ... vs an already-built binary (skip the tag rebuild)
```

The baseline argument is either a **git tag** — built fresh in a throwaway git
worktree and stashed at `baseline/engine`, so it's exactly reproducible from
version control — or a **path** to an existing binary, used as-is. Bounds and
match size are env-overridable (`ELO0 ELO1 ALPHA BETA ROUNDS GAMES CONCURRENCY
BOOK`); the verdict is decided from the final `LLR` against its bounds
(`±2.94` for `alpha=beta=0.05`): `PASS` (H1, promote), `FAIL` (H0, revert), or
`INCONCLUSIVE` (raise `ROUNDS`). The exit code is non-zero on `FAIL`.

Under the hood it runs exactly this fastchess invocation:

```
fastchess \
  -engine cmd=./target/release/engine name=new \
  -engine cmd=./baseline/engine        name=base \
  -each proto=uci tc=5+0.05 \
  -openings file=books/openings.epd format=epd order=random \
  -repeat -rounds 5000 -games 2 -concurrency 8 \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
  -pgnout file=sprt.pgn
```

- `-repeat -games 2` plays each opening twice with colours reversed (fairness).
- `-sprt elo0=0 elo1=5` tests "no gain" vs "≥5 Elo gain" at 5% error each way —
  a typical bar for a small improvement. The run ends when a hypothesis is
  accepted.
- `-openings` needs an opening book (an `.epd`/`.pgn` of start positions) so
  games aren't all the same line. We commit a curated one at
  [`books/openings.epd`](../books/openings.epd) — ~24 balanced mainlines spanning
  1.e4 / 1.d4 / 1.c4 / 1.Nf3 (issue #31). With `-repeat -games 2` each is played
  once from each side, so any small opening imbalance cancels. Regenerate it (to
  add or curate lines, edit `OPENINGS` in `examples/genbook.rs` first) with:

  ```
  cargo run --release --example genbook > books/openings.epd
  ```

  The generator plays each line through the engine's own `generate_legal`, so an
  illegal or mistyped move panics by name rather than emitting a bad position.
- `-concurrency` to taste (≈ physical cores); strength is independent of it.

### Absolute Elo: gauntlet vs rated anchors

SPRT gives *relative* Elo (candidate vs the previous version) — the scale floats,
with no tie to any external number. To answer "roughly how strong are we on a
public scale?" you play a **gauntlet** against opponents whose rating is already
known, then read your rating off relative to theirs.
[`scripts/gauntlet.sh`](../scripts/gauntlet.sh) does this:

```
scripts/gauntlet.sh                                    # Stockfish rungs 1320..2400 @ 10+0.1
MIN_ELO=1400 MAX_ELO=2000 STEP=100 scripts/gauntlet.sh # tighter, lower band
MODE=file ANCHORS=scripts/anchors.txt scripts/gauntlet.sh 30+0.3
```

Each anchor yields an estimate (`anchor_rating + measured_diff`); the headline is
their **inverse-variance-weighted mean** (tighter anchors count for more). Two
anchor modes:

- **`sf` (default)** — one throttled Stockfish per rung; `MIN_ELO`/`MAX_ELO`/`STEP`
  control the rating range. Cheap (one binary), but Stockfish's `UCI_Elo` is itself
  approximate (treat as ±100) and floored at 1320. Needs `stockfish` on `PATH`.
- **`file`** — a roster of real engines with published ratings, one per line as
  `name | rating | cmd | opts`, where `opts` is passed verbatim to fastchess (so it
  carries `option.X=Y` UCI options *and* per-engine limits like `nodes=1`). Two
  ready rosters are committed, kept separate so a single mean never mixes pools:
  - [`scripts/anchors-maia.txt`](../scripts/anchors-maia.txt) — **human (Lichess)
    Elo**: lc0 + [Maia](https://github.com/CSSLab/maia-chess) nets (1100–1900) run
    at `nodes=1`, each playing like a human of that rating. Needs `lc0` and the
    nets under `~/chess-engines/maia/`.
  - [`scripts/anchors-ccrl.txt`](../scripts/anchors-ccrl.txt) — **engine (CCRL)
    Elo**: standalone UCI engines with [CCRL Blitz](https://computerchess.org.uk/ccrl/404/)
    ratings. The defensible "where am I on the engine lists" number. On Apple
    Silicon the catch is that CCRL-rated weak/mid engines are x86-era (inline asm),
    so they're cross-compiled for x86_64 and run via **Rosetta 2** (the committed
    roster uses Cinnamon 2.2a=2071 and 2.3=2212).
  - [`scripts/anchors.txt`](../scripts/anchors.txt) is a template/example.

  One command provisions both pools (downloads Maia nets, cross-builds the Cinnamon
  anchors, builds Ordo) and regenerates the rosters with absolute paths for the
  current machine: [`scripts/setup-anchors.sh`](../scripts/setup-anchors.sh)
  (needs `brew install stockfish lc0` and Rosetta). Run the two pools separately —
  never merge human and engine Elo in one mean:

  ```
  scripts/setup-anchors.sh
  MODE=file ANCHORS=scripts/anchors-ccrl.txt scripts/gauntlet.sh   # engine (CCRL) Elo
  MODE=file ANCHORS=scripts/anchors-maia.txt scripts/gauntlet.sh   # human (Lichess) Elo
  ```

Pick anchors that **bracket** the engine (some below, some above): a rung the
candidate sweeps 100%/0% gives only a bound and is excluded from the mean (and
flagged, so you re-range). If [`ordo`](https://github.com/michiguel/Ordo) is on
`PATH` it runs a rigorous multi-anchor cross-check on the combined PGN. (Ordo has
no Homebrew formula; on macOS build it with the spinlock→mutex fallback, since
Darwin lacks POSIX spinlocks: `git clone https://github.com/michiguel/Ordo && cd
Ordo && make CC=clang CFLAGS="-DNDEBUG -DMY_SEMAPHORES -DNSPINLOCKS -flto -I myopt
-I sysport" && cp ordo /opt/homebrew/bin/`.)

Caveats: Elo is **pool-relative** — CCRL, CEGT, SSDF and human FIDE are different
pools and not directly comparable; time control and hardware are part of any
rating, so quote them. Maia in particular is the **Lichess human** pool, not CCRL.
Use this as an occasional sanity check, not a per-change gate — SPRT stays the
development loop.

## The working method (every change, Phase 1+)

1. Make one change.
2. `cargo test` + `cargo clippy` — correctness.
3. `scripts/sprt.sh <prev-tag>` — strength vs the previous tagged build.
4. Keep the change only if SPRT reports `PASS`; otherwise revert.
5. When a gainer lands, **tag a release** and rebuild the baseline from it, so
   there's always an opponent to measure the next change against — one command
   ([`scripts/release.sh`](../scripts/release.sh), issue #32):

```
scripts/release.sh 0.6.0 "null-move pruning: +40 Elo vs v0.5.0"
```

It refuses on a dirty tree or failing `cargo test`, then bumps `Cargo.toml`,
commits, tags `v0.6.0`, pushes (commit + tag), rebuilds `--release`, and copies
the fresh build to `baseline/engine` so the *next* change measures against the
engine that just won. `--dry-run` runs every gate and prints the plan without
mutating anything. Git tags stay the source of truth; `baseline/` is gitignored
and any baseline is rebuildable from its tag (that's what `scripts/sprt.sh` does).

## Speed: criterion micro-benchmarks

SPRT answers *is it stronger?*; benchmarks answer the orthogonal *is it faster?*
— catching nodes/sec regressions in the hot paths with numbers instead of
guesses (issue #30). The benches live in [`benches/engine.rs`](../benches/engine.rs)
and reuse the published perft FENs, so they time the exact positions the
correctness tests pin. They're **local-only** (not run in CI); criterion always
builds optimized, so there's no separate `--release` flag to remember.

```
cargo bench                                  # run every group, print results
cargo bench --bench engine sliders           # one group (substring filter)
```

The groups: `perft` (startpos d5, Kiwipete d4 — the end-to-end movegen +
make/unmake loop), `generate_legal`, `eval` (`Material::evaluate`),
`make_unmake` (the round-trip), and `sliders` (magic bitboards vs the
`ray_attacks` oracle they replaced — quantifies the issue #27 win).

### Catching regressions: save a baseline, compare against it

Criterion's value is the diff between two runs. Tag a known-good point, then
compare later work against it:

```
cargo bench -- --save-baseline v0.5.0        # stash current numbers as "v0.5.0"
# …make a change…
cargo bench -- --baseline v0.5.0             # re-run, report % change vs baseline
```

A run with no `--baseline` compares against the *last* run (stored under
`target/criterion/`), so plain `cargo bench` twice already shows a delta. Use a
named baseline when you want a stable reference (e.g. a release) to measure
several iterations against. Criterion flags each change `improved` / `regressed`
past a noise threshold, so looks-like-a-win noise doesn't fool you — the same
discipline as iron rule #3, applied to speed.

## See also

- [03-roadmap.md](03-roadmap.md) — the phased plan and the working method.
- [ADR 0004](decisions/0004-cutechess-testing.md) — why cutechess-cli.
- Chess Programming Wiki: *Match*, *SPRT* (linked from [resources.md](resources.md)).
