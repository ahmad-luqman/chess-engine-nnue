#!/usr/bin/env bash
#
# sprt.sh — run an SPRT of the current build vs a baseline (issue #33).
#
# Every strength change must be SPRT-tested (iron rule #3). This codifies the
# by-hand Phase 2 recipe into one command: build the candidate, obtain the
# baseline opponent (from a git tag, built in a throwaway worktree, or a path to
# an existing binary), ensure the opening book exists, run fastchess, and print
# the LLR / Elo / accept-reject verdict.
#
# Usage:
#   scripts/sprt.sh <baseline> [tc]
#
#   <baseline>   a git tag (e.g. v0.5.0) built fresh in a temp worktree, OR a
#                path to an existing engine binary to use as-is.
#   [tc]         time control, fastchess syntax (default: 5+0.05 = 5s + 0.05s/move).
#
# SPRT bounds and match shape are overridable via environment:
#   ELO0=0 ELO1=5 ALPHA=0.05 BETA=0.05   # the hypothesis bounds
#   ROUNDS=5000 GAMES=2 CONCURRENCY=8     # match size (GAMES=2 + -repeat = colour-reversed pairs)
#   BOOK=books/openings.epd               # opening book (regenerated if missing)
#
# Examples:
#   scripts/sprt.sh v0.5.0                 # candidate vs v0.5.0 at 5+0.05
#   scripts/sprt.sh v0.5.0 8+0.08          # ... at a slower TC
#   ELO1=3 CONCURRENCY=4 scripts/sprt.sh baseline/engine
#
# fastchess lives at /opt/homebrew/bin/fastchess on the dev box (see
# docs/07-testing.md and ADR 0004). Strength is independent of -concurrency.
set -euo pipefail

# Run from the repo root regardless of where the script is invoked from.
cd "$(git rev-parse --show-toplevel)"

if [[ $# -lt 1 ]]; then
    echo "usage: scripts/sprt.sh <baseline-tag-or-path> [tc]" >&2
    exit 2
fi

BASELINE="$1"
TC="${2:-5+0.05}"

# Tunables (env-overridable). Defaults match the issue #33 / docs/07 recipe.
ELO0="${ELO0:-0}"
ELO1="${ELO1:-5}"
ALPHA="${ALPHA:-0.05}"
BETA="${BETA:-0.05}"
ROUNDS="${ROUNDS:-5000}"
GAMES="${GAMES:-2}"
CONCURRENCY="${CONCURRENCY:-8}"
BOOK="${BOOK:-books/openings.epd}"

FASTCHESS="${FASTCHESS:-/opt/homebrew/bin/fastchess}"
CANDIDATE="target/release/engine"

command -v "$FASTCHESS" >/dev/null 2>&1 || { echo "error: fastchess not found at $FASTCHESS" >&2; exit 1; }

# ── 1. Build the candidate (the current working tree) ─────────────────────────
echo ">> building candidate (cargo build --release)…" >&2
cargo build --release

# ── 2. Resolve the baseline opponent ──────────────────────────────────────────
# A path to an existing binary is used directly; anything else is treated as a
# git tag/ref and built fresh in a temporary worktree so the result is exactly
# reproducible from version control (baseline/ stays gitignored — see #32).
WORKTREE=""
cleanup() { [[ -n "$WORKTREE" && -d "$WORKTREE" ]] && git worktree remove --force "$WORKTREE" 2>/dev/null || true; }
trap cleanup EXIT

if [[ -x "$BASELINE" && -f "$BASELINE" ]]; then
    BASE_BIN="$BASELINE"
    echo ">> baseline: using existing binary $BASE_BIN" >&2
else
    git rev-parse -q --verify "refs/tags/$BASELINE^{commit}" >/dev/null 2>&1 \
        || git rev-parse -q --verify "$BASELINE^{commit}" >/dev/null 2>&1 \
        || { echo "error: '$BASELINE' is neither an executable binary nor a known git ref" >&2; exit 1; }

    WORKTREE="$(mktemp -d)/baseline-$BASELINE"
    echo ">> baseline: building $BASELINE in a temp worktree…" >&2
    git worktree add --detach "$WORKTREE" "$BASELINE" >&2
    ( cd "$WORKTREE" && cargo build --release ) >&2
    mkdir -p baseline
    cp "$WORKTREE/target/release/engine" baseline/engine
    BASE_BIN="baseline/engine"
    echo ">> baseline: stashed $BASELINE build at $BASE_BIN" >&2
fi

# ── 3. Ensure the opening book exists ─────────────────────────────────────────
if [[ ! -s "$BOOK" ]]; then
    echo ">> book $BOOK missing — generating via examples/genbook…" >&2
    mkdir -p "$(dirname "$BOOK")"
    cargo run --release --example genbook > "$BOOK"
fi

# ── 4. Run the SPRT ───────────────────────────────────────────────────────────
LOG="sprt.log"
echo ">> SPRT  candidate vs $BASELINE  tc=$TC  elo0=$ELO0 elo1=$ELO1 alpha=$ALPHA beta=$BETA" >&2
echo ">> logging to $LOG (PGN -> sprt.pgn); LLR decides at (-2.94, 2.94) for alpha=beta=0.05" >&2

"$FASTCHESS" \
    -engine cmd="$CANDIDATE" name=new \
    -engine cmd="$BASE_BIN"  name=base \
    -each proto=uci tc="$TC" \
    -openings file="$BOOK" format=epd order=random \
    -repeat -rounds "$ROUNDS" -games "$GAMES" -concurrency "$CONCURRENCY" \
    -sprt elo0="$ELO0" elo1="$ELO1" alpha="$ALPHA" beta="$BETA" \
    -pgnout file=sprt.pgn \
    2>&1 | tee "$LOG"

# ── 5. Report the verdict ─────────────────────────────────────────────────────
# Decide from the final LLR vs the bounds fastchess prints on the LLR line:
#   LLR: <value> (<pct>%) (<low>, <high>) [<elo0>, <elo1>]
# LLR >= high  -> H1 accepted (candidate is >= elo1 stronger)  -> PASS
# LLR <= low   -> H0 accepted (no meaningful gain)             -> FAIL
# otherwise    -> rounds exhausted before a bound was crossed  -> INCONCLUSIVE
echo "--------------------------------------------------"
ELO_LINE="$(grep -E '^Elo:' "$LOG" | tail -1 || true)"
LLR_LINE="$(grep -E '^LLR:' "$LOG" | tail -1 || true)"
echo "${ELO_LINE:-Elo: (not reported)}"
echo "${LLR_LINE:-LLR: (not reported)}"

if [[ "$LLR_LINE" =~ LLR:\ (-?[0-9.]+)\ .*\((-?[0-9.]+),\ (-?[0-9.]+)\)\ \[ ]]; then
    LLR="${BASH_REMATCH[1]}"; LOW="${BASH_REMATCH[2]}"; HIGH="${BASH_REMATCH[3]}"
    VERDICT="$(awk -v l="$LLR" -v lo="$LOW" -v hi="$HIGH" 'BEGIN{
        if (l+0 >= hi+0) print "PASS";
        else if (l+0 <= lo+0) print "FAIL";
        else print "INCONCLUSIVE"; }')"
    case "$VERDICT" in
        PASS) echo ">> VERDICT: PASS — H1 accepted; candidate is stronger (>= elo1=$ELO1). Promote it." ;;
        FAIL) echo ">> VERDICT: FAIL — H0 accepted; no meaningful gain. Revert the change." ;;
        *)    echo ">> VERDICT: INCONCLUSIVE — LLR=$LLR within (-$HIGH..$HIGH); ran out of rounds. Increase ROUNDS." ;;
    esac
    [[ "$VERDICT" == "FAIL" ]] && exit 1 || true
else
    echo ">> VERDICT: could not parse an LLR line from $LOG — inspect it manually." >&2
    exit 1
fi
