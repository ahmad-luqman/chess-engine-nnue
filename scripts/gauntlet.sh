#!/usr/bin/env bash
#
# gauntlet.sh — estimate the engine's *absolute* Elo against rated anchors.
#
# scripts/sprt.sh measures *relative* Elo (candidate vs the previous version).
# This measures *absolute* Elo: play the current build against opponents whose
# rating is already known, then read our rating off relative to theirs. Each
# anchor yields an estimate (anchor_rating + measured_diff); the headline number
# is their inverse-variance-weighted mean (tighter anchors count for more).
#
# Two anchor modes:
#
#   sf  (default) — one throttled Stockfish per rating rung. This is where you
#                   "control the rating range": MIN_ELO / MAX_ELO / STEP. Cheap,
#                   one binary, but Stockfish's UCI_Elo is itself approximate
#                   (treat results as +/-100ish) and its floor is 1320.
#
#   file          — a roster of real engines with published ratings (e.g. CCRL).
#                   ANCHORS=<file>, one per line: name | rating | cmd | opts...
#                   ('opts' is passed verbatim as fastchess -engine tokens, so it
#                    carries UCI options like option.UCI_Elo=1800 AND per-engine
#                    limits like nodes=1 / st=N; '#' and blank lines ignored).
#                    More defensible if anchors are rated on the scale you quote.
#
# Usage:
#   scripts/gauntlet.sh [tc]
#
# Env (with defaults):
#   MODE=sf                         # sf | file
#   SF=stockfish                    # stockfish binary (sf mode)
#   MIN_ELO=1320 MAX_ELO=2400 STEP=200   # rung range (sf mode); clamped to SF's 1320 floor
#   ANCHORS=scripts/anchors.txt     # roster file (file mode)
#   ROUNDS=50 CONCURRENCY=8         # per-anchor match size (-repeat -games 2 => 2*ROUNDS games)
#   TC=10+0.1                       # time control (also overridable as $1)
#   BOOK=books/openings.epd         # opening book (regenerated if missing)
#   CANDIDATE=target/release/engine # engine under test (auto-built if it's the default path)
#
# Examples:
#   scripts/gauntlet.sh                                  # SF rungs 1320..2400 @ 10+0.1
#   MIN_ELO=1400 MAX_ELO=2000 STEP=100 scripts/gauntlet.sh    # tighter, lower band
#   MODE=file ANCHORS=scripts/anchors.txt scripts/gauntlet.sh 30+0.3
#
# Output: a per-anchor table, the weighted absolute Elo +/- error, and (if `ordo`
# is installed) a rigorous cross-check from the combined PGN. Anchors the
# candidate sweeps 100%/0% give only a bound, not a point — they're flagged and
# excluded from the mean, nudging you to pick rungs that actually bracket it.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

TC="${1:-${TC:-10+0.1}}"
MODE="${MODE:-sf}"
SF="${SF:-stockfish}"
MIN_ELO="${MIN_ELO:-1320}"
MAX_ELO="${MAX_ELO:-2400}"
STEP="${STEP:-200}"
ANCHORS="${ANCHORS:-scripts/anchors.txt}"
ROUNDS="${ROUNDS:-50}"
CONCURRENCY="${CONCURRENCY:-8}"
BOOK="${BOOK:-books/openings.epd}"
CANDIDATE="${CANDIDATE:-target/release/engine}"
FASTCHESS="${FASTCHESS:-/opt/homebrew/bin/fastchess}"

command -v "$FASTCHESS" >/dev/null 2>&1 || { echo "error: fastchess not found at $FASTCHESS" >&2; exit 1; }

# Trim leading/trailing whitespace (pure bash — safe with apostrophes, quotes, '=').
trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; s="${s%"${s##*[![:space:]]}"}"; printf '%s' "$s"; }

# ── Candidate build ───────────────────────────────────────────────────────────
if [[ "$CANDIDATE" == "target/release/engine" ]]; then
    echo ">> building candidate (cargo build --release)…" >&2
    cargo build --release
fi
[[ -x "$CANDIDATE" ]] || { echo "error: candidate binary '$CANDIDATE' not found/executable" >&2; exit 1; }

# ── Assemble the anchor roster: parallel arrays NAMES / RATINGS / CMDS / OPTS ──
NAMES=(); RATINGS=(); CMDS=(); OPTS=()

if [[ "$MODE" == "sf" ]]; then
    command -v "$SF" >/dev/null 2>&1 || {
        echo "error: stockfish ('$SF') not on PATH. Install it (brew install stockfish) or use MODE=file." >&2
        exit 1; }
    if (( MIN_ELO < 1320 )); then
        echo ">> note: Stockfish UCI_Elo floor is 1320; clamping MIN_ELO $MIN_ELO -> 1320" >&2
        MIN_ELO=1320
    fi
    for (( elo=MIN_ELO; elo<=MAX_ELO; elo+=STEP )); do
        NAMES+=("sf$elo"); RATINGS+=("$elo"); CMDS+=("$SF")
        OPTS+=("option.UCI_LimitStrength=true option.UCI_Elo=$elo")
    done
elif [[ "$MODE" == "file" ]]; then
    [[ -f "$ANCHORS" ]] || { echo "error: roster file '$ANCHORS' not found (MODE=file)" >&2; exit 1; }
    # The 4th field is passed VERBATIM as fastchess -engine tokens, so it can carry
    # both UCI options (option.UCI_Elo=1800) and per-engine limits (nodes=1, st=N).
    while IFS='|' read -r name rating cmd opts; do
        # Skip blanks/comments BEFORE any processing (comment text may hold quotes).
        probe="$(trim "$name")"
        [[ -z "$probe" || "$probe" == \#* ]] && continue
        name="$(trim "$name")"; rating="$(trim "$rating")"
        cmd="$(trim "$cmd")";   opts="$(trim "${opts:-}")"
        NAMES+=("$name"); RATINGS+=("$rating"); CMDS+=("$cmd"); OPTS+=("$opts")
    done < "$ANCHORS"
else
    echo "error: MODE must be 'sf' or 'file' (got '$MODE')" >&2; exit 1
fi

(( ${#NAMES[@]} > 0 )) || { echo "error: no anchors assembled" >&2; exit 1; }

# ── Ensure the opening book ───────────────────────────────────────────────────
if [[ ! -s "$BOOK" ]]; then
    echo ">> book $BOOK missing — generating via examples/genbook…" >&2
    mkdir -p "$(dirname "$BOOK")"
    cargo run --release --example genbook > "$BOOK"
fi

# ── Run the gauntlet: one match per anchor ────────────────────────────────────
PGN="gauntlet.pgn"; LOG="gauntlet.log"; : > "$PGN"; : > "$LOG"
EST="$(mktemp)"; trap 'rm -f "$EST"' EXIT      # rows: "rating diff err"  (numeric only)

echo ">> gauntlet: candidate vs ${#NAMES[@]} anchors  tc=$TC  rounds=$ROUNDS (x2 games)" >&2
printf '\n%-12s %8s %8s %16s %14s\n' "anchor" "rating" "score%" "Elo diff (+/-)" "your Elo"
printf -- '------------------------------------------------------------------------\n'

for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"; rating="${RATINGS[$i]}"; cmd="${CMDS[$i]}"; opts="${OPTS[$i]}"
    mpgn="$(mktemp).pgn"
    # shellcheck disable=SC2086  # $opts is intentionally word-split into flags
    out="$("$FASTCHESS" \
        -engine cmd="$CANDIDATE" name=mine \
        -engine cmd="$cmd" name="$name" $opts \
        -each proto=uci tc="$TC" \
        -openings file="$BOOK" format=epd order=random \
        -repeat -rounds "$ROUNDS" -games 2 -concurrency "$CONCURRENCY" \
        -pgnout file="$mpgn" 2>&1)"
    echo "$out" >> "$LOG"
    cat "$mpgn" >> "$PGN" 2>/dev/null || true; rm -f "$mpgn"

    # fastchess prints a periodic interim Elo/Games block as the match runs, then a
    # final cumulative one. Take the LAST of each (not the first): an early report is
    # a small sample, and if the candidate sweeps the opening games it reads
    # `Elo: inf` and would be misflagged "no signal" while real signal arrives later.
    elo_line="$(grep '^Elo:' <<<"$out" | tail -1 || true)"
    pts_line="$(grep '^Games:' <<<"$out" | tail -1 || true)"
    diff="$(sed -nE 's/^Elo: *([^ ]+) .*/\1/p' <<<"$elo_line")"
    err="$(sed -nE 's/^Elo: [^ ]+ \+\/- ([^,]+).*/\1/p'  <<<"$elo_line")"
    score="$(sed -nE 's/.*\(([0-9.]+) %\).*/\1/p' <<<"$pts_line")"
    score="${score:-?}"

    # Keep only finite estimates; a 100%/0% sweep yields inf/nan -> bound, not point.
    if awk -v d="$diff" -v e="$err" 'BEGIN{exit !(d+0==d && e+0==e && e>0)}' 2>/dev/null; then
        your="$(awk -v r="$rating" -v d="$diff" 'BEGIN{printf "%.0f", r+d}')"
        printf '%-12s %8s %8s %16s %14s\n' "$name" "$rating" "$score" "$(printf '%+.0f +/- %.0f' "$diff" "$err")" "$your"
        echo "$rating $diff $err" >> "$EST"
    else
        bound="$(awk -v s="$score" 'BEGIN{ if (s+0>=99.5) print "out-of-range (won ~all)"; else if (s+0<=0.5) print "out-of-range (lost ~all)"; else print "no signal"}')"
        printf '%-12s %8s %8s %16s %14s\n' "$name" "$rating" "$score" "—" "$bound"
    fi
done

# ── Headline: inverse-variance-weighted absolute Elo ──────────────────────────
printf -- '------------------------------------------------------------------------\n'
if [[ -s "$EST" ]]; then
    awk '{ w=1/($3*$3); sw+=w; swx+=w*($1+$2) }
         END{ printf ">> ABSOLUTE ELO ESTIMATE: %.0f +/- %.0f  (inverse-variance weighted over %d anchor%s)\n", swx/sw, sqrt(1/sw), NR, (NR==1?"":"s") }' "$EST"
else
    echo ">> no anchor produced a finite estimate — every match was a sweep."
    echo "   Adjust the range to bracket the engine, e.g. MIN_ELO/MAX_ELO/STEP (sf mode)."
fi

# ── Optional rigorous cross-check via Ordo ────────────────────────────────────
if command -v ordo >/dev/null 2>&1 && [[ -s "$PGN" ]]; then
    # Pin the median anchor to fix the scale; Ordo solves the rest from the PGN.
    mid=$(( ${#NAMES[@]} / 2 ))
    echo ">> ordo cross-check (pinning ${NAMES[$mid]}=${RATINGS[$mid]})…"
    if ordo -p "$PGN" -a "${RATINGS[$mid]}" -A "${NAMES[$mid]}" -o gauntlet-ratings.txt >/dev/null 2>&1; then
        grep -E "mine|${NAMES[$mid]}" gauntlet-ratings.txt 2>/dev/null || cat gauntlet-ratings.txt
        echo "   full table: gauntlet-ratings.txt"
    else
        echo "   ordo run failed; combined PGN saved at $PGN for manual ordo/bayeselo."
    fi
else
    echo ">> (install 'ordo' for a rigorous multi-anchor cross-check; combined PGN: $PGN)"
fi
