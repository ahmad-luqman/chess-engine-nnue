#!/usr/bin/env bash
#
# release.sh — promote a passing build to a tagged baseline (issue #32).
#
# The per-gainer release discipline (bump Cargo.toml, tag, push, rebuild the
# baseline opponent) was done by hand for v0.2.0…v0.5.0. This scripts it: once an
# SPRT passes (run scripts/sprt.sh first), one command bumps the version, tags,
# pushes, and refreshes baseline/engine so the *next* change is measured against
# the engine that just won.
#
# Usage:
#   scripts/release.sh [--dry-run] <version> <message>
#
#   <version>   semver without the leading 'v' (e.g. 0.6.0); the tag is v<version>.
#   <message>   annotated-tag / commit message (quote it).
#   --dry-run   run every gate (clean tree, no tag collision, cargo test) and
#               print what would happen, but make no commit/tag/push/baseline change.
#
# Example:
#   scripts/release.sh 0.6.0 "null-move pruning: +40 Elo vs v0.5.0"
#
# Invariants (iron rules #2/#3): refuses on a dirty tree or failing tests; git
# tags stay the source of truth (baseline/ is gitignored and rebuildable from any
# tag via scripts/sprt.sh). Run an SPRT and confirm it PASSes before promoting.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

if [[ $# -ne 2 ]]; then
    echo "usage: scripts/release.sh [--dry-run] <version> <message>" >&2
    exit 2
fi

VERSION="$1"
MESSAGE="$2"
TAG="v$VERSION"

# Accept a plain semver (optionally a pre-release/build suffix); reject a stray 'v'.
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-+.][0-9A-Za-z.-]+)?$ ]]; then
    echo "error: version '$VERSION' is not semver (expected e.g. 0.6.0, no leading 'v')" >&2
    exit 2
fi

run() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "   [dry-run] would run: $*"
    else
        "$@"
    fi
}

# ── Gate 1: clean working tree ────────────────────────────────────────────────
if [[ -n "$(git status --porcelain)" ]]; then
    echo "error: working tree is dirty — commit or stash first." >&2
    git status --short >&2
    exit 1
fi

# ── Gate 2: tag must not already exist ────────────────────────────────────────
if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null 2>&1; then
    echo "error: tag $TAG already exists." >&2
    exit 1
fi

# ── Gate 3: tests are green (release matters, but tests are the correctness gate) ─
echo ">> cargo test…" >&2
cargo test

echo ">> gates passed; releasing $TAG" >&2

# ── Bump the package version (first 'version =' line, under [package]) ─────────
if [[ "$DRY_RUN" -eq 1 ]]; then
    CURRENT="$(sed -n -E '1,/^version = /s/^version = "(.*)"/\1/p' Cargo.toml)"
    echo "   [dry-run] would bump Cargo.toml version $CURRENT -> $VERSION"
else
    sed -i '' -E "1,/^version = /s/^version = \".*\"/version = \"$VERSION\"/" Cargo.toml
    # Refresh Cargo.lock's own version entry so the commit is self-consistent.
    cargo update -p chess-engine-nnue --precise "$VERSION" >/dev/null 2>&1 || cargo build --release >/dev/null
fi

# ── Commit, tag, push ─────────────────────────────────────────────────────────
run git add Cargo.toml Cargo.lock
run git commit -m "release: $TAG" -m "$MESSAGE"
run git tag -a "$TAG" -m "$MESSAGE"
run git push
run git push origin "$TAG"

# ── Rebuild the baseline opponent from the freshly released build ──────────────
echo ">> cargo build --release && refresh baseline/engine" >&2
run cargo build --release
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "   [dry-run] would: mkdir -p baseline && cp target/release/engine baseline/engine"
else
    mkdir -p baseline
    cp target/release/engine baseline/engine
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo ">> dry-run complete — no changes made."
else
    echo ">> released $TAG; baseline/engine now is the $TAG build. Next change SPRTs against it."
fi
