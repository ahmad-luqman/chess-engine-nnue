# 08 — Search internals (deep dive)

Phase 2 turns the Phase 1 toy searcher into a *real* engine: a position identity
it can cache and compare (Zobrist), a transposition table, principled move
ordering, quiescence at the leaves, and draw detection. This is the running
deep-dive for that work — one section lands per issue (#23–#28), now continuing
into Phase 3's selective search (§6 PVS #34, §7 LMR #37). It assumes the
board and move machinery from [05-board-representation.md](05-board-representation.md)
and [06-move-generation.md](06-move-generation.md), and the Phase 1 negamax in
`src/search.rs`.

---

## 1. Zobrist hashing — a position's fingerprint (`src/zobrist.rs`, issue #23)

A transposition table and repetition detection both need to answer "have I seen
*this exact position* before?" cheaply. We need a 64-bit key that is (a) nearly
unique per position and (b) maintainable *incrementally* — recomputing it from
scratch each node would cost more than it saves.

### The idea

Assign a random 64-bit constant to every board **feature**:

- `[color][piece_type][square]` — 2×6×64 piece-placement keys,
- one **side-to-move** key (XORed in when it's Black's turn),
- **castling rights** — a 16-entry table indexed by the 4-bit rights bitset,
- **en-passant file** — 8 keys (see the capturable rule below).

A position's key is the **XOR of the constants for the features it has**. The
trick is that XOR is its own inverse (`x ^ k ^ k == x`), so changing a feature is
just XOR-ing its constant: out when it leaves, in when it arrives. A move touches
a handful of features, so the update is O(features changed), not O(64).

```
move e2–e4:  hash ^= PIECE[White][Pawn][e2]   // pawn leaves e2
             hash ^= PIECE[White][Pawn][e4]   // pawn arrives on e4
             hash ^= SIDE                      // turn flips
             (+ castling delta, ep delta as applicable)
```

### Where it lives

- **Constants** are generated at *compile time* by a `const fn` splitmix64 stream
  from a fixed seed (`zobrist::KEYS`). Compile-time means zero startup cost and
  the table sits in read-only data; fixed seed means hashes are reproducible
  across runs (matters for future opening books / tuning caches).
- **`Board::hash`** holds the live key. It is seeded from scratch by the FEN
  parser (`zobrist::compute`), updated incrementally in `Board::make_move`, and
  restored in `Board::unmake_move`.
- **Unmake is O(1) by snapshot**: `make_move` stashes the pre-move key in `Undo`
  and `unmake_move` restores it. We *could* replay the deltas in reverse (XOR is
  invertible), but `Undo` already exists to carry un-recomputable state, so a
  snapshot is simpler and just as cheap.

### The en-passant subtlety (the part that bites)

The ep file is hashed **only when the side to move can actually capture en
passant** — i.e. one of their pawns attacks the ep square — *not* merely when an
ep square exists. This is a correctness requirement, not an optimization:

```
1.e4  e5  2.Nf3   → Nf3 clears the ep square        → ep = None
1.Nf3 e5  2.e4    → e4 sets ep = e3, but no black pawn can take it
```

Both lines reach the **same position**, so they must share a key. They do only if
the dangling, non-capturable e3 ep square contributes nothing. `compute` and
`make_move` both route ep hashing through `Board::ep_zobrist`, so they agree by
construction.

### How we know it's correct

Three layers, in increasing strength:

1. **Round-trip** — `make` then `unmake` restores the key exactly. Free: `hash`
   is a `Board` field and `Board: Eq`, so the existing make/unmake equality test
   already covers it.
2. **Incremental ≡ from-scratch** — a `debug_assert_eq!(self.hash,
   compute(self))` in `make_move` fires at every node. Under `cargo test` the
   perft walk drives millions of nodes through it; in `--release` it compiles
   away.
3. **Transposition** — `1.e4 e5 2.Nf3` and `1.Nf3 e5 2.e4` produce equal hashes.

> ⚠️ **Trap worth remembering:** layers 1 and 2 pass *even with the naive ep
> rule*, because both code paths agree with each other. Only layer 3 catches it.
> A green incremental≡from-scratch is necessary but not sufficient.

### What it buys (later)

Nothing on its own — hashing is invisible to move generation, so perft counts are
unchanged and there's no strength change (hence no SPRT for #23). The payoff
arrives when #24 (TT) and #28 (repetition/fifty-move) build on `Board::hash`.

---

## 2. Transposition table — caching searched positions (`src/tt.rs`, issue #24)

Now that a position has a cheap identity (its Zobrist key), we can cache the
result of searching it. The same position recurs constantly — via different move
orders within one search, and across iterative-deepening iterations — and the TT
turns each recurrence from "search it again" into "look it up".

### The entry and the table

The table is a flat `Vec<Entry>` sized to a power of two; the index is the low
bits of the key (`hash & (len-1)`). Each entry records:

- `key` — the **full** 64-bit hash, so a slot shared by many positions
  (collision) is detected and ignored rather than trusted,
- `best_move` — the strongest move-ordering signal (consumed in #25),
- `score` + `bound` — the value and *what kind* of value it is,
- `depth` — how deep that value was searched,
- `age` — which search generation wrote it, for replacement.

### Bounds: a score is rarely the whole story

Alpha-beta doesn't always compute an exact score. When a node fails high (a move
beats `beta`) we stop early, so all we know is the score is *at least* that — a
**lower bound**. When every move fails low (none beats `alpha`) we know the score
is *at most* `alpha` — an **upper bound**. Only a node that finishes with a move
strictly inside the window yields an **exact** score. The TT stores which case it
was, because that determines how a probe may use it:

```
Exact → return the stored score directly
Lower → usable only if it proves a fail-high (score ≥ beta) → cut
Upper → usable only if it proves a fail-low  (score ≤ alpha) → cut
```

and only when the stored search was **at least as deep** as what we need now
(`entry.depth >= depth`). We keep the search **fail-hard** (a probe cut returns
`beta`/`alpha`, exactly what searching the node would have), so switching the TT
off is byte-for-byte identical to the Phase 1 search.

### Mate scores need re-anchoring

A mate score means "mate in N plies *from this node*". The same position can sit
at different distances from the root in different lines, so we store mate scores
relative to the node — add `ply` on the way in, subtract it on the way out
(`score_to_tt`/`score_from_tt`). Without this a cached mate would claim the wrong
distance and the engine would misorder or misreport forced mates.

### Replacement and lifetime

One slot per index, so a store sometimes evicts. We keep the more useful entry:
an empty slot or one from an older search is always overwritten; otherwise the
**deeper** entry wins. The table is owned by the UCI layer and *borrowed* by the
search, so it survives across iterative-deepening iterations and across moves in
a game. `ucinewgame` clears it; `setoption name Hash value <MB>` resizes it
(default 16 MB).

### The honest caveat: fixed-depth scores aren't bit-invariant

Depth-preferred probing means a position stored at depth 5 and re-probed at a
depth-2 node returns the depth-5 score — a "depth-leak". It only makes a leaf
*more* accurate, and every engine does it, but it means a fixed-depth search
with the TT on can report a slightly different *score* than with it off. So the
correctness tests assert the **best move** is unchanged (plus the existing
exact-score tactical/mate tests still pass), not bit-identical scores. See
ADR 0006.

### What it buys

Within one search, transposition cutoffs prune re-searched subtrees; across
iterations, the previous depth's results are cached. The *big* win, though, is
searching the stored best move first — that's move ordering (#25), where this
work pays off. On its own, and especially while the engine still throws away won
endgames for lack of quiescence (#26) and draw detection (#28), the TT's
measurable Elo is smaller and noisier — expected from splitting #24 and #25.

---

## 3. Move ordering — searching the best move first (`src/search.rs`, issue #25)

Alpha-beta's entire payoff is conditional on **move order**. A beta cutoff fires
the moment one move proves good enough to refute the line; if that move is first,
the node costs one child instead of all of them. With perfect ordering the tree
shrinks from `b^d` toward `b^(d/2)` — the difference between depth 4 and depth 8
for the same work. Phase 1 searched moves in generation order (arbitrary); this
issue scores them.

### The ordering, by band

Each move gets a score and we search highest first. The bands, top to bottom:

1. **TT move** — the best move stored for *this position* (ADR 0006). It's the
   strongest single signal: in iterative deepening it's the move that was best one
   ply shallower, almost always still best. Searching it first at the root is the
   biggest practical speedup in the whole engine.
2. **Captures & promotions, by MVV-LVA** — *Most Valuable Victim − Least Valuable
   Attacker*. `PxQ` before `QxQ`: grabbing the queen with a pawn is both more
   profitable and risks less, so it's the likelier refutation. Scored
   `victim·16 − attacker` (+ the promoted-piece value for promotions).
3. **Killer moves** — two *quiet* moves per ply that recently caused a beta
   cutoff. A move that refuted one line at this depth often refutes its siblings
   (a sibling threat, a recapture square), even though it captures nothing.
4. **History heuristic** — a `[from][side][to]` table incremented by `depth²` on
   every quiet cutoff. It's the global, position-independent tiebreak among the
   remaining quiets: moves that have been good *somewhere* are tried earlier
   *everywhere*.

The bands are spaced far apart (TT ≫ captures ≫ killers ≫ history) and the
history score is clamped below the killer band, so no in-band value can ever
outrank a higher band.

### Why it doesn't change the result

Ordering only changes *what order* moves are tried, never *which* score the node
returns — alpha-beta gives the same value for any ordering. So the correctness
tests assert the chosen move and the tactical/mate scores are unchanged; only the
node count drops. Killers and history are reset each search; history accumulates
across a search's iterative-deepening iterations.

### What it buys

Everything the TT (#24) set up now pays off. On Kiwipete to a fixed depth 7 the
ordered search finishes in ~1 second (~1.1M nodes); v0.1.0, searching in
generation order, doesn't finish depth 7 in two minutes. In a game that extra
reachable depth is the strength gain — and it's what finally makes the Phase 2
SPRT against v0.1.0 decisive.

---

## 4. Quiescence search — quiet leaves (`src/search.rs`, issue #26)

A fixed-depth search stops at depth 0 no matter what's happening on the board. If
that leaf falls in the *middle of a capture sequence* — say White has just taken
a pawn but Black's recapture is one ply past the horizon — the static eval scores
the half-finished trade as if it were over. This is the **horizon effect**, and
v0.1.0 shows it textbook-clearly: the start-position score swings ~100cp between
even and odd depths, and even dips negative, purely from where the leaf lands in
a pawn trade.

### The fix: don't evaluate a noisy position

At depth 0 we call `qsearch` instead of evaluating directly. `qsearch` keeps
playing out **captures and promotions** until the position is quiet, then
evaluates. Its backbone is the **stand-pat** score:

```
stand_pat = evaluate(board)
if stand_pat >= beta { return beta }   // already good enough; don't bother capturing
if stand_pat > alpha { alpha = stand_pat }
for each capture/promotion (MVV-LVA order):
    score = -qsearch(-beta, -alpha)
    ... usual alpha-beta ...
```

The insight is that the side to move is **not forced to capture** — it can "stand
pat" on the current position — so the static eval is a *lower bound* on the node's
value. We only explore captures that might beat it. That both makes the search
sound (we never force a side into a bad capture) and keeps it cheap (most nodes
fail high on stand-pat immediately).

### Why it terminates

Unlike the main search, `qsearch` has no depth counter — yet it always halts,
because captures strictly remove material from a finite board, so any chain of
captures is bounded. A `ply` cap guards against pathological promotion lines
regardless.

### Scope and simplifications

We generate captures and promotions by filtering the legal moves (a dedicated
capture generator is a later speedup) and order them by MVV-LVA — the same victim
/attacker scoring as #25, no TT move or killers here. Two deliberate
simplifications for this first cut: we don't generate checks, and we stand-pat
even when in check (so a leaf that is actually checkmate may be scored by eval).
Both are standard early-stage compromises; check evasions in quiescence can come
later if they prove worth it.

### What it buys

The start-position score stabilises across depths (the even/odd swing collapses)
and tactical leaves resolve their exchanges, so the engine stops both
overvaluing won-but-about-to-be-recaptured material and undervaluing sound
sacrifices. Quiescence is historically one of the single biggest strength jumps
in a fixed-depth engine.

---

## 5. Draw detection — repetition & fifty-move (`src/search.rs`, issue #28)

Without draw rules the engine has two blind spots: it shuffles forever in
dead-drawn endgames (it never sees that repeating is a draw), and it can't tell
that a line is *not* winning. Phase 2 closes both by scoring **threefold
repetition** and the **fifty-move rule** as draws (0).

### Repetition

A position has repeated if its Zobrist key matches one seen earlier in the line.
We keep a stack of ancestor keys — seeded from the game history the GUI sends via
`position … moves`, then pushed/popped along the search path — and at each node
check whether the current key appears in it. Two refinements:

- **Only scan back `halfmove_clock` plies.** A pawn move or capture is
  irreversible and resets the clock, so no position before it can recur; scanning
  further is wasted and wrong.
- **A single in-tree repeat counts as a draw.** Strictly the game rule is
  *threefold*, but inside the search one repetition is enough to recognize the
  line is going nowhere and stop pursuing it.

The repetition check runs **before the TT probe**: a repetition is a property of
the *path*, not the position, so a TT score stored on a non-repeating path must
not be allowed to mask it.

### Fifty-move rule

If `halfmove_clock` reaches 100 plies (50 full moves with no pawn move or
capture), the position is a draw — but **checkmate takes precedence**: a mate
delivered on the 50th move is a mate, not a draw. So the fifty-move check comes
*after* generating moves and finding the node is not terminal, and the TT cut is
suppressed at the boundary (the clock isn't part of the key, so a cached score
could otherwise hide the draw).

### Where the history lives

The issue sketched a key stack on `Board`, but `Board` is a deliberately *pure
value* (it derives `Clone`/`Eq` and is cloned per search). Hanging a growing,
per-game `Vec` off it would muddy that and make every clone copy the history. So
the stack lives on the **search context** instead, seeded from the game history —
see [ADR 0007](decisions/0007-repetition-history.md). `Board` stays clean; make
/unmake are untouched.

### What it buys

Self-play stops shuffling drawn endgames to the move cap — games end as proper
draws — and the engine no longer keeps "winning" a position it's actually just
repeating. It's primarily a correctness fix (SPRT target: non-regression), though
recognizing repetitions also helps it convert won endgames instead of stumbling
into a draw.

---

## 6. Principal Variation Search — scouting the rest (`src/search.rs`, issue #34)

Phase 3 opens the *selective-search* layer, and PVS is its foundation: the
later techniques (LMR #37, null-move pruning #36) all **reduce a search and
re-search on fail-high**, which is the shape PVS introduces here.

### The bet

Move ordering (§3) plus the TT move make the first move almost always the best.
So for every *other* move we don't ask "what's its exact score?" — only "is it
worse than the move we already have?". A **null window** `(alpha, alpha + 1)`
answers that and nothing more, and because its bound is tight it prunes far more
of the subtree than the full `(alpha, beta)` window would. We pay the price of a
full-width re-search only on the rare scout that fails high (a genuine new best).

### The loop

The first move is searched full-window `(alpha, beta)` — it's the PV candidate,
and its exact score becomes the `alpha` the scouts measure against. Each later
move is scouted with `(alpha, alpha + 1)`; if the scout returns `s` with
`alpha < s < beta`, the move might be a new PV, so it's re-searched at full width.

```text
i == 0:  score = -negamax(-beta,      -alpha)            // full window, the PV
i  > 0:  s     = -negamax(-(alpha+1), -alpha)            // null-window scout
         if alpha < s < beta:                            // scout failed high
             score = -negamax(-beta, -alpha)             // re-search wide
```

The `s < beta` guard is the subtle part. When our *own* window is already null
(we are ourselves a scout one level up), `beta == alpha + 1`, so the scout window
*is* the full window and no re-search is owed — the guard suppresses it. The same
PVS lives at the root (`run_root`), which takes no beta cutoff (its beta is
`+INF`), so there the re-search condition collapses to just `s > alpha`.

### Why it doesn't change the result

`negamax` is **fail-hard**, so a null-window scout returns either `alpha`
(fail-low) or `alpha + 1` (fail-high) — never a score `>= beta` on its own, so
cutoffs still come only from the first move or a re-search, exactly as in plain
alpha-beta. A fail-low move would have failed low under the full window too; a
fail-high move gets re-searched to the same exact score plain alpha-beta would
have found. So the chosen move and score are unchanged — PVS is a pure speed
optimization. See [ADR 0009](decisions/0009-principal-variation-search.md) for why
we kept fail-hard rather than converting to fail-soft first.

### How we know it's correct

Result-invariance is asserted against golden scores captured from the *pre-PVS*
engine **with the TT disabled** — there PVS-vs-plain-alpha-beta is byte-identical.
The TT is disabled on purpose: scouts store bounds from null windows that a
full-window search never would, so a *TT-enabled* fixed-depth score can legitimately
differ (the same instability the TT probe documents). A `SearchContext` counter
tracks the scout/re-search ratio; a test asserts it stays low (measured ~0.00% on
a normal middlegame — tens of re-searches per ~10⁶ scouts).

### What it buys

A lower effective branching factor: the tight scout windows prune deeper, so the
engine reaches more depth in the same time once ordering is good. At shallow fixed
depth the raw node count is mixed (the scout/TT interaction, and on quiet
weakly-ordered positions extra re-searches) — the real gate is the SPRT vs v0.5.0.
Its larger payoff is structural: LMR and NMP now have the reduce-then-re-search
machinery they need.

---

## 7. Late move reductions — searching late moves shallower (`src/search.rs`, issue #37)

PVS proved nearly Elo-neutral alone — it only narrows windows. **LMR is what cashes
the scaffolding in.** The bet: with good ordering, a move sitting late in the list
is unlikely to be best, so don't spend full depth proving it — search it *shallower*
and only pay full depth if it surprises us.

### The three-tier search

LMR slots into the PVS scout arm of `negamax`. For a late, eligible move:

```text
1. reduced scout:   s = -negamax(depth-1-r, -alpha-1, -alpha)   // shallow null window
2. if s > alpha:    s = -negamax(depth-1,   -alpha-1, -alpha)   // full depth, null window
3. if alpha<s<beta: s = -negamax(depth-1,   -beta,    -alpha)   // full depth, full window (PVS)
```

Tier 1 is the saving — most late moves fail low even shallow, and we believe them.
Tier 2 is the safety net: a reduced scout that beats alpha might just be
*under-searched*, so we re-verify at full depth before trusting it. Tier 3 is the
ordinary PVS re-search for a genuine new PV. When the reduction `r == 0` (every
non-eligible move) tiers 1–2 collapse to the plain PVS scout — LMR touches **only**
eligible late quiets.

### Who gets reduced

Reduce move `i` only if it's genuinely unlikely-and-quiet: `depth >= 3`, `i >= 3`,
not a capture/promotion (`!is_tactical`), the node isn't in check, the move isn't a
killer, and the move doesn't give check. The forcing moves — captures, checks,
killers, the TT/PV move (ordered first, so `i == 0`) — keep full depth, which is
what stops LMR from walking past a tactic: a mating move is a check, so it is never
reduced.

The `gives_check` test is a full `in_check` scan (not incremental), so it sits last
in the `&&` chain and runs only after the cheap tests pass; the node-in-check scan
is gated on `depth >= 3` so shallow nodes skip it entirely.

### How much

One table, `R(depth, i) = floor(0.75 + ln(depth)·ln(i)/2)` — reductions grow with
depth and with how late the move is — clamped so the reduced depth stays `>= 1`.
Keeping it in a single function makes the formula SPRT-tunable without touching the
search. The table is built once at **startup** (`search::init`), never lazily on a
search's clock — the lesson from the magic-table init bug (a one-time build billed
to the first move truncates it at a short TC).

### Why no invariance test

Unlike PVS, LMR **deliberately** changes the fixed-depth result — it prunes lines a
full search would visit. So there's nothing to assert byte-equal. Correctness is
behavioural instead: a tactical suite (forced mates + a decisively winning position,
each captured from the *pre-LMR* engine at the depth it already solved them) asserts
LMR still finds the win, and a counter test (`lmr_reductions` / `lmr_researches`)
asserts reductions fire in bulk with a bounded re-search rate — a high rate would
mean we're over-reducing.

### What it buys

The big one: **~2 plies deeper in the same time** (warm depth-in-time vs v0.5.0:
5.5→7.1 at 100 ms, 6.3→8.2 at 200 ms, 7.1→9.3 at 500 ms). This is the structural
payoff PVS was laying groundwork for, and the combined PVS+LMR SPRT vs v0.5.0 is the
acceptance gate for both. Later refinements — SEE-gated reductions (#39),
history-scaled reductions (#25) — reduce more for moves that look bad and less for
moves that look good; they're deferred to keep this first signal clean.
