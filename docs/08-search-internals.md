# 08 — Search internals (deep dive)

Phase 2 turns the Phase 1 toy searcher into a *real* engine: a position identity
it can cache and compare (Zobrist), a transposition table, principled move
ordering, quiescence at the leaves, and draw detection. This is the running
deep-dive for that work — one section lands per issue (#23–#28). It assumes the
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

*Sections for quiescence (#26) and draw detection (#28) land with those issues.*
