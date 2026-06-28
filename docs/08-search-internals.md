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

*Sections for the transposition table (#24), move ordering (#25), quiescence
(#26), and draw detection (#28) land with those issues.*
