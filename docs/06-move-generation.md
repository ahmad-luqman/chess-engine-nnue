# 06 — Move generation (deep dive)

How the engine turns a position into the exact list of **legal** moves, applies
and reverts them, and *proves the whole thing correct* before any search exists.
This documents `src/moves.rs`, the generator half of `src/movegen.rs`, the
make/unmake half of `src/board.rs`, and `src/perft.rs`. It builds on the board
representation in [05-board-representation.md](05-board-representation.md).

The Phase 0 contract is narrow but absolute: **generate all legal moves, and only
legal moves, for any position** — verified by perft against published node counts.
Everything in later phases (search, eval, NNUE) sits on top and silently inherits
any bug here, which is why this is the one subsystem we gate on hard.

---

## 1. Encoding a move in 16 bits (`moves.rs`)

A move is three things: a **from** square, a **to** square, and a **flag** saying
what kind of move it is. All three pack into a single `u16`:

```
bits  0– 5   from square (0..=63)      6 bits
bits  6–11   to square   (0..=63)      6 bits
bits 12–15   flag        (0..=15)      4 bits
```

Why pack instead of a struct with three fields? Because **move lists are the hot
data of search**: a position has ~35 legal moves, and search visits millions of
positions, sorting and copying these lists constantly. Sixteen bits keeps a move
in a register and a whole move list cache-resident. This is the Stockfish/Dragon
lineage choice (see the root `CLAUDE.md`).

### The 4-bit flag scheme

The flag isn't an arbitrary enum — its bits are laid out so the questions search
asks most often are single bit-tests:

```
 code  bit3=promo  bit2=capture   meaning
   0       .           .          quiet
   1       .           .          double pawn push
   2       .           .          king-side castle
   3       .           .          queen-side castle
   4       .           x          capture
   5       .           x          en-passant capture
   8       x           .          knight promotion
   9       x           .          bishop promotion
  10       x           .          rook   promotion
  11       x           .          queen  promotion
  12       x           x          knight promotion-capture
  …                                …
  15       x           x          queen  promotion-capture
```

So `is_capture()` is `flag & 0b0100`, `is_promotion()` is `flag & 0b1000`, and the
promoted piece is the low two bits (`0=knight … 3=queen`). One subtlety the code
calls out: those promo codes do **not** line up with `PieceType`'s numbering
(`Pawn=0, Knight=1, …`), so `promotion_piece()` uses an explicit `match`, never a
cast.

### En passant gets its own code — on purpose

Notice code 5 (en-passant) is distinct from code 4 (ordinary capture), even though
both set the capture bit. This redundancy is deliberate and matters downstream:
**the pawn an en-passant move captures is not on the `to` square** — it sits beside
the moving pawn. Make/unmake must branch on `is_en_passant()` to find the victim;
it can never infer the captured square from the capture bit alone. Encoding the
distinction now means the bug can't appear later.

`Display` renders UCI long algebraic — `e2e4`, `e7e8q`, and castling as the king's
own move `e1g1` (never `O-O`, which UCI doesn't use). `Move::NONE` (from == to == 0)
is the null sentinel search will use later.

## 2. Attacks: what a piece *threatens* (`movegen.rs`, first half)

Before generating moves you need, for any piece on any square, the set of squares
it attacks. Pieces split by geometry:

- **Non-sliders** (knight, king, pawn) attack a *fixed* pattern relative to their
  square, independent of the rest of the board. These are precomputed once into
  `[Bitboard; 64]` lookup tables at compile time (`const fn`).
- **Sliders** (bishop, rook, queen) attack *along rays until blocked*, so their
  attacks depend on occupancy and are computed per call by walking each ray until
  it hits an occupied square (the blocker is included — it might be a capture).

```
knight on d4  ->  table lookup            (occupancy-independent)
rook   on d4  ->  walk N/E/S/W rays       (stops at first blocker each way)
```

The recurring bug in table generation is **file wraparound**: "one file left of a1"
must be off-board, not h-file of the rank below. Every builder guards it by
computing target file/rank as signed integers and discarding anything outside
`0..8`.

> **Update (Phase 2, issue #27):** sliders now use **magic bitboards** — one
> multiply + one table lookup — replacing the ray loop. The ray loop survives as
> `ray_attacks`: it's the *oracle* the magic tables are built and verified
> against. The public `rook_attacks`/`bishop_attacks` signatures are unchanged,
> so `is_square_attacked` and `generate_legal` were untouched, and perft node
> counts are identical (the gate). See the section below and
> [ADR 0008](decisions/0008-magic-bitboards.md).

#### Magic bitboards (the fast slider path)

A slider's reach depends on blockers, so it can't be a plain per-square table.
The magic trick: for a square, only the ray squares *between* it and the board
edge can block it (the **relevant mask** — the edge square's occupancy never
matters). A carefully chosen **magic** multiplier maps each masked occupancy to a
table index:

```
attacks = TABLE[sq][ ((occupied & mask[sq]) * magic[sq]) >> shift[sq] ]
```

Different occupancies may collide on an index *as long as they share the same
attack set* (a harmless collision) — that's what makes the tables compact (≤ 2¹²
entries per rook square).

We **find** the magics at first use with a fixed-seed PRNG rather than hard-coding
published constants, and verify every candidate against `ray_attacks` for all
occupancy subsets before accepting it — so the tables are correct by construction,
not by trusting a transcribed constant. Build cost is a few ms, once, behind a
`OnceLock` (`src/magic.rs`).

### `is_square_attacked` — running the tables backwards

The key primitive for legality and check detection: *is square `s` attacked by
color `c`?* The trick is **symmetry**. A knight attacks `s` from exactly the
squares a knight *on* `s` would attack, so:

```
attacked by a knight  ⇔  knight_attacks(s) ∩ (enemy knights) ≠ ∅
attacked by a slider  ⇔  rook/bishop_attacks(s, occ) ∩ (enemy rooks/bishops/queens) ≠ ∅
```

Pawns are the one exception — their attacks aren't symmetric — so we probe with the
*opposite* color's pawn table from `s`. One function answers both "is my king in
check?" and "can the king legally step here?".

## 3. Legal move generation: pseudo-legal + filter (`movegen.rs`, second half)

There are two schools for generating legal moves:

1. **Fully legal**: compute pins, checks, and evasions up front so every move
   emitted is already legal. Fast, but intricate and easy to get subtly wrong.
2. **Pseudo-legal + filter**: emit every move that respects piece geometry and
   own-piece blocking, then throw out the ones that leave your own king in check.
   Slower, but obviously correct.

Phase 0 takes door #2 — **correctness first**. The filter is "copy-make":

```rust
// for each pseudo-legal move, on a throwaway board:
let undo = work.make_move(mv);
let legal = !is_square_attacked(work, king_square(work, us), enemy);
work.unmake_move(mv, undo);     // keep `mv` iff legal
```

### Three hard cases this solves *for free*

The beauty of "make the move, then ask if my king is safe" is that nasty cases
need no special code:

- **Pinned pieces** — a piece pinned to the king simply produces a position where
  the king is attacked after it moves; the filter drops it.
- **Double check** — only king moves will leave the king safe; everything else is
  filtered, automatically.
- **The en-passant pin** — the famous trap:

  ```
  8/8/8/8/k2pP2R/8/8/4K3 b - e3
       black king a4 … black pawn d4 … white pawn e4 … white rook h4
  ```

  Capturing `e4` en-passant removes *two* pawns from the 4th rank at once,
  unveiling the rook's attack on the king. A naive generator misses this; copy-make
  catches it because after the capture the king is simply in check.

### The one case copy-make can't see: castling

The filter only checks where the king *lands*. Castling slides the king *across*
squares it never ends on, so a king could legally castle "through" an attacked f1.
Castling is therefore validated explicitly during generation — rights present, the
squares between king and rook empty, and the king's **entire path** (start, transit,
destination) unattacked. (The square beside the queen's rook, the b-file, may be
attacked — only the king's three squares matter.) These moves are appended
already-legal and skip the filter.

Pawns carry the rest of the fiddly rules: single and double pushes, diagonal
captures, **four promotion moves per reachable back-rank square**, and en-passant.

## 4. make / unmake — apply and perfectly revert (`board.rs`)

Search explores by making a move, recursing, then taking it back. Both halves go
through `put_piece`/`remove_piece` (the two audited mutators from doc 05), so the
bitboard and mailbox views stay in lockstep.

`make_move` does the full update — moving piece (with promotion, castled rook, and
en-passant victim), side to move, castling rights, ep square, and the clocks — and
returns an **`Undo`**:

```rust
struct Undo {            // only the IRREVERSIBLE state
    captured: Option<Piece>,   // what was removed (incl. ep victim)
    castling: CastlingRights,  // rights BEFORE the move
    ep_square: Option<Square>,
    halfmove_clock: u16,
}
```

It stores only what a move *destroys and can't recompute*. The moving piece, side
to move, and move number are all recoverable from the move itself, so they aren't
saved.

### Two details worth their own note

- **Castling rights via a touch-mask.** Rights are cleared by ANDing away
  `castling_loss_mask(from) | castling_loss_mask(to)`. A king's home square clears
  both its rights; a rook's home square clears that side's right. Applying it to
  *both* endpoints folds three rules into one: king moves, rook moves, *and* rook
  captures (the `to` endpoint) all drop the right uniformly.
- **The `Undo` return value *is* the undo stack.** Rather than a `Vec<Undo>` field
  on `Board`, `make_move` returns the `Undo` and the caller holds it across the
  recursion. This sidesteps a borrow-checker fight (search wants `&mut self` while
  also reading history) and keeps `Board` a plain value with no hidden state.

## 5. Perft — the correctness oracle (`perft.rs`)

`perft(depth)` counts the leaf nodes of the legal-move tree to a fixed depth by
making and unmaking every move:

```rust
fn perft(board, depth):
    if depth == 0: return 1
    moves = generate_legal(board)
    if depth == 1: return moves.len()     // bulk-count leaves: no make/unmake
    sum perft(child, depth-1) over all moves
```

Why it's *the* gate before any search work: a single illegal, missing, or
duplicated move *anywhere* in the tree perturbs the total, and the reference counts
for standard positions are published to the node. Matching them simultaneously
proves **generation, legality filtering, make, and unmake** all correct. A movegen
bug that slips past here would silently corrupt every eval and search measurement
built on top — so the iron rule is blunt: *no search before perft passes.*

The standard test positions each stress a different corner — the start position,
**Kiwipete** (castling + pins + captures galore), and "positions 3/4/5" (en-passant
+ checks, promotions + pins, tactical mess). `perft_divide` prints the node count
per root move, the classic way to bisect a wrong total against a reference engine
move by move.

Because `cargo test` runs *debug* builds, the everyday suite checks shallow depths
(fast); the full depth-5/6 gate — including Kiwipete depth-5 = 193,690,690 nodes —
is `#[ignore]`d and run in release, per "measure in `--release`":

```
cargo test --release -- --ignored perft_matches_reference_deep
cargo run  --release -- perft 6                # the binary's perft CLI
```

## How this sets up later phases

- **Search (Phase 1+)** consumes `generate_legal` and make/unmake directly. Its hot
  loop is exactly the perft loop with evaluation and pruning bolted on.
- **Move ordering (Phase 2)** reads the flag bits for free — captures and promotions
  (the high-value moves) are a bitmask away, no recomputation.
- **Speed (Phase 2)** swapped ray-loop sliders for **magic bitboards** (done,
  issue #27 — see above), with perft as the regression test that kept the rewrite
  honest; a staged/fully-legal generator may still come later.
- **NNUE (Phase 4)** hooks its accumulator update into make/unmake, which already
  expresses change at exactly the `(piece, square)` granularity the network needs.

The through-line: **emit only legal moves, make them reversibly, and prove it with
perft — so every later phase can trust the move list without re-checking it.**
