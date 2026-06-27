# 05 — Board representation (deep dive)

How the engine stores a chess position, *why* it's stored that way, and how each
design choice pays off later. This documents `src/board.rs` and `src/bitboard.rs`.
For the original decision, see [ADR 0002](decisions/0002-board-bitboards.md).

---

## The problem a single bitboard can't solve

A **bitboard** is a `u64` — 64 bits, one per square (`a1 = bit 0 … h8 = bit 63`).
A set bit means "something is here." But one bitboard only encodes **occupancy**:

```
   a b c d e f g h
8  . . . . . . . .
7  . . . . . . . .
   ...                 a single bitboard answers
4  . . . . X . . .     "is a piece on e4?"  ->  yes
   ...                 but NOT "which piece?"
1  . . . . . . . .
```

It cannot tell you *what* is on e4. To recover piece identity we layer several
bitboards and read the answer from the overlap.

## The layered solution: 6 piece + 2 color bitboards

`Board` keeps eight bitboards (`src/board.rs`):

```rust
piece_bb: [Bitboard; 6],   // indexed by PieceType: Pawn..King
color_bb: [Bitboard; 2],   // indexed by Color: White, Black
```

- `piece_bb[Knight]` = every square holding a knight, **either color**.
- `color_bb[White]` = every square holding a **white** piece, **any kind**.

A square's full identity is the **intersection** of one piece board and one color
board:

```
white knights = piece_bb[Knight] & color_bb[White]     // one AND of two u64s
```

So "white knight on e4" means bit e4 is set in *both* `piece_bb[Knight]` and
`color_bb[White]` (and in no other piece board).

### Why this layout helps — categories become single instructions

The win is that whole **categories of squares** are one machine instruction:

| Question | Computation | Cost |
|----------|-------------|------|
| All white pieces? | `color_bb[White]` | free (stored) |
| All knights (any color)? | `piece_bb[Knight]` | free (stored) |
| All black rooks? | `piece_bb[Rook] & color_bb[Black]` | 1 AND |
| Every occupied square? | `color_bb[White] \| color_bb[Black]` | 1 OR (`occupied()`) |
| Every empty square? | `!occupied()` | 1 NOT |
| Squares a rook can be captured on? | `enemy = color_bb[!stm]` | free |

Move generation lives on these set operations. A rook slides along a ray until it
hits an `occupied()` bit; a capture target is any move-square that intersects the
enemy color board. None of this is a loop over squares — it's `&`, `|`, `<<`,
`popcount` on 64-bit words, which is why bitboard engines are fast.

### Alternative considered: 12 bitboards

Some engines use 12 boards (one per color×piece) instead of 6+2. Both work; 6+2 is
slightly more compact and makes "all knights regardless of color" free. See ADR
0002.

## The redundant mailbox — the second view

Scanning 6–12 bitboards every time you ask "what's on *this one* square?" is
wasteful when you ask millions of times (make/unmake, evaluation). So `Board`
also keeps a plain array — the **mailbox**:

```rust
mailbox: [Option<Piece>; 64],   // mailbox[28] == Some(white knight)
```

`piece_on(sq)` is then a single array read, O(1):

```rust
pub fn piece_on(&self, sq: Square) -> Option<Piece> {
    self.mailbox[sq.0 as usize]
}
```

### Two views, two strengths

```
bitboards  ->  fast for "all squares matching a pattern"   (movegen, attacks)
mailbox    ->  fast for "what exactly is on THIS square"    (make/unmake, eval)
```

Keeping both is deliberate redundancy. It looks wrong from an app-dev
"single-source-of-truth" instinct, but in an engine it's correct: each view
answers a different question cheaply, and the cost of maintaining both (a couple
of writes per move) is trivial next to the millions of reads it saves. Stockfish
does the same.

## The invariant — and who guards it

The two views must never disagree:

> A square has its bit set in exactly one `piece_bb` and one `color_bb`
> **iff** `mailbox[sq] == Some(that piece)`.

That invariant is the *real product* of `board.rs`. Every later module trusts it.
It is enforced in exactly two places — the only methods that mutate the board:

```rust
put_piece(sq, piece)   // set the bit in piece_bb + color_bb, write mailbox[sq]
remove_piece(sq)       // clear both bits, clear mailbox[sq]
```

Because all mutation funnels through these two functions, there is one place to
get right and one place to audit. The unit tests in `board.rs` assert the two
views stay in sync after put/remove; **perft** (issue #17) later stress-tests the
same invariant across millions of make/unmake cycles — if a sync bug exists, perft
node counts diverge from the published values.

## State beyond the pieces

A position is more than piece placement. `Board` also stores the irreversible/
side state needed for legal play:

```rust
side_to_move:   Color
castling:       CastlingRights   // 4 flags packed in a u8 (WK/WQ/BK/BQ)
ep_square:      Option<Square>   // en-passant target, if any
halfmove_clock: u16              // for the 50-move rule
fullmove_number:u16
```

This is exactly the information a FEN string encodes — which is why the FEN parser
(issue #11) is just "fill an `empty()` board's fields from text."

## How this sets up later phases

- **Move generation (Phase 0)** is written entirely in bitboard set ops over these
  boards.
- **make/unmake (#16)** mutates via `put_piece`/`remove_piece` and pushes the prior
  irreversible state onto an undo stack.
- **Zobrist hashing (Phase 2)** XORs a key per `(piece, square)` change — the exact
  granularity `put_piece`/`remove_piece` already operate at.
- **NNUE (Phase 4)** needs to know precisely which `(piece, square)` features
  changed each move to update its accumulator incrementally — again, exactly what
  these two mutators express. Tracking king squares cleanly matters here too
  (king-relative features). See [04-nnue.md](04-nnue.md).

The through-line: **represent the position so that the operations every later
phase needs are cheap, and funnel all change through two audited mutators.**
