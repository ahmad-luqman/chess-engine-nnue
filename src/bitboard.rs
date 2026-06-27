//! Bitboard: a set of squares packed into a `u64`.
//!
//! Bit `i` (value `1 << i`) represents `Square(i)`. With our a1 = 0 convention,
//! bit 0 = a1, bit 7 = h1, bit 63 = h8.
//!
//! Almost every hot path in the engine is bitboard math, so these primitives
//! must be tiny and obvious. Rust maps `count_ones`/`trailing_zeros` to single
//! CPU instructions (`POPCNT` / `TZCNT`).

use crate::types::Square;

#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub struct Bitboard(pub u64);

impl Bitboard {
    /// No squares set.
    pub const EMPTY: Bitboard = Bitboard(0);

    /// A bitboard with exactly one square set.
    pub fn from_square(sq: Square) -> Bitboard {
        Bitboard(1u64 << sq.0)
    }

    /// Number of squares set (population count).
    pub fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// True if no squares are set.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    // ───────────────────────────────────────────────────────────────────────
    //  YOUR CONTRIBUTION (Phase 0, first task)
    //
    //  Implement the three methods below. They're small, but they're the
    //  vocabulary the whole move generator is written in, so getting the
    //  semantics right matters. Reference: the a1 = 0, `1 << i` convention above.
    //
    //  Run `cargo test` when done — tests at the bottom of this file check them.
    // ───────────────────────────────────────────────────────────────────────

    /// Is `sq` a member of this set?
    pub fn contains(self, sq: Square) -> bool {
        // AND with a one-square mask; non-zero iff bit `sq.0` was set.
        self.0 & (1u64 << sq.0) != 0
    }

    /// Return a copy with `sq` added to the set (set the bit).
    ///
    /// Design choice to make consciously: immutable (`-> Bitboard`, shown) vs.
    /// mutating (`&mut self`). We use immutable here for composability; you'll
    /// see both styles in real engines.
    pub fn with(self, sq: Square) -> Bitboard {
        // OR in the one-square mask. Idempotent: setting an already-set bit is
        // a harmless no-op.
        Bitboard(self.0 | (1u64 << sq.0))
    }

    /// Remove and return the least-significant set square (the lowest index),
    /// mutating self to clear it. Returns `None` if empty.
    ///
    /// This is THE iteration primitive: `while let Some(sq) = bb.pop_lsb() { ... }`
    /// walks every piece in a bitboard. Hint: `trailing_zeros` gives the index of
    /// the lowest set bit; `self.0 & (self.0 - 1)` clears it.
    pub fn pop_lsb(&mut self) -> Option<Square> {
        // Guard empty FIRST: trailing_zeros() on 0 returns 64 (a bogus square).
        if self.0 == 0 {
            return None;
        }
        // Index of the lowest set bit (0..=63), then clear that bit with the
        // `x & (x - 1)` trick (single BLSR instruction).
        let idx = self.0.trailing_zeros() as u8;
        self.0 &= self.0 - 1;
        Some(Square(idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(i: u8) -> Square {
        Square(i)
    }

    #[test]
    fn contains_and_with() {
        let bb = Bitboard::EMPTY.with(sq(0)).with(sq(63));
        assert!(bb.contains(sq(0)));
        assert!(bb.contains(sq(63)));
        assert!(!bb.contains(sq(1)));
        assert_eq!(bb.count(), 2);
    }

    #[test]
    fn pop_lsb_walks_in_order() {
        let mut bb = Bitboard::EMPTY.with(sq(5)).with(sq(2)).with(sq(40));
        assert_eq!(bb.pop_lsb(), Some(sq(2)));
        assert_eq!(bb.pop_lsb(), Some(sq(5)));
        assert_eq!(bb.pop_lsb(), Some(sq(40)));
        assert_eq!(bb.pop_lsb(), None);
        assert!(bb.is_empty());
    }
}
