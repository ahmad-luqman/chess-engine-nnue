//! Transposition table: a cache of searched positions, keyed by Zobrist hash.
//!
//! During search the same position is reached by many move orders (a
//! *transposition*). Without a cache we'd re-search each arrival from scratch.
//! The TT stores, per position, the result of searching it — a score, how deep
//! that score was searched, what kind of bound it is, and the best move found —
//! so a later arrival can reuse it: take an immediate cutoff when the stored
//! search was deep enough, and (from #25) try the stored best move first.
//!
//! ## Indexing and collisions
//!
//! The table is a flat `Vec` sized to a power of two, indexed by the low bits of
//! the hash (`hash & mask`). Many positions map to the same slot; we store the
//! *full* 64-bit key in the entry and only trust a slot whose key matches, so a
//! collision is detected and ignored rather than silently returning a wrong
//! score. An all-zero (`key == 0`) slot is the never-written sentinel.
//!
//! ## Replacement
//!
//! One slot per index, so a store must sometimes evict. We keep the entry that
//! is most useful to a future probe: prefer a *deeper* search, but always
//! overwrite a slot from an older search (a different `age`) since stale entries
//! help less. See [`TranspositionTable::store`].

use crate::moves::Move;

/// What a stored score tells us relative to the alpha-beta window it was searched
/// under — see [the wiki](https://www.chessprogramming.org/Node_Types).
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum Bound {
    /// An exact value: the search neither failed high nor low (a PV node).
    #[default]
    Exact,
    /// A lower bound: the true score is at least this (a fail-high / beta cutoff).
    Lower,
    /// An upper bound: the true score is at most this (a fail-low, no move beat alpha).
    Upper,
}

/// One cached position. ~24 bytes after padding; small enough that a 16 MB table
/// holds ~700k entries.
#[derive(Copy, Clone, Debug)]
pub struct Entry {
    /// Full Zobrist key of the cached position (collision guard; 0 = empty slot).
    pub key: u64,
    /// Best move found at this position — the strongest move-ordering signal (#25).
    pub best_move: Move,
    /// Stored score, **mate-adjusted relative to this node** (see `score_to_tt`).
    pub score: i32,
    /// Remaining search depth the score was computed to.
    pub depth: u8,
    /// How to interpret `score` against a window.
    pub bound: Bound,
    /// Search generation this entry was written in, for age-based replacement.
    pub age: u8,
}

impl Default for Entry {
    fn default() -> Entry {
        Entry { key: 0, best_move: Move::NONE, score: 0, depth: 0, bound: Bound::Exact, age: 0 }
    }
}

/// A fixed-size, direct-mapped transposition table.
pub struct TranspositionTable {
    entries: Vec<Entry>,
    /// `len - 1`; ANDed with the hash to index. Zero for a disabled table.
    mask: usize,
    /// Current search generation; bumped by [`new_search`](Self::new_search).
    age: u8,
}

impl TranspositionTable {
    /// A table holding as many entries as fit in `mb` megabytes, rounded **down**
    /// to a power of two (so the index mask is cheap). Always at least one entry.
    pub fn new(mb: usize) -> TranspositionTable {
        let bytes = mb.max(1) * 1024 * 1024;
        let count = (bytes / core::mem::size_of::<Entry>()).max(1);
        let count = prev_power_of_two(count);
        TranspositionTable { entries: vec![Entry::default(); count], mask: count - 1, age: 0 }
    }

    /// A do-nothing table: every probe misses and every store is dropped. Used as
    /// the "no TT" reference in tests so a search behaves exactly like v0.1.0.
    pub fn disabled() -> TranspositionTable {
        TranspositionTable { entries: Vec::new(), mask: 0, age: 0 }
    }

    /// True if this table actually stores anything.
    fn is_enabled(&self) -> bool {
        !self.entries.is_empty()
    }

    /// Reallocate to `mb` megabytes, discarding all entries (UCI `setoption Hash`).
    pub fn resize(&mut self, mb: usize) {
        *self = TranspositionTable::new(mb);
    }

    /// Forget everything (UCI `ucinewgame`).
    pub fn clear(&mut self) {
        for e in &mut self.entries {
            *e = Entry::default();
        }
        self.age = 0;
    }

    /// Begin a new search generation. Entries from prior searches become
    /// preferentially replaceable while their scores remain usable until overwritten.
    pub fn new_search(&mut self) {
        self.age = self.age.wrapping_add(1);
    }

    /// Look up `key`. Returns the entry only on a full-key match (so collisions
    /// miss rather than mislead); `None` for a miss or a disabled table.
    pub fn probe(&self, key: u64) -> Option<Entry> {
        if !self.is_enabled() {
            return None;
        }
        let entry = self.entries[key as usize & self.mask];
        if entry.key == key {
            Some(entry)
        } else {
            None
        }
    }

    /// Insert or replace the entry for `key`. Replacement keeps the more useful
    /// entry: an empty slot or one from an older search is always overwritten;
    /// otherwise we keep whichever was searched deeper.
    pub fn store(&mut self, key: u64, best_move: Move, score: i32, depth: u8, bound: Bound) {
        if !self.is_enabled() {
            return;
        }
        let slot = &mut self.entries[key as usize & self.mask];
        let replace = slot.key == 0 || slot.age != self.age || depth >= slot.depth;
        if replace {
            *slot = Entry { key, best_move, score, depth, bound, age: self.age };
        }
    }
}

/// The largest power of two `<= n` (for `n >= 1`).
fn prev_power_of_two(n: usize) -> usize {
    // `next_power_of_two` of (n/2 + 1) lands on the power at or below n; simpler to
    // shift the highest set bit. For n>=1 this is 1 << floor(log2 n).
    1usize << (usize::BITS - 1 - n.leading_zeros())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_of_two_sizing() {
        assert_eq!(prev_power_of_two(1), 1);
        assert_eq!(prev_power_of_two(2), 2);
        assert_eq!(prev_power_of_two(3), 2);
        assert_eq!(prev_power_of_two(1000), 512);
        // A 16 MB table is a power-of-two count of entries.
        let tt = TranspositionTable::new(16);
        assert!(tt.entries.len().is_power_of_two());
    }

    #[test]
    fn store_then_probe_round_trips() {
        let mut tt = TranspositionTable::new(1);
        tt.new_search();
        let mv = Move::quiet(crate::types::Square(12), crate::types::Square(28));
        tt.store(0xDEAD_BEEF, mv, 42, 5, Bound::Exact);
        let e = tt.probe(0xDEAD_BEEF).expect("just stored");
        assert_eq!(e.best_move, mv);
        assert_eq!(e.score, 42);
        assert_eq!(e.depth, 5);
        assert_eq!(e.bound, Bound::Exact);
    }

    #[test]
    fn collision_on_a_different_key_misses() {
        let mut tt = TranspositionTable::new(1);
        tt.new_search();
        tt.store(0x1234, Move::NONE, 1, 1, Bound::Exact);
        // A key sharing the same index slot but differing in high bits must not
        // be served the wrong entry.
        let other = 0x1234 ^ (1u64 << 40);
        assert!(tt.probe(other).is_none());
    }

    #[test]
    fn disabled_table_never_stores_or_serves() {
        let mut tt = TranspositionTable::disabled();
        tt.new_search();
        tt.store(0xABC, Move::NONE, 1, 9, Bound::Exact);
        assert!(tt.probe(0xABC).is_none());
    }

    #[test]
    fn deeper_entry_is_preferred_within_a_search() {
        let mut tt = TranspositionTable::new(1);
        tt.new_search();
        tt.store(0x55, Move::NONE, 1, 8, Bound::Exact);
        // A shallower store in the same generation must not evict the deep entry.
        tt.store(0x55, Move::NONE, 2, 3, Bound::Exact);
        assert_eq!(tt.probe(0x55).unwrap().depth, 8);
        assert_eq!(tt.probe(0x55).unwrap().score, 1);
    }
}
