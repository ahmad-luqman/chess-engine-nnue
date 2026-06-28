//! Zobrist hashing: a cheap, near-unique 64-bit key for a position.
//!
//! The idea (Zobrist, 1970): assign a random 64-bit constant to every board
//! *feature* — each (color, piece-type, square), the side to move, the castling
//! rights, and the en-passant file — and define a position's key as the XOR of
//! the constants for the features it has. Because XOR is its own inverse, a move
//! that changes a handful of features updates the key in O(features changed):
//! XOR the departing feature's constant out, XOR the arriving one in. That
//! incremental update lives in [`Board::make_move`]; this module supplies the
//! constants and a from-scratch [`compute`] used to seed a freshly parsed board
//! and to assert the incremental key never drifts.
//!
//! The constants are generated at compile time from a fixed seed (a `const`
//! splitmix64 stream), so hashes are reproducible across runs — important for
//! debugging and for any on-disk structure (opening books) later.
//!
//! **En-passant subtlety:** the ep file is hashed only when an enemy pawn can
//! *actually* capture en passant, not merely when an ep square exists — see
//! [`Board::ep_zobrist`]. Two positions reached by different move orders are the
//! same position only under that rule (e.g. `1.e4 e5 2.Nf3` ≡ `1.Nf3 e5 2.e4`).

use crate::board::Board;
use crate::types::{Color, PieceType};

/// Arbitrary nonzero seed (the golden-ratio constant) for the key stream. Any
/// fixed value works; this one is conventional for splitmix64.
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// One step of splitmix64 — a tiny, high-quality PRNG that is `const`-friendly
/// (just wrapping arithmetic and shifts), so the whole key table is built by the
/// compiler. Advances `state` and returns the next pseudo-random word.
const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The full set of Zobrist constants, generated once at compile time.
pub struct Keys {
    /// `[color][piece_type][square]` — one key per colored piece per square.
    pub piece: [[[u64; 64]; 6]; 2],
    /// XORed into the key exactly when it is Black's turn.
    pub side: u64,
    /// Indexed by the 4-bit [`CastlingRights`](crate::board::CastlingRights)
    /// bitset, so any rights change is a single XOR-out/XOR-in of two entries.
    pub castling: [u64; 16],
    /// One key per file `a..h`, XORed in only for a *capturable* ep square.
    pub ep_file: [u64; 8],
}

/// Build every key from one deterministic splitmix64 stream. The draw order is
/// fixed (pieces, then side, castling, ep files), so the constants are stable as
/// long as this function is unchanged.
const fn generate() -> Keys {
    let mut state = SEED;

    let mut piece = [[[0u64; 64]; 6]; 2];
    let mut c = 0;
    while c < 2 {
        let mut p = 0;
        while p < 6 {
            let mut s = 0;
            while s < 64 {
                piece[c][p][s] = splitmix64(&mut state);
                s += 1;
            }
            p += 1;
        }
        c += 1;
    }

    let side = splitmix64(&mut state);

    let mut castling = [0u64; 16];
    let mut i = 0;
    while i < 16 {
        castling[i] = splitmix64(&mut state);
        i += 1;
    }

    let mut ep_file = [0u64; 8];
    let mut f = 0;
    while f < 8 {
        ep_file[f] = splitmix64(&mut state);
        f += 1;
    }

    Keys { piece, side, castling, ep_file }
}

/// The compile-time-generated Zobrist constants. `make_move`, `compute`, and the
/// FEN parser all read from this single source so their keys can never disagree.
pub const KEYS: Keys = generate();

/// Compute a board's Zobrist key from scratch.
///
/// Used to seed a freshly parsed board (FEN) and as the oracle for the
/// `debug_assert` in `make_move` that the incrementally maintained key matches a
/// full recomputation at every node. Must XOR exactly the same features, in any
/// order, that the incremental update toggles — including the *capturable*-only
/// en-passant rule via [`Board::ep_zobrist`].
pub fn compute(board: &Board) -> u64 {
    let mut h = 0u64;

    for color in [Color::White, Color::Black] {
        for pt in [
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
            PieceType::King,
        ] {
            let mut bb = board.pieces(pt).intersect(board.color(color));
            while let Some(sq) = bb.pop_lsb() {
                h ^= KEYS.piece[color.index()][pt.index()][sq.0 as usize];
            }
        }
    }

    if board.side_to_move == Color::Black {
        h ^= KEYS.side;
    }
    h ^= KEYS.castling[board.castling.0 as usize];
    h ^= board.ep_zobrist();

    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use core::str::FromStr;

    fn board(fen: &str) -> Board {
        Board::from_str(fen).unwrap()
    }

    #[test]
    fn keys_are_distinct_and_nonzero() {
        // A weak but cheap sanity check: no key collides with another and none is
        // zero (a zero key would silently contribute nothing). Collect a sample.
        let mut seen = std::collections::HashSet::new();
        for c in 0..2 {
            for p in 0..6 {
                for s in 0..64 {
                    let k = KEYS.piece[c][p][s];
                    assert_ne!(k, 0);
                    assert!(seen.insert(k), "duplicate piece key");
                }
            }
        }
        assert!(seen.insert(KEYS.side));
        for k in KEYS.castling {
            assert!(seen.insert(k));
        }
        for k in KEYS.ep_file {
            assert!(seen.insert(k));
        }
    }

    #[test]
    fn fen_sets_key_to_from_scratch_value() {
        let b = board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert_eq!(b.hash, compute(&b));
    }

    #[test]
    fn capturable_ep_changes_the_key_but_idle_ep_does_not() {
        // A real ep opportunity (white pawn on e5, black just played ...d5) must
        // change the key versus the same position with no ep square...
        let live = board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
        let live_no_ep = board("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1");
        assert_ne!(live.hash, live_no_ep.hash, "capturable ep must be hashed");

        // ...but an ep square no pawn can act on (white pawn on e4, not e5) must
        // hash identically to the same position with no ep square. This is the
        // rule that makes transposed positions share a key.
        let idle = board("4k3/8/8/3p4/4P3/8/8/4K3 w - d6 0 1");
        let idle_no_ep = board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1");
        assert_eq!(idle.hash, idle_no_ep.hash, "idle ep must not be hashed");
    }
}
