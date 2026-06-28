//! Magic bitboards: sliding-piece attacks as one multiply-shift table lookup.
//!
//! A rook/bishop's reach depends on the blockers along its rays, so it can't be
//! a plain per-square table like the knight/king attacks. The ray-walk loop
//! (`movegen::ray_attacks`) is correct but scans square by square. Magic
//! bitboards turn the lookup into:
//!
//! ```text
//! attacks = TABLE[square][ ((occupied & mask) * magic) >> shift ]
//! ```
//!
//! The trick (Romstad/Kannan): only the squares *between* the slider and the
//! board edge can block it — the **relevant mask** — and there are at most
//! 2¹² of those for a rook. A well-chosen **magic** multiplier scatters each of
//! those occupancy patterns to a distinct index (or to one that happens to share
//! the same attack set — a harmless collision), so a multiply and a shift index
//! straight into a precomputed table.
//!
//! ## Finding magics at startup, not hard-coding them
//!
//! Published magic constants exist, but a single mistyped digit corrupts attacks
//! silently (only perft would catch it). Instead we *find* magics at first use
//! with a fixed-seed PRNG and **verify every candidate against the ray-walk
//! oracle** before accepting it. Same determinism (fixed seed), but correct by
//! construction. Build cost is a few milliseconds, paid once via [`OnceLock`].

use std::sync::OnceLock;

use crate::bitboard::Bitboard;
use crate::movegen::{ray_attacks, BISHOP_DIRS, ROOK_DIRS};
use crate::types::Square;

/// Per-square magic: the relevant-occupancy `mask`, the `magic` multiplier, the
/// `shift` (`64 - relevant_bits`), and the attack table indexed by the magic.
struct SquareMagic {
    mask: u64,
    magic: u64,
    shift: u32,
    attacks: Vec<Bitboard>,
}

impl SquareMagic {
    /// Map an occupancy to its table index: mask off the irrelevant squares,
    /// multiply by the magic, and take the high `relevant_bits` bits.
    fn index(&self, occupied: u64) -> usize {
        ((occupied & self.mask).wrapping_mul(self.magic) >> self.shift) as usize
    }

    fn lookup(&self, occupied: Bitboard) -> Bitboard {
        self.attacks[self.index(occupied.0)]
    }
}

/// All 64 rook and 64 bishop square-magics, built once.
struct Magics {
    rook: Vec<SquareMagic>,
    bishop: Vec<SquareMagic>,
}

static MAGICS: OnceLock<Magics> = OnceLock::new();

fn magics() -> &'static Magics {
    MAGICS.get_or_init(build)
}

/// Squares a rook on `sq` attacks given `occupied` — magic lookup.
pub fn rook_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    magics().rook[sq.0 as usize].lookup(occupied)
}

/// Squares a bishop on `sq` attacks given `occupied` — magic lookup.
pub fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    magics().bishop[sq.0 as usize].lookup(occupied)
}

/// The relevant-occupancy mask for a slider on `sq`: the ray squares whose
/// occupancy can block it — i.e. every ray square *except* the one against the
/// edge (a piece on the edge can't block anything further along that ray, so its
/// occupancy is irrelevant). The slider's own square is excluded too.
fn relevant_mask(sq: Square, dirs: &[(i8, i8)]) -> u64 {
    let mut mask = 0u64;
    let (f0, r0) = (sq.file() as i8, sq.rank() as i8);
    for &(df, dr) in dirs {
        let (mut f, mut r) = (f0 + df, r0 + dr);
        // Include (f, r) only while the *next* step stays on the board, which
        // drops the edge square of each ray.
        while (0..8).contains(&(f + df)) && (0..8).contains(&(r + dr)) {
            mask |= 1u64 << (r * 8 + f);
            f += df;
            r += dr;
        }
    }
    mask
}

/// xorshift64* step — a tiny deterministic PRNG for the magic search.
fn next_rand(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// A *sparse* random word (ANDing three randoms leaves few bits set); magics with
/// few set bits scatter occupancies far better, so the search converges fast.
fn sparse_rand(state: &mut u64) -> u64 {
    next_rand(state) & next_rand(state) & next_rand(state)
}

/// Find a magic for one square and build its attack table, verified against the
/// ray-walk oracle for every occupancy subset of the mask.
fn build_square(sq: Square, dirs: &[(i8, i8)], rng: &mut u64) -> SquareMagic {
    let mask = relevant_mask(sq, dirs);
    let bits = mask.count_ones();
    let shift = 64 - bits;
    let size = 1usize << bits;

    // Enumerate every occupancy subset of the mask (carry-rippler) and its true
    // attack set — the reference the magic must reproduce.
    let mut subsets = Vec::with_capacity(size);
    let mut reference = Vec::with_capacity(size);
    let mut sub = 0u64;
    loop {
        subsets.push(sub);
        reference.push(ray_attacks(sq, Bitboard(sub), dirs));
        sub = sub.wrapping_sub(mask) & mask; // next subset; wraps to 0 when done
        if sub == 0 {
            break;
        }
    }

    // Try magics until one maps every subset without a *harmful* collision (two
    // subsets sharing an index but needing different attack sets).
    loop {
        let magic = sparse_rand(rng);
        // Cheap reject: a good magic spreads the top bits of mask*magic.
        if (mask.wrapping_mul(magic) >> 56).count_ones() < 6 {
            continue;
        }
        let mut table = vec![Bitboard::EMPTY; size];
        let mut filled = vec![false; size];
        let mut ok = true;
        for (i, &occ) in subsets.iter().enumerate() {
            let idx = (occ.wrapping_mul(magic) >> shift) as usize;
            if !filled[idx] {
                filled[idx] = true;
                table[idx] = reference[i];
            } else if table[idx] != reference[i] {
                ok = false;
                break;
            }
        }
        if ok {
            return SquareMagic { mask, magic, shift, attacks: table };
        }
    }
}

/// Build every square-magic. Deterministic: fixed seed, oracle-verified.
fn build() -> Magics {
    let mut rng: u64 = 0x00C0_FFEE_1234_5678; // arbitrary nonzero seed
    let rook = (0..64).map(|s| build_square(Square(s), &ROOK_DIRS, &mut rng)).collect();
    let bishop = (0..64).map(|s| build_square(Square(s), &BISHOP_DIRS, &mut rng)).collect();
    Magics { rook, bishop }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pseudo-random occupancy for testing (independent of the build PRNG).
    fn rand_occ(state: &mut u64) -> Bitboard {
        // Medium density: AND two randoms so the board isn't nearly full.
        Bitboard(next_rand(state) & next_rand(state))
    }

    #[test]
    fn magic_attacks_match_the_ray_oracle_everywhere() {
        // The correctness gate for magics: for every square and a spread of
        // random occupancies, the magic lookup must equal the ray-walk result
        // exactly — for rooks and bishops.
        let mut state = 0xDEAD_BEEF_CAFE_F00D;
        for s in 0..64u8 {
            let sq = Square(s);
            for _ in 0..200 {
                let occ = rand_occ(&mut state);
                assert_eq!(
                    rook_attacks(sq, occ),
                    ray_attacks(sq, occ, &ROOK_DIRS),
                    "rook mismatch on {sq} with occ {:#x}",
                    occ.0
                );
                assert_eq!(
                    bishop_attacks(sq, occ),
                    ray_attacks(sq, occ, &BISHOP_DIRS),
                    "bishop mismatch on {sq} with occ {:#x}",
                    occ.0
                );
            }
        }
    }

    #[test]
    fn corner_rook_on_empty_board_reaches_both_edges() {
        // a1 rook on an empty board hits all of the a-file and 1st rank (14 sqs).
        let a1 = Square(0);
        assert_eq!(rook_attacks(a1, Bitboard::EMPTY).count(), 14);
    }
}
