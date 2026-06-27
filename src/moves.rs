//! Move encoding: a whole move packed into 16 bits.
//!
//! A move is `(from, to, flag)`. We pack all three into a single `u16` rather
//! than a multi-field struct because move *lists* are the hot data of search:
//! generation produces dozens per position and the engine sorts and walks them
//! millions of times. Sixteen bits keeps a move in a register and a `MoveList`
//! cache-friendly — the Stockfish/Dragon lineage this engine follows does the
//! same. This module is encoding *only*; producing legal moves is issue #15.
//!
//! Layout:
//!
//! ```text
//! bits  0– 5   from square (0..=63)
//! bits  6–11   to square   (0..=63)
//! bits 12–15   flag        (MoveFlag, 0..=15)
//! ```
//!
//! The 4-bit flag is the classic chessprogramming-wiki scheme, chosen so that
//! the questions search asks most often are single bit tests:
//!
//! ```text
//!  code  bit3=promo bit2=capture  meaning
//!   0      .           .          quiet
//!   1      .           .          double pawn push
//!   2      .           .          king-side castle
//!   3      .           .          queen-side castle
//!   4      .           x          capture
//!   5      .           x          en-passant capture
//!   8      x           .          knight promotion
//!   9      x           .          bishop promotion
//!  10      x           .          rook promotion
//!  11      x           .          queen promotion
//!  12      x           x          knight promotion-capture
//!  13      x           x          bishop promotion-capture
//!  14      x           x          rook promotion-capture
//!  15      x           x          queen promotion-capture
//! ```
//!
//! Codes 6 and 7 are unused. The low two bits of a promotion code select the
//! promoted piece (0=knight … 3=queen), so `promotion_piece` is a 2-bit lookup.

use crate::types::{PieceType, Square};

/// What kind of move this is — promotion, capture, castle, en-passant — encoded
/// as one of the 16 codes above.
///
/// A newtype over `u8` (not an `enum`) so the raw code can be packed into and
/// unpacked from the move word with plain shifts, no `transmute` and no
/// reliance on enum discriminant layout.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct MoveFlag(u8);

impl MoveFlag {
    pub const QUIET: MoveFlag = MoveFlag(0);
    pub const DOUBLE_PAWN_PUSH: MoveFlag = MoveFlag(1);
    pub const KING_CASTLE: MoveFlag = MoveFlag(2);
    pub const QUEEN_CASTLE: MoveFlag = MoveFlag(3);
    pub const CAPTURE: MoveFlag = MoveFlag(4);
    pub const EN_PASSANT: MoveFlag = MoveFlag(5);
    pub const KNIGHT_PROMO: MoveFlag = MoveFlag(8);
    pub const BISHOP_PROMO: MoveFlag = MoveFlag(9);
    pub const ROOK_PROMO: MoveFlag = MoveFlag(10);
    pub const QUEEN_PROMO: MoveFlag = MoveFlag(11);
    pub const KNIGHT_PROMO_CAPTURE: MoveFlag = MoveFlag(12);
    pub const BISHOP_PROMO_CAPTURE: MoveFlag = MoveFlag(13);
    pub const ROOK_PROMO_CAPTURE: MoveFlag = MoveFlag(14);
    pub const QUEEN_PROMO_CAPTURE: MoveFlag = MoveFlag(15);

    const PROMO_BIT: u8 = 0b1000;
    const CAPTURE_BIT: u8 = 0b0100;

    /// True for any promotion (with or without capture).
    pub fn is_promotion(self) -> bool {
        self.0 & Self::PROMO_BIT != 0
    }

    /// True for any move that removes an enemy piece — ordinary captures,
    /// en-passant, and promotion-captures all set the capture bit.
    pub fn is_capture(self) -> bool {
        self.0 & Self::CAPTURE_BIT != 0
    }

    /// True only for en-passant. This is a *distinct code*, not just "a capture
    /// whose target is empty": the pawn it removes does not stand on the move's
    /// `to` square, so make/unmake (issue #16) must key off this flag — never
    /// the capture bit — to find the captured pawn.
    pub fn is_en_passant(self) -> bool {
        self == Self::EN_PASSANT
    }

    /// True for either castle. Use [`is_king_castle`](Self::is_king_castle) to
    /// tell the sides apart.
    pub fn is_castle(self) -> bool {
        self == Self::KING_CASTLE || self == Self::QUEEN_CASTLE
    }

    /// True for the king-side castle specifically.
    pub fn is_king_castle(self) -> bool {
        self == Self::KING_CASTLE
    }

    /// The piece a pawn promotes to, or `None` if this is not a promotion.
    ///
    /// The low two bits of a promotion code select the piece. We map them with
    /// an explicit `match` rather than casting: `PieceType` is `Pawn=0,
    /// Knight=1, …`, so the codes (0=knight) do **not** line up with the enum's
    /// indices, and the cast would silently produce the wrong piece.
    pub fn promotion_piece(self) -> Option<PieceType> {
        if !self.is_promotion() {
            return None;
        }
        Some(match self.0 & 0b11 {
            0 => PieceType::Knight,
            1 => PieceType::Bishop,
            2 => PieceType::Rook,
            _ => PieceType::Queen,
        })
    }

    /// The raw 0..=15 code, for packing into a [`Move`].
    fn code(self) -> u8 {
        self.0
    }
}

/// A move, packed into 16 bits as `from | to << 6 | flag << 12`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Move(u16);

impl Move {
    /// The null move: no origin, no destination, no flag. Distinct from every
    /// real move (a real move never has `from == to`). Search later uses it as
    /// the "no move yet" sentinel and for null-move pruning.
    pub const NONE: Move = Move(0);

    const TO_SHIFT: u16 = 6;
    const FLAG_SHIFT: u16 = 12;
    const SQUARE_MASK: u16 = 0b11_1111;

    /// Pack an arbitrary `(from, to, flag)` into a move.
    pub fn new(from: Square, to: Square, flag: MoveFlag) -> Move {
        Move(
            (from.0 as u16)
                | ((to.0 as u16) << Self::TO_SHIFT)
                | ((flag.code() as u16) << Self::FLAG_SHIFT),
        )
    }

    /// A plain non-capturing, non-special move (no promotion, castle, or ep).
    pub fn quiet(from: Square, to: Square) -> Move {
        Move::new(from, to, MoveFlag::QUIET)
    }

    /// An ordinary capture (not en-passant, not a promotion).
    pub fn capture(from: Square, to: Square) -> Move {
        Move::new(from, to, MoveFlag::CAPTURE)
    }

    /// The origin square.
    pub fn from(self) -> Square {
        Square((self.0 & Self::SQUARE_MASK) as u8)
    }

    /// The destination square.
    pub fn to(self) -> Square {
        Square(((self.0 >> Self::TO_SHIFT) & Self::SQUARE_MASK) as u8)
    }

    /// This move's flag.
    pub fn flag(self) -> MoveFlag {
        MoveFlag((self.0 >> Self::FLAG_SHIFT) as u8)
    }

    /// True if this move captures (see [`MoveFlag::is_capture`]).
    pub fn is_capture(self) -> bool {
        self.flag().is_capture()
    }

    /// True if this move promotes (see [`MoveFlag::is_promotion`]).
    pub fn is_promotion(self) -> bool {
        self.flag().is_promotion()
    }

    /// True if this move is en-passant (see [`MoveFlag::is_en_passant`]).
    pub fn is_en_passant(self) -> bool {
        self.flag().is_en_passant()
    }

    /// True if this move castles (see [`MoveFlag::is_castle`]).
    pub fn is_castle(self) -> bool {
        self.flag().is_castle()
    }

    /// True for the king-side castle specifically (see
    /// [`MoveFlag::is_king_castle`]).
    pub fn is_king_castle(self) -> bool {
        self.flag().is_king_castle()
    }

    /// The promoted-to piece, or `None` (see [`MoveFlag::promotion_piece`]).
    pub fn promotion_piece(self) -> Option<PieceType> {
        self.flag().promotion_piece()
    }
}

impl core::fmt::Display for Move {
    /// Render in UCI long algebraic notation: `from` then `to`, plus a lowercase
    /// promotion letter when promoting — `e2e4`, `e7e8q`. Castling is written as
    /// the king's own move (`e1g1`/`e1c1`), because `from`/`to` already encode
    /// exactly that; UCI has no `O-O`. Captures carry no marker.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}{}", self.from(), self.to())?;
        if let Some(pt) = self.promotion_piece() {
            // Always lowercase in UCI, regardless of the moving side's color.
            let letter = match pt {
                PieceType::Knight => 'n',
                PieceType::Bishop => 'b',
                PieceType::Rook => 'r',
                PieceType::Queen => 'q',
                // Pawns and kings are never promotion targets.
                _ => unreachable!("promotion_piece never yields pawn/king"),
            };
            write!(f, "{letter}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::str::FromStr;

    fn sq(name: &str) -> Square {
        Square::from_str(name).unwrap()
    }

    /// Every flag code in use, for exhaustive sweeps.
    const ALL_FLAGS: [MoveFlag; 14] = [
        MoveFlag::QUIET,
        MoveFlag::DOUBLE_PAWN_PUSH,
        MoveFlag::KING_CASTLE,
        MoveFlag::QUEEN_CASTLE,
        MoveFlag::CAPTURE,
        MoveFlag::EN_PASSANT,
        MoveFlag::KNIGHT_PROMO,
        MoveFlag::BISHOP_PROMO,
        MoveFlag::ROOK_PROMO,
        MoveFlag::QUEEN_PROMO,
        MoveFlag::KNIGHT_PROMO_CAPTURE,
        MoveFlag::BISHOP_PROMO_CAPTURE,
        MoveFlag::ROOK_PROMO_CAPTURE,
        MoveFlag::QUEEN_PROMO_CAPTURE,
    ];

    #[test]
    fn pack_unpack_round_trips_every_square_pair() {
        // from/to must survive packing for all 64×64 pairs, with the flag bits
        // sitting above them and never bleeding into the square fields.
        for from in 0..64u8 {
            for to in 0..64u8 {
                let m = Move::new(Square(from), Square(to), MoveFlag::CAPTURE);
                assert_eq!(m.from(), Square(from));
                assert_eq!(m.to(), Square(to));
                assert_eq!(m.flag(), MoveFlag::CAPTURE);
            }
        }
    }

    #[test]
    fn flag_round_trips_for_every_code() {
        // A fixed square pair carried through every flag: the flag comes back
        // intact and the squares are undisturbed.
        let (from, to) = (sq("e2"), sq("e4"));
        for &flag in &ALL_FLAGS {
            let m = Move::new(from, to, flag);
            assert_eq!(m.flag(), flag);
            assert_eq!(m.from(), from);
            assert_eq!(m.to(), to);
        }
    }

    #[test]
    fn capture_predicate_tracks_the_capture_bit() {
        assert!(!MoveFlag::QUIET.is_capture());
        assert!(!MoveFlag::DOUBLE_PAWN_PUSH.is_capture());
        assert!(!MoveFlag::KNIGHT_PROMO.is_capture());
        assert!(MoveFlag::CAPTURE.is_capture());
        assert!(MoveFlag::EN_PASSANT.is_capture());
        assert!(MoveFlag::QUEEN_PROMO_CAPTURE.is_capture());
    }

    #[test]
    fn promotion_predicate_tracks_the_promo_bit() {
        assert!(!MoveFlag::QUIET.is_promotion());
        assert!(!MoveFlag::CAPTURE.is_promotion());
        assert!(!MoveFlag::EN_PASSANT.is_promotion());
        assert!(MoveFlag::KNIGHT_PROMO.is_promotion());
        assert!(MoveFlag::QUEEN_PROMO_CAPTURE.is_promotion());
    }

    #[test]
    fn en_passant_is_only_code_five() {
        assert!(MoveFlag::EN_PASSANT.is_en_passant());
        // An ordinary capture sets the capture bit but is NOT en-passant — the
        // distinction make/unmake relies on.
        assert!(!MoveFlag::CAPTURE.is_en_passant());
        assert!(!MoveFlag::QUEEN_PROMO_CAPTURE.is_en_passant());
    }

    #[test]
    fn castle_predicates() {
        assert!(MoveFlag::KING_CASTLE.is_castle());
        assert!(MoveFlag::QUEEN_CASTLE.is_castle());
        assert!(MoveFlag::KING_CASTLE.is_king_castle());
        assert!(!MoveFlag::QUEEN_CASTLE.is_king_castle());
        assert!(!MoveFlag::QUIET.is_castle());
    }

    #[test]
    fn promotion_piece_maps_each_code() {
        // Non-capturing promotions.
        assert_eq!(MoveFlag::KNIGHT_PROMO.promotion_piece(), Some(PieceType::Knight));
        assert_eq!(MoveFlag::BISHOP_PROMO.promotion_piece(), Some(PieceType::Bishop));
        assert_eq!(MoveFlag::ROOK_PROMO.promotion_piece(), Some(PieceType::Rook));
        assert_eq!(MoveFlag::QUEEN_PROMO.promotion_piece(), Some(PieceType::Queen));
        // Capturing promotions select the same pieces.
        assert_eq!(
            MoveFlag::KNIGHT_PROMO_CAPTURE.promotion_piece(),
            Some(PieceType::Knight)
        );
        assert_eq!(
            MoveFlag::QUEEN_PROMO_CAPTURE.promotion_piece(),
            Some(PieceType::Queen)
        );
        // Non-promotions yield nothing.
        assert_eq!(MoveFlag::QUIET.promotion_piece(), None);
        assert_eq!(MoveFlag::CAPTURE.promotion_piece(), None);
        assert_eq!(MoveFlag::EN_PASSANT.promotion_piece(), None);
    }

    #[test]
    fn move_forwards_predicates_to_flag() {
        let promo_cap = Move::new(sq("e7"), sq("d8"), MoveFlag::QUEEN_PROMO_CAPTURE);
        assert!(promo_cap.is_capture());
        assert!(promo_cap.is_promotion());
        assert!(!promo_cap.is_en_passant());
        assert_eq!(promo_cap.promotion_piece(), Some(PieceType::Queen));

        let ep = Move::new(sq("e5"), sq("d6"), MoveFlag::EN_PASSANT);
        assert!(ep.is_capture());
        assert!(ep.is_en_passant());
        assert!(!ep.is_promotion());
    }

    #[test]
    fn convenience_constructors() {
        assert_eq!(
            Move::quiet(sq("e2"), sq("e4")),
            Move::new(sq("e2"), sq("e4"), MoveFlag::QUIET)
        );
        assert_eq!(
            Move::capture(sq("e4"), sq("d5")),
            Move::new(sq("e4"), sq("d5"), MoveFlag::CAPTURE)
        );
    }

    #[test]
    fn none_is_distinct_and_degenerate() {
        // The sentinel: from == to == a1, which no legal move ever is.
        assert_eq!(Move::NONE.from(), Square(0));
        assert_eq!(Move::NONE.to(), Square(0));
        assert_eq!(Move::NONE.flag(), MoveFlag::QUIET);
    }

    #[test]
    fn display_is_uci_long_algebraic() {
        // Quiet and capture render identically — UCI has no capture marker.
        assert_eq!(Move::quiet(sq("e2"), sq("e4")).to_string(), "e2e4");
        assert_eq!(Move::capture(sq("e4"), sq("d5")).to_string(), "e4d5");
        // Castling is the king's own move, not O-O.
        assert_eq!(
            Move::new(sq("e1"), sq("g1"), MoveFlag::KING_CASTLE).to_string(),
            "e1g1"
        );
        assert_eq!(
            Move::new(sq("e1"), sq("c1"), MoveFlag::QUEEN_CASTLE).to_string(),
            "e1c1"
        );
        // Promotions append a lowercase piece letter.
        assert_eq!(
            Move::new(sq("e7"), sq("e8"), MoveFlag::QUEEN_PROMO).to_string(),
            "e7e8q"
        );
        assert_eq!(
            Move::new(sq("e7"), sq("e8"), MoveFlag::KNIGHT_PROMO).to_string(),
            "e7e8n"
        );
        // A promotion-capture still shows only the promotion suffix.
        assert_eq!(
            Move::new(sq("e7"), sq("d8"), MoveFlag::ROOK_PROMO_CAPTURE).to_string(),
            "e7d8r"
        );
    }
}
