//! The board: a chess position.
//!
//! Representation (see docs/decisions/0002-board-bitboards.md): 6 piece-type
//! bitboards + 2 color bitboards give "all squares matching a pattern" in a
//! single AND, while a redundant `mailbox[64]` answers "what's on this one
//! square?" in O(1). The two views are kept in lockstep by `put_piece` /
//! `remove_piece` — that sync discipline is where make/unmake bugs hide later,
//! which is why perft (issue #17) exists to catch them.

use crate::bitboard::Bitboard;
use crate::types::{Color, Piece, PieceType, Square};

/// Castling availability as four independent flags packed into a `u8`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct CastlingRights(pub u8);

impl CastlingRights {
    pub const WHITE_KING: u8 = 0b0001;
    pub const WHITE_QUEEN: u8 = 0b0010;
    pub const BLACK_KING: u8 = 0b0100;
    pub const BLACK_QUEEN: u8 = 0b1000;

    pub const NONE: CastlingRights = CastlingRights(0);
    pub const ALL: CastlingRights = CastlingRights(0b1111);

    /// True if the given flag (e.g. `CastlingRights::WHITE_KING`) is set.
    pub fn has(self, flag: u8) -> bool {
        self.0 & flag != 0
    }
}

/// A full chess position.
///
/// `piece_bb` / `color_bb` are the bitboard view; `mailbox` is the per-square
/// view. Invariant: a square has a bit set in exactly one `piece_bb` and one
/// `color_bb` iff `mailbox[sq] == Some(that piece)`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Board {
    piece_bb: [Bitboard; 6],
    color_bb: [Bitboard; 2],
    mailbox: [Option<Piece>; 64],
    pub side_to_move: Color,
    pub castling: CastlingRights,
    pub ep_square: Option<Square>,
    pub halfmove_clock: u16,
    pub fullmove_number: u16,
}

impl Board {
    /// An empty board with White to move and no rights — the blank slate a FEN
    /// parser (issue #11) fills in.
    pub fn empty() -> Board {
        Board {
            piece_bb: [Bitboard::EMPTY; 6],
            color_bb: [Bitboard::EMPTY; 2],
            mailbox: [None; 64],
            side_to_move: Color::White,
            castling: CastlingRights::NONE,
            ep_square: None,
            halfmove_clock: 0,
            fullmove_number: 1,
        }
    }

    /// All squares holding a piece of the given kind (either color).
    pub fn pieces(&self, pt: PieceType) -> Bitboard {
        self.piece_bb[pt.index()]
    }

    /// All squares holding a piece of the given color (any kind).
    pub fn color(&self, c: Color) -> Bitboard {
        self.color_bb[c.index()]
    }

    /// All occupied squares — the union of both colors.
    pub fn occupied(&self) -> Bitboard {
        self.color_bb[0].union(self.color_bb[1])
    }

    /// The piece on `sq`, if any (O(1) mailbox lookup).
    pub fn piece_on(&self, sq: Square) -> Option<Piece> {
        self.mailbox[sq.0 as usize]
    }

    /// Place `piece` on an empty square, updating both views.
    pub fn put_piece(&mut self, sq: Square, piece: Piece) {
        debug_assert!(self.piece_on(sq).is_none(), "put_piece onto occupied square");
        self.piece_bb[piece.piece_type.index()] = self.piece_bb[piece.piece_type.index()].with(sq);
        self.color_bb[piece.color.index()] = self.color_bb[piece.color.index()].with(sq);
        self.mailbox[sq.0 as usize] = Some(piece);
    }

    /// Remove and return the piece on `sq` (if any), updating both views.
    pub fn remove_piece(&mut self, sq: Square) -> Option<Piece> {
        let piece = self.mailbox[sq.0 as usize]?;
        self.piece_bb[piece.piece_type.index()] =
            self.piece_bb[piece.piece_type.index()].without(sq);
        self.color_bb[piece.color.index()] = self.color_bb[piece.color.index()].without(sq);
        self.mailbox[sq.0 as usize] = None;
        Some(piece)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piece(color: Color, piece_type: PieceType) -> Piece {
        Piece { color, piece_type }
    }

    #[test]
    fn put_and_remove_keep_both_views_in_sync() {
        let mut b = Board::empty();
        let e4 = Square(28);
        let wn = piece(Color::White, PieceType::Knight);

        b.put_piece(e4, wn);
        assert_eq!(b.piece_on(e4), Some(wn));
        assert!(b.pieces(PieceType::Knight).contains(e4));
        assert!(b.color(Color::White).contains(e4));
        assert!(!b.color(Color::Black).contains(e4));
        assert!(b.occupied().contains(e4));
        assert_eq!(b.occupied().count(), 1);

        let removed = b.remove_piece(e4);
        assert_eq!(removed, Some(wn));
        assert_eq!(b.piece_on(e4), None);
        assert!(b.occupied().is_empty());
    }

    #[test]
    fn occupancy_is_union_of_colors() {
        let mut b = Board::empty();
        b.put_piece(Square(0), piece(Color::White, PieceType::Rook)); // a1
        b.put_piece(Square(63), piece(Color::Black, PieceType::King)); // h8
        assert_eq!(b.occupied().count(), 2);
        assert_eq!(b.color(Color::White).count(), 1);
        assert_eq!(b.color(Color::Black).count(), 1);
    }

    #[test]
    fn remove_from_empty_square_is_none() {
        let mut b = Board::empty();
        assert_eq!(b.remove_piece(Square(20)), None);
    }

    #[test]
    fn castling_flags() {
        let r = CastlingRights::ALL;
        assert!(r.has(CastlingRights::WHITE_KING));
        assert!(r.has(CastlingRights::BLACK_QUEEN));
        assert!(!CastlingRights::NONE.has(CastlingRights::WHITE_KING));
    }
}
