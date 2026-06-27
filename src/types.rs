//! Core domain types: colors, pieces, squares.
//!
//! Design notes (see docs/decisions/0002-board-bitboards.md):
//! - `Square` is a 0..63 index. Convention: a1 = 0, b1 = 1, ..., h8 = 63
//!   (file-major, rank-ascending). This LSB = a1 convention must stay
//!   consistent everywhere — bitboard bit `i` is the piece on `Square(i)`.

/// Side to move / piece owner.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Color {
    White,
    Black,
}

impl Color {
    /// The opposing color.
    pub fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

/// The kind of a piece, independent of color.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

/// A board square, 0..=63 with a1 = 0 and h8 = 63.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Square(pub u8);

impl Square {
    /// File index 0..=7 (a..h).
    pub fn file(self) -> u8 {
        self.0 & 7
    }

    /// Rank index 0..=7 (1..8).
    pub fn rank(self) -> u8 {
        self.0 >> 3
    }
}
