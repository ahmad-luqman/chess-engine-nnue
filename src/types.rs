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

    /// Index 0 (White) or 1 (Black), for indexing per-color arrays.
    pub fn index(self) -> usize {
        self as usize
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

impl PieceType {
    /// Index 0..=5 (Pawn..King), for indexing per-piece-type arrays.
    pub fn index(self) -> usize {
        self as usize
    }
}

/// A colored piece: the combination of a color and a kind.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Piece {
    pub color: Color,
    pub piece_type: PieceType,
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

    /// Build a square from file (0..=7, a..h) and rank (0..=7, 1..8).
    ///
    /// The index is `rank * 8 + file`, the exact inverse of `file()`/`rank()`:
    /// the low 3 bits are the file, the high 3 bits the rank. Debug-asserts the
    /// inputs are in range — callers inside the engine are trusted (the
    /// untrusted boundary is `from_str`, which validates instead).
    pub fn from_file_rank(file: u8, rank: u8) -> Square {
        debug_assert!(file < 8 && rank < 8, "file/rank out of range");
        Square((rank << 3) | file)
    }
}

/// Why an algebraic square string (e.g. "e4") could not be parsed.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ParseSquareError {
    /// Not exactly two characters (one file letter + one rank digit).
    WrongLength,
    /// File letter outside 'a'..='h'.
    BadFile,
    /// Rank digit outside '1'..='8'.
    BadRank,
}

impl core::fmt::Display for Square {
    /// Render as algebraic coordinates, e.g. `Square(28)` -> "e4".
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let file = (b'a' + self.file()) as char;
        let rank = (b'1' + self.rank()) as char;
        write!(f, "{file}{rank}")
    }
}

impl core::str::FromStr for Square {
    type Err = ParseSquareError;

    /// Parse algebraic coordinates like "e4". Input here is untrusted (it comes
    /// from FEN/UCI), so every malformed case returns an error rather than
    /// panicking.
    fn from_str(s: &str) -> Result<Square, ParseSquareError> {
        let bytes = s.as_bytes();
        if bytes.len() != 2 {
            return Err(ParseSquareError::WrongLength);
        }
        let file = bytes[0];
        let rank = bytes[1];
        if !(b'a'..=b'h').contains(&file) {
            return Err(ParseSquareError::BadFile);
        }
        if !(b'1'..=b'8').contains(&rank) {
            return Err(ParseSquareError::BadRank);
        }
        Ok(Square::from_file_rank(file - b'a', rank - b'1'))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::str::FromStr;

    #[test]
    fn file_rank_round_trip() {
        // Every square decodes to (file, rank) and re-encodes to itself.
        for i in 0..64u8 {
            let sq = Square(i);
            assert_eq!(Square::from_file_rank(sq.file(), sq.rank()), sq);
        }
    }

    #[test]
    fn known_squares() {
        assert_eq!(Square::from_file_rank(0, 0), Square(0)); // a1
        assert_eq!(Square::from_file_rank(4, 3), Square(28)); // e4
        assert_eq!(Square::from_file_rank(7, 7), Square(63)); // h8
        assert_eq!(Square(0).to_string(), "a1");
        assert_eq!(Square(28).to_string(), "e4");
        assert_eq!(Square(63).to_string(), "h8");
    }

    #[test]
    fn parse_round_trips_display() {
        for i in 0..64u8 {
            let sq = Square(i);
            assert_eq!(Square::from_str(&sq.to_string()), Ok(sq));
        }
    }

    #[test]
    fn parse_rejects_bad_input() {
        use ParseSquareError::*;
        assert_eq!(Square::from_str(""), Err(WrongLength));
        assert_eq!(Square::from_str("e"), Err(WrongLength));
        assert_eq!(Square::from_str("e4 "), Err(WrongLength));
        assert_eq!(Square::from_str("z4"), Err(BadFile));
        assert_eq!(Square::from_str("e9"), Err(BadRank));
        assert_eq!(Square::from_str("e0"), Err(BadRank));
    }
}
