//! FEN: Forsyth–Edwards Notation, the standard one-line text encoding of a
//! position.
//!
//! Why this matters now: a move generator is worthless until you can prove it
//! against *known* positions (startpos, Kiwipete, the standard perft suite).
//! FEN is the input format those positions ship in, so it's the gate before
//! movegen — you can't perft-test what you can't load.
//!
//! A FEN string is six space-separated fields:
//!
//! ```text
//!   rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
//!   └──────── 1 ────────┘ 2 └─ 3 ─┘ 4 5 6
//! ```
//!
//! 1. piece placement, **rank 8 first** down to rank 1, files a→h within a rank;
//! 2. side to move (`w`/`b`); 3. castling rights (`KQkq` subset or `-`);
//! 4. en-passant target square or `-`; 5. halfmove clock; 6. fullmove number.
//!
//! The placement field reads top-of-board-first, but our square index runs
//! a1 = 0 … h8 = 63 (bottom-first). That inversion — FEN token `t` is board rank
//! `7 - t` — is the one easy place to get the board upside-down, so it gets an
//! explicit comment at the loop below.

use crate::board::{Board, CastlingRights};
use crate::types::{Color, Piece, PieceType, Square};
use core::str::FromStr;

/// Why a FEN string could not be parsed. Like [`crate::types::ParseSquareError`],
/// every malformed input maps to a specific variant rather than a panic — FEN
/// comes from files and GUIs, i.e. untrusted input.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ParseFenError {
    /// Not exactly six space-separated fields.
    WrongFieldCount,
    /// Placement field malformed: bad piece letter, a rank not summing to 8
    /// files, or not exactly 8 ranks.
    BadPiecePlacement,
    /// Side-to-move field was not `w` or `b`.
    BadSideToMove,
    /// Castling field had a character outside `KQkq` (and was not `-`).
    BadCastling,
    /// En-passant field was neither `-` nor a valid square.
    BadEnPassant,
    /// Halfmove clock was not a non-negative integer.
    BadHalfmove,
    /// Fullmove number was not a non-negative integer.
    BadFullmove,
}

/// Map a FEN piece letter to a colored piece. Uppercase = White, lowercase =
/// Black; `pnbrqk` are the six kinds. Returns `None` for any other byte.
fn piece_from_fen(byte: u8) -> Option<Piece> {
    let color = if byte.is_ascii_uppercase() { Color::White } else { Color::Black };
    let piece_type = match byte.to_ascii_lowercase() {
        b'p' => PieceType::Pawn,
        b'n' => PieceType::Knight,
        b'b' => PieceType::Bishop,
        b'r' => PieceType::Rook,
        b'q' => PieceType::Queen,
        b'k' => PieceType::King,
        _ => return None,
    };
    Some(Piece { color, piece_type })
}

/// The FEN letter for a piece — the inverse of [`piece_from_fen`].
fn fen_from_piece(piece: Piece) -> char {
    let letter = match piece.piece_type {
        PieceType::Pawn => 'p',
        PieceType::Knight => 'n',
        PieceType::Bishop => 'b',
        PieceType::Rook => 'r',
        PieceType::Queen => 'q',
        PieceType::King => 'k',
    };
    // White pieces are uppercase; Black keep the lowercase letter above.
    match piece.color {
        Color::White => letter.to_ascii_uppercase(),
        Color::Black => letter,
    }
}

/// Parse the piece-placement field (field 1) onto an empty board.
fn parse_placement(field: &str, board: &mut Board) -> Result<(), ParseFenError> {
    let ranks: Vec<&str> = field.split('/').collect();
    if ranks.len() != 8 {
        return Err(ParseFenError::BadPiecePlacement);
    }

    for (token_index, rank_str) in ranks.iter().enumerate() {
        // FEN lists rank 8 first; our ranks count up from 0 = rank 1. So the
        // first token (index 0) is board rank 7, the last (index 7) is rank 0.
        let rank = 7 - token_index as u8;
        let mut file: u8 = 0;
        for byte in rank_str.bytes() {
            if byte.is_ascii_digit() {
                // A digit is a run of empty squares; advance the file cursor.
                file += byte - b'0';
            } else {
                let piece = piece_from_fen(byte).ok_or(ParseFenError::BadPiecePlacement)?;
                if file >= 8 {
                    return Err(ParseFenError::BadPiecePlacement);
                }
                board.put_piece(Square::from_file_rank(file, rank), piece);
                file += 1;
            }
        }
        // Every rank must describe exactly 8 files — no more, no fewer.
        if file != 8 {
            return Err(ParseFenError::BadPiecePlacement);
        }
    }
    Ok(())
}

/// Parse the castling field (field 3) into a [`CastlingRights`] bitset.
fn parse_castling(field: &str) -> Result<CastlingRights, ParseFenError> {
    if field == "-" {
        return Ok(CastlingRights::NONE);
    }
    let mut bits = 0u8;
    for byte in field.bytes() {
        bits |= match byte {
            b'K' => CastlingRights::WHITE_KING,
            b'Q' => CastlingRights::WHITE_QUEEN,
            b'k' => CastlingRights::BLACK_KING,
            b'q' => CastlingRights::BLACK_QUEEN,
            _ => return Err(ParseFenError::BadCastling),
        };
    }
    Ok(CastlingRights(bits))
}

impl FromStr for Board {
    type Err = ParseFenError;

    /// Parse a full six-field FEN string into a [`Board`].
    fn from_str(s: &str) -> Result<Board, ParseFenError> {
        let fields: Vec<&str> = s.split(' ').collect();
        if fields.len() != 6 {
            return Err(ParseFenError::WrongFieldCount);
        }

        let mut board = Board::empty();
        parse_placement(fields[0], &mut board)?;

        board.side_to_move = match fields[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err(ParseFenError::BadSideToMove),
        };

        board.castling = parse_castling(fields[2])?;

        board.ep_square = match fields[3] {
            "-" => None,
            sq => Some(Square::from_str(sq).map_err(|_| ParseFenError::BadEnPassant)?),
        };

        board.halfmove_clock = fields[4].parse().map_err(|_| ParseFenError::BadHalfmove)?;
        board.fullmove_number = fields[5].parse().map_err(|_| ParseFenError::BadFullmove)?;

        // `put_piece` and the field assignments above are hash-agnostic, so seed
        // the Zobrist key from scratch now that the position is fully built.
        board.hash = crate::zobrist::compute(&board);

        Ok(board)
    }
}

impl Board {
    /// Serialize this position back to a FEN string — the exact inverse of
    /// [`FromStr`], so `Board::from_str(&b.to_fen())` reproduces `b`.
    pub fn to_fen(&self) -> String {
        let mut fen = String::new();

        // Field 1: placement, rank 8 down to rank 1 (mirror of the parse order).
        for rank in (0..8).rev() {
            let mut empties = 0u8;
            for file in 0..8 {
                match self.piece_on(Square::from_file_rank(file, rank)) {
                    Some(piece) => {
                        // Flush any pending run of empty squares as a digit first.
                        if empties > 0 {
                            fen.push((b'0' + empties) as char);
                            empties = 0;
                        }
                        fen.push(fen_from_piece(piece));
                    }
                    None => empties += 1,
                }
            }
            if empties > 0 {
                fen.push((b'0' + empties) as char);
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        // Field 2: side to move.
        fen.push(' ');
        fen.push(match self.side_to_move {
            Color::White => 'w',
            Color::Black => 'b',
        });

        // Field 3: castling rights, always in KQkq order, or '-' if none.
        fen.push(' ');
        if self.castling == CastlingRights::NONE {
            fen.push('-');
        } else {
            for (flag, letter) in [
                (CastlingRights::WHITE_KING, 'K'),
                (CastlingRights::WHITE_QUEEN, 'Q'),
                (CastlingRights::BLACK_KING, 'k'),
                (CastlingRights::BLACK_QUEEN, 'q'),
            ] {
                if self.castling.has(flag) {
                    fen.push(letter);
                }
            }
        }

        // Field 4: en-passant target square, or '-'.
        fen.push(' ');
        match self.ep_square {
            Some(sq) => fen.push_str(&sq.to_string()),
            None => fen.push('-'),
        }

        // Fields 5 & 6: the two move counters.
        fen.push_str(&format!(" {} {}", self.halfmove_clock, self.fullmove_number));
        fen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    // Kiwipete — the canonical perft position that exercises castling, ep, pins.
    const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    #[test]
    fn startpos_round_trips() {
        let board = Board::from_str(STARTPOS).unwrap();
        assert_eq!(board.to_fen(), STARTPOS);
    }

    #[test]
    fn kiwipete_round_trips() {
        let board = Board::from_str(KIWIPETE).unwrap();
        assert_eq!(board.to_fen(), KIWIPETE);
    }

    #[test]
    fn parses_startpos_fields() {
        let board = Board::from_str(STARTPOS).unwrap();
        // White to move, all castling, no ep, counters at their initial values.
        assert_eq!(board.side_to_move, Color::White);
        assert_eq!(board.castling, CastlingRights::ALL);
        assert_eq!(board.ep_square, None);
        assert_eq!(board.halfmove_clock, 0);
        assert_eq!(board.fullmove_number, 1);
        // a1 is a white rook (index 0), e1 (Square 4) a white king, e8 a black king.
        assert_eq!(
            board.piece_on(Square(0)),
            Some(Piece { color: Color::White, piece_type: PieceType::Rook })
        );
        assert_eq!(
            board.piece_on(Square::from_str("e1").unwrap()),
            Some(Piece { color: Color::White, piece_type: PieceType::King })
        );
        assert_eq!(
            board.piece_on(Square::from_str("e8").unwrap()),
            Some(Piece { color: Color::Black, piece_type: PieceType::King })
        );
    }

    #[test]
    fn en_passant_square_round_trips() {
        // After 1. e4, Black to move, ep target e3. Parse and re-emit unchanged.
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
        let board = Board::from_str(fen).unwrap();
        assert_eq!(board.ep_square, Some(Square::from_str("e3").unwrap()));
        assert_eq!(board.side_to_move, Color::Black);
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn no_castling_rights_round_trips() {
        let fen = "8/8/8/4k3/8/8/4K3/8 w - - 0 1";
        let board = Board::from_str(fen).unwrap();
        assert_eq!(board.castling, CastlingRights::NONE);
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn rejects_malformed_input() {
        use ParseFenError::*;
        // Five fields instead of six.
        assert_eq!(Board::from_str("8/8/8/8/8/8/8/8 w - -"), Err(WrongFieldCount));
        // Seven ranks in the placement field.
        assert_eq!(Board::from_str("8/8/8/8/8/8/8 w - - 0 1"), Err(BadPiecePlacement));
        // A rank that sums to 9 files (8 + 1).
        assert_eq!(Board::from_str("8/8/8/8/8/8/8/8P w - - 0 1"), Err(BadPiecePlacement));
        // 'x' is not a piece letter.
        assert_eq!(
            Board::from_str("xnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            Err(BadPiecePlacement)
        );
        // Side to move neither w nor b.
        assert_eq!(Board::from_str("8/8/8/8/8/8/8/8 x - - 0 1"), Err(BadSideToMove));
        // Stray castling character.
        assert_eq!(Board::from_str("8/8/8/8/8/8/8/8 w Z - 0 1"), Err(BadCastling));
        // En-passant field that is not a square.
        assert_eq!(Board::from_str("8/8/8/8/8/8/8/8 w - z9 0 1"), Err(BadEnPassant));
        // Non-numeric halfmove clock.
        assert_eq!(Board::from_str("8/8/8/8/8/8/8/8 w - - x 1"), Err(BadHalfmove));
    }
}
