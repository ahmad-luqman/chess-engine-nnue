//! The board: a chess position.
//!
//! Representation (see docs/decisions/0002-board-bitboards.md): 6 piece-type
//! bitboards + 2 color bitboards give "all squares matching a pattern" in a
//! single AND, while a redundant `mailbox[64]` answers "what's on this one
//! square?" in O(1). The two views are kept in lockstep by `put_piece` /
//! `remove_piece` — that sync discipline is where make/unmake bugs hide later,
//! which is why perft (issue #17) exists to catch them.

use crate::bitboard::Bitboard;
use crate::moves::{Move, MoveFlag};
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

/// The information [`Board::make_move`] must squirrel away to let
/// [`Board::unmake_move`] restore the position exactly.
///
/// It holds only the *irreversible* state — what a move destroys and cannot
/// recompute: the captured piece (if any), and the castling rights, en-passant
/// square, and halfmove clock as they were *before* the move. Everything else
/// (the moving piece, the side to move, the move number) is recoverable from the
/// move itself, so it is not stored. (A Zobrist key will join this struct when
/// the transposition table arrives in Phase 2.)
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Undo {
    captured: Option<Piece>,
    castling: CastlingRights,
    ep_square: Option<Square>,
    halfmove_clock: u16,
}

/// The rook's `(from, to)` squares for a castling move, derived from the king's
/// move. King-side brings the rook from the h-file to the f-file; queen-side
/// from the a-file to the d-file; both on the king's own rank.
fn castle_rook_squares(mv: Move) -> (Square, Square) {
    let rank = mv.from().rank();
    if mv.is_king_castle() {
        (Square::from_file_rank(7, rank), Square::from_file_rank(5, rank))
    } else {
        (Square::from_file_rank(0, rank), Square::from_file_rank(3, rank))
    }
}

/// The castling-rights bits that touching `sq` (as a move's origin or
/// destination) clears: a king's home square clears both its rights, a rook's
/// home square clears that side's right, every other square clears nothing.
/// Applied to both endpoints of a move, this handles king moves, rook moves, and
/// rook *captures* uniformly.
fn castling_loss_mask(sq: Square) -> u8 {
    match (sq.file(), sq.rank()) {
        (4, 0) => CastlingRights::WHITE_KING | CastlingRights::WHITE_QUEEN, // e1
        (0, 0) => CastlingRights::WHITE_QUEEN,                              // a1
        (7, 0) => CastlingRights::WHITE_KING,                               // h1
        (4, 7) => CastlingRights::BLACK_KING | CastlingRights::BLACK_QUEEN, // e8
        (0, 7) => CastlingRights::BLACK_QUEEN,                              // a8
        (7, 7) => CastlingRights::BLACK_KING,                               // h8
        _ => 0,
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

    /// Apply `mv` to the position and return the [`Undo`] needed to reverse it.
    ///
    /// `mv` is assumed legal (the generator's job). This updates *everything*:
    /// the moving piece (with promotion, castled rook, and en-passant victim),
    /// side to move, castling rights, en-passant square, and the clocks — unlike
    /// the legality-only application the generator used internally before. The
    /// returned `Undo` captures exactly the irreversible state so that
    /// [`unmake_move`](Self::unmake_move) restores the position bit-for-bit.
    ///
    /// The returned value *is* the undo stack: callers (search, perft) keep it
    /// across recursion and hand it back. Nothing is stored on the `Board`,
    /// which keeps it a plain value with no hidden history.
    pub fn make_move(&mut self, mv: Move) -> Undo {
        let us = self.side_to_move;
        let from = mv.from();
        let to = mv.to();

        let prev = Undo {
            captured: None, // overwritten below if this move captures
            castling: self.castling,
            ep_square: self.ep_square,
            halfmove_clock: self.halfmove_clock,
        };

        let moving = self.remove_piece(from).expect("a move originates on an occupied square");
        let is_pawn = moving.piece_type == PieceType::Pawn;

        // Remove the captured piece, if any. En-passant's victim is beside the
        // mover (destination file, origin rank), never on `to`.
        let captured = if mv.is_en_passant() {
            let victim = Square::from_file_rank(to.file(), from.rank());
            self.remove_piece(victim)
        } else {
            self.remove_piece(to)
        };

        // Place the moving piece, swapping in the promoted piece if promoting.
        let placed = match mv.promotion_piece() {
            Some(piece_type) => Piece { color: us, piece_type },
            None => moving,
        };
        self.put_piece(to, placed);

        // Relocate the rook on a castle (the king's own move is already done).
        if mv.is_castle() {
            let (rook_from, rook_to) = castle_rook_squares(mv);
            let rook = self.remove_piece(rook_from).expect("a castling rook is present");
            self.put_piece(rook_to, rook);
        }

        // A double pawn push (and nothing else) sets the en-passant square — the
        // square it skipped over, midway between origin and destination.
        self.ep_square = if mv.flag() == MoveFlag::DOUBLE_PAWN_PUSH {
            Some(Square::from_file_rank(from.file(), (from.rank() + to.rank()) / 2))
        } else {
            None
        };

        // Castling rights fall away when a king or rook leaves its home square,
        // or when a rook is captured on its home square — covered by masking on
        // both `from` and `to`.
        self.castling =
            CastlingRights(self.castling.0 & !(castling_loss_mask(from) | castling_loss_mask(to)));

        // The fifty-move clock resets on any pawn move or capture, else ticks up.
        self.halfmove_clock = if is_pawn || captured.is_some() {
            0
        } else {
            self.halfmove_clock + 1
        };

        // Hand over to the opponent; the move number counts completed Black moves.
        self.side_to_move = us.flip();
        if us == Color::Black {
            self.fullmove_number += 1;
        }

        Undo { captured, ..prev }
    }

    /// Reverse a [`make_move`](Self::make_move), restoring the position exactly.
    /// `mv` and the `undo` returned by the matching `make_move` are both required.
    pub fn unmake_move(&mut self, mv: Move, undo: Undo) {
        // Flip back first so `us` is the side that originally moved.
        let us = self.side_to_move.flip();
        self.side_to_move = us;
        if us == Color::Black {
            self.fullmove_number -= 1;
        }

        let from = mv.from();
        let to = mv.to();

        // Send the castled rook home (before touching the king square).
        if mv.is_castle() {
            let (rook_from, rook_to) = castle_rook_squares(mv);
            let rook = self.remove_piece(rook_to).expect("the castled rook to revert");
            self.put_piece(rook_from, rook);
        }

        // Lift the moved piece off `to`; a promotion becomes a pawn again,
        // otherwise the very piece we removed is what goes home.
        let placed = self.remove_piece(to).expect("the moved piece to revert");
        let original = if mv.is_promotion() {
            Piece { color: us, piece_type: PieceType::Pawn }
        } else {
            placed
        };
        self.put_piece(from, original);

        // Put back anything we captured, on its real square (ep victim aside).
        if let Some(captured) = undo.captured {
            let square = if mv.is_en_passant() {
                Square::from_file_rank(to.file(), from.rank())
            } else {
                to
            };
            self.put_piece(square, captured);
        }

        self.castling = undo.castling;
        self.ep_square = undo.ep_square;
        self.halfmove_clock = undo.halfmove_clock;
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

    // ── make / unmake (issue #16) ──────────────────────────────────────────

    use crate::movegen::generate_legal;
    use core::str::FromStr;

    const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    const POS3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
    const POS4: &str = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
    const POS5: &str = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1";
    // A custom en-passant position — the five standard FENs all have ep "-", so
    // without this the roundtrip test would never exercise ep restoration.
    const EP_POS: &str = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1";

    fn board(fen: &str) -> Board {
        Board::from_str(fen).unwrap()
    }

    #[test]
    fn make_unmake_restores_position_exactly() {
        // For every legal move of each position, make then unmake must return the
        // board bit-for-bit (Board derives Eq). Iterating *all* legal moves is
        // what guarantees coverage of castling, rook-capture, promotion, and ep —
        // a quiet-only sample would pass even if state restoration were broken.
        for fen in [STARTPOS, KIWIPETE, POS3, POS4, POS5, EP_POS] {
            let original = board(fen);
            for mv in generate_legal(&original) {
                let mut b = original.clone();
                let undo = b.make_move(mv);
                assert_ne!(b, original, "{mv} in {fen} changed nothing");
                b.unmake_move(mv, undo);
                assert_eq!(b, original, "{mv} in {fen} did not round-trip");
            }
        }
    }

    /// Count leaf nodes at depth `depth` by making and unmaking every legal move.
    /// This exercises the whole make/unmake machinery recursively; the dedicated
    /// perft module and full depth-5/6 table are issue #17.
    fn perft(board: &mut Board, depth: u32) -> u64 {
        let moves = generate_legal(board);
        if depth == 1 {
            return moves.len() as u64;
        }
        let mut nodes = 0;
        for mv in moves {
            let undo = board.make_move(mv);
            nodes += perft(board, depth - 1);
            board.unmake_move(mv, undo);
        }
        nodes
    }

    #[test]
    fn perft_depth3_matches_standard_positions() {
        // Depth 3 across all five positions: each stresses a different mix
        // (ep + checks, promotions, castling, pins), and at depth 3 only correct
        // make/unmake produces these node counts. Deeper perft is gated for #17,
        // since debug-build depth 4+ is slow.
        for (fen, expected) in [
            (STARTPOS, 8902u64),
            (KIWIPETE, 97862),
            (POS3, 2812),
            (POS4, 9467),
            (POS5, 62379),
        ] {
            assert_eq!(perft(&mut board(fen), 3), expected, "perft(3) wrong for {fen}");
        }
    }
}
