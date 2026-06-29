//! The board: a chess position.
//!
//! Representation (see docs/decisions/0002-board-bitboards.md): 6 piece-type
//! bitboards + 2 color bitboards give "all squares matching a pattern" in a
//! single AND, while a redundant `mailbox[64]` answers "what's on this one
//! square?" in O(1). The two views are kept in lockstep by `put_piece` /
//! `remove_piece` — that sync discipline is where make/unmake bugs hide later,
//! which is why perft (issue #17) exists to catch them.

use crate::bitboard::Bitboard;
use crate::movegen::pawn_attacks;
use crate::moves::{Move, MoveFlag};
use crate::types::{Color, Piece, PieceType, Square};
use crate::zobrist;

/// The Zobrist key for `piece` standing on `sq` — the per-feature constant that
/// gets XORed in when the piece arrives and out when it leaves.
fn piece_key(piece: Piece, sq: Square) -> u64 {
    zobrist::KEYS.piece[piece.color.index()][piece.piece_type.index()][sq.0 as usize]
}

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

/// The set of `(piece, square)` input features a single move toggles — exactly
/// what an NNUE accumulator update needs to add/subtract a few weight columns
/// instead of recomputing from scratch (see `docs/04-nnue.md`).
///
/// `make_move` already enumerates every feature it touches for the incremental
/// Zobrist key; this records the same toggles as two short lists. **Nothing reads
/// it yet** — it is plumbing for the NNUE inference work (#46), which will consume
/// [`added`](Self::added)/[`removed`](Self::removed) at make time and replay them
/// in reverse at unmake.
///
/// Capacity is a fixed 2 per list, which is the exact maximum across all move
/// kinds: a quiet move toggles 1+1, a capture or en-passant 2 removed + 1 added,
/// a promotion (with or without capture) 1–2 removed + 1 added, and castling 2+2
/// (king and rook). Unused slots hold a sentinel so the struct stays `Copy + Eq`
/// for [`Undo`]'s derives; the count gates every read, so the sentinel is never
/// observed.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct DirtyPiece {
    added: [(Piece, Square); 2],
    removed: [(Piece, Square); 2],
    num_added: u8,
    num_removed: u8,
}

impl DirtyPiece {
    /// Placeholder filling unused array slots; `num_*` keeps it out of every read.
    const SENTINEL: (Piece, Square) =
        (Piece { color: Color::White, piece_type: PieceType::Pawn }, Square(0));

    /// A delta that toggles nothing — the starting point a move fills in, and the
    /// final value for a null move (which touches no piece features).
    fn empty() -> DirtyPiece {
        DirtyPiece {
            added: [DirtyPiece::SENTINEL; 2],
            removed: [DirtyPiece::SENTINEL; 2],
            num_added: 0,
            num_removed: 0,
        }
    }

    /// Record that `piece` arrived on `sq` (a weight column to add).
    fn add(&mut self, piece: Piece, sq: Square) {
        debug_assert!((self.num_added as usize) < self.added.len(), "DirtyPiece added overflow");
        self.added[self.num_added as usize] = (piece, sq);
        self.num_added += 1;
    }

    /// Record that `piece` left `sq` (a weight column to subtract).
    fn remove(&mut self, piece: Piece, sq: Square) {
        debug_assert!(
            (self.num_removed as usize) < self.removed.len(),
            "DirtyPiece removed overflow"
        );
        self.removed[self.num_removed as usize] = (piece, sq);
        self.num_removed += 1;
    }

    /// The features this move added — columns the accumulator gains.
    pub fn added(&self) -> &[(Piece, Square)] {
        &self.added[..self.num_added as usize]
    }

    /// The features this move removed — columns the accumulator loses.
    pub fn removed(&self) -> &[(Piece, Square)] {
        &self.removed[..self.num_removed as usize]
    }
}

/// The information [`Board::make_move`] must squirrel away to let
/// [`Board::unmake_move`] restore the position exactly.
///
/// It holds only the *irreversible* state — what a move destroys and cannot
/// recompute: the captured piece (if any), and the castling rights, en-passant
/// square, and halfmove clock as they were *before* the move. Everything else
/// (the moving piece, the side to move, the move number) is recoverable from the
/// move itself, so it is not stored.
///
/// The pre-move Zobrist `hash` rides along too: rather than replay the move's
/// XORs in reverse, `unmake_move` simply restores this snapshot in O(1) — the
/// struct already exists to carry exactly this kind of un-recomputable state.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Undo {
    captured: Option<Piece>,
    castling: CastlingRights,
    ep_square: Option<Square>,
    halfmove_clock: u16,
    hash: u64,
    /// The `(piece, square)` features this move toggled — see [`DirtyPiece`]. Rides
    /// along on the undo record (Stockfish's `StateInfo` model) so the NNUE
    /// accumulator update (#46) can read it at make *and* unmake without changing
    /// any call site. Empty for a null move.
    pub dirty: DirtyPiece,
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
    /// Incrementally maintained Zobrist key (see `crate::zobrist`). Updated in
    /// `make_move`, snapshot-restored in `unmake_move`, and seeded from scratch
    /// by the FEN parser. Cheap collision-resistant identity for the TT (#24)
    /// and repetition detection (#28).
    pub hash: u64,
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
            hash: 0,
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

        // Snapshot the irreversible pre-move state; the `Undo` is assembled from
        // these once, at the end, to avoid building a throwaway struct mid-flight.
        let prev_castling = self.castling;
        let prev_ep_square = self.ep_square;
        let prev_halfmove_clock = self.halfmove_clock;
        let prev_hash = self.hash; // pre-move snapshot; unmake restores it verbatim

        // Record the toggled features for the NNUE accumulator, in lockstep with
        // the Zobrist XORs that follow — same removes, same adds.
        let mut dirty = DirtyPiece::empty();

        // Maintain the Zobrist key incrementally: every board feature this move
        // touches is XORed out of `hash` as it leaves and in as it arrives.
        let mut hash = self.hash;
        // Drop the old en-passant contribution first (it reads the *current* ep
        // square and side to move); the new one is added at the end.
        hash ^= self.ep_zobrist();

        let moving = self.remove_piece(from).expect("a move originates on an occupied square");
        hash ^= piece_key(moving, from);
        dirty.remove(moving, from);
        let is_pawn = moving.piece_type == PieceType::Pawn;

        // Remove the captured piece, if any. En-passant's victim is beside the
        // mover (destination file, origin rank), never on `to`.
        let captured_square =
            if mv.is_en_passant() { Square::from_file_rank(to.file(), from.rank()) } else { to };
        let captured = self.remove_piece(captured_square);
        if let Some(victim) = captured {
            hash ^= piece_key(victim, captured_square);
            dirty.remove(victim, captured_square);
        }

        // Place the moving piece, swapping in the promoted piece if promoting.
        let placed = match mv.promotion_piece() {
            Some(piece_type) => Piece { color: us, piece_type },
            None => moving,
        };
        self.put_piece(to, placed);
        hash ^= piece_key(placed, to);
        dirty.add(placed, to);

        // Relocate the rook on a castle (the king's own move is already done).
        if mv.is_castle() {
            let (rook_from, rook_to) = castle_rook_squares(mv);
            let rook = self.remove_piece(rook_from).expect("a castling rook is present");
            hash ^= piece_key(rook, rook_from);
            dirty.remove(rook, rook_from);
            self.put_piece(rook_to, rook);
            hash ^= piece_key(rook, rook_to);
            dirty.add(rook, rook_to);
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
        let old_castling = self.castling;
        self.castling =
            CastlingRights(self.castling.0 & !(castling_loss_mask(from) | castling_loss_mask(to)));
        // One XOR-out/XOR-in over the 16-entry table; a no-op when rights are
        // unchanged (the two entries are identical, so they cancel).
        hash ^= zobrist::KEYS.castling[old_castling.0 as usize]
            ^ zobrist::KEYS.castling[self.castling.0 as usize];

        // The fifty-move clock resets on any pawn move or capture, else ticks up.
        self.halfmove_clock =
            if is_pawn || captured.is_some() { 0 } else { self.halfmove_clock + 1 };

        // Hand over to the opponent; the move number counts completed Black moves.
        self.side_to_move = us.flip();
        hash ^= zobrist::KEYS.side;
        if us == Color::Black {
            self.fullmove_number += 1;
        }

        // Add the new en-passant contribution now that the ep square, side to
        // move, and pawn positions are all final.
        hash ^= self.ep_zobrist();
        self.hash = hash;
        // The incrementally maintained key must always equal a from-scratch
        // recomputation; a mismatch means a feature toggle was missed above.
        debug_assert_eq!(self.hash, zobrist::compute(self), "incremental hash drifted");

        Undo {
            captured,
            castling: prev_castling,
            ep_square: prev_ep_square,
            halfmove_clock: prev_halfmove_clock,
            hash: prev_hash,
            dirty,
        }
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
        // XOR is invertible, so we *could* replay the move's key deltas; storing
        // the pre-move key and restoring it is simpler and O(1).
        self.hash = undo.hash;
    }

    /// Make a **null move**: hand the turn to the opponent without moving a piece.
    /// Used by null-move pruning in the search — if even passing leaves us winning,
    /// the position is too good to bother searching the real moves.
    ///
    /// Only two board features change: the side to move flips, and the en-passant
    /// square is cleared (a pass can never be answered by an en-passant capture of
    /// a pawn that just double-pushed, because no pawn just double-pushed). Both
    /// are mirrored into the incrementally-maintained Zobrist key exactly as
    /// [`make_move`](Self::make_move) does, so the key stays consistent. Pairs with
    /// [`unmake_null_move`](Self::unmake_null_move).
    pub fn make_null_move(&mut self) -> Undo {
        let prev = Undo {
            captured: None,
            castling: self.castling,
            ep_square: self.ep_square,
            halfmove_clock: self.halfmove_clock,
            hash: self.hash, // pre-move snapshot; unmake restores it verbatim
            dirty: DirtyPiece::empty(), // a null move toggles no piece features
        };

        let mut hash = self.hash;
        hash ^= self.ep_zobrist(); // drop the old ep contribution (reads current ep/stm)
        self.ep_square = None;
        self.side_to_move = self.side_to_move.flip();
        hash ^= zobrist::KEYS.side;
        hash ^= self.ep_zobrist(); // add the new ep contribution (now 0)
        self.hash = hash;
        // Like make_move, the incremental key must equal a from-scratch recompute.
        debug_assert_eq!(self.hash, zobrist::compute(self), "incremental hash drifted (null)");

        prev
    }

    /// Reverse a [`make_null_move`](Self::make_null_move). Castling rights and the
    /// halfmove clock are untouched by a null move, so only the side, ep square,
    /// and key need restoring.
    pub fn unmake_null_move(&mut self, undo: Undo) {
        self.side_to_move = self.side_to_move.flip();
        self.ep_square = undo.ep_square;
        self.hash = undo.hash;
    }

    /// The Zobrist contribution of the en-passant square, which is **nonzero
    /// only when the side to move can actually capture en passant** — i.e. a
    /// pawn of theirs sits on a square attacking `ep_square`.
    ///
    /// This "capturable-only" rule is what lets transposed positions share a
    /// key: after `1.Nf3 e5 2.e4` the ep square is e3 but no black pawn can take
    /// it, so it must hash identically to `1.e4 e5 2.Nf3` (where Nf3 cleared the
    /// ep square entirely). Hashing on `ep_square.is_some()` alone would break
    /// that. `compute` and `make_move` both route through here, so they agree by
    /// construction.
    pub(crate) fn ep_zobrist(&self) -> u64 {
        match self.ep_square {
            Some(ep) if self.ep_capturable(ep) => zobrist::KEYS.ep_file[ep.file() as usize],
            _ => 0,
        }
    }

    /// Whether the side to move has a pawn positioned to capture on `ep`.
    /// Pawn attacks are symmetric in reverse: the squares a side-to-move pawn
    /// could attack `ep` *from* are exactly `pawn_attacks(enemy_color, ep)`.
    fn ep_capturable(&self, ep: Square) -> bool {
        let us = self.side_to_move;
        let our_pawns = self.pieces(PieceType::Pawn).intersect(self.color(us));
        !pawn_attacks(us.flip(), ep).intersect(our_pawns).is_empty()
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

    // ── DirtyPiece feature deltas (issue #43) ───────────────────────────────

    /// The `(piece, square)` features of `slice`, squares as raw indices, sorted
    /// by square so a set comparison is order-independent (squares are unique
    /// within an add/remove list, giving a total order).
    fn feats(slice: &[(Piece, Square)]) -> Vec<(Piece, u8)> {
        let mut v: Vec<(Piece, u8)> = slice.iter().map(|&(p, s)| (p, s.0)).collect();
        v.sort_by_key(|&(_, s)| s);
        v
    }

    /// Same, for a `(piece, "e4")` expectation spec written with algebraic squares.
    fn spec(items: &[(Piece, &str)]) -> Vec<(Piece, u8)> {
        let mut v: Vec<(Piece, u8)> =
            items.iter().map(|&(p, s)| (p, Square::from_str(s).unwrap().0)).collect();
        v.sort_by_key(|&(_, s)| s);
        v
    }

    /// Make the unique legal move matching `from`/`to`/`promo`, returning its undo.
    fn make_uci(board: &mut Board, from: &str, to: &str, promo: Option<PieceType>) -> Undo {
        let from = Square::from_str(from).unwrap();
        let to = Square::from_str(to).unwrap();
        let mv = generate_legal(board)
            .into_iter()
            .find(|m| m.from() == from && m.to() == to && m.promotion_piece() == promo)
            .expect("a legal move matching from/to/promo");
        board.make_move(mv)
    }

    fn assert_dirty(undo: &Undo, added: &[(Piece, &str)], removed: &[(Piece, &str)]) {
        assert_eq!(feats(undo.dirty.added()), spec(added), "added features");
        assert_eq!(feats(undo.dirty.removed()), spec(removed), "removed features");
    }

    #[test]
    fn dirty_piece_reconstructs_board_change_across_perft() {
        // The acceptance gate: at every node of a perft walk, replaying the move's
        // DirtyPiece onto the pre-move board (removes first so put_piece's
        // empty-assert holds on a capture `to` square, then adds) must reproduce
        // the post-move *piece placement* exactly. We compare only the piece views
        // — stm/castling/ep/clock/hash legitimately differ and DirtyPiece does not
        // track them.
        fn walk(board: &mut Board, depth: u32) {
            for mv in generate_legal(board) {
                let before = board.clone();
                let undo = board.make_move(mv);

                let mut recon = before.clone();
                for &(piece, sq) in undo.dirty.removed() {
                    // Bonus check: the feature names the piece actually on the square.
                    assert_eq!(
                        recon.remove_piece(sq),
                        Some(piece),
                        "{mv}: removed mismatch at {sq}"
                    );
                }
                for &(piece, sq) in undo.dirty.added() {
                    recon.put_piece(sq, piece);
                }
                assert_eq!(recon.piece_bb, board.piece_bb, "{mv}: piece_bb diverged");
                assert_eq!(recon.color_bb, board.color_bb, "{mv}: color_bb diverged");
                assert_eq!(recon.mailbox, board.mailbox, "{mv}: mailbox diverged");

                if depth > 1 {
                    walk(board, depth - 1);
                }
                board.unmake_move(mv, undo);
            }
        }
        for fen in [STARTPOS, KIWIPETE, POS3, POS4, POS5, EP_POS] {
            walk(&mut board(fen), 3);
        }
    }

    #[test]
    fn dirty_piece_quiet_move() {
        let wp = piece(Color::White, PieceType::Pawn);
        let undo = make_uci(&mut board(STARTPOS), "e2", "e4", None);
        assert_dirty(&undo, &[(wp, "e4")], &[(wp, "e2")]);
    }

    #[test]
    fn dirty_piece_capture() {
        let wp = piece(Color::White, PieceType::Pawn);
        let bp = piece(Color::Black, PieceType::Pawn);
        let undo = make_uci(&mut board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1"), "e4", "d5", None);
        assert_dirty(&undo, &[(wp, "d5")], &[(wp, "e4"), (bp, "d5")]);
    }

    #[test]
    fn dirty_piece_en_passant() {
        // White e5 pawn takes d6 e.p.; the victim sits on d5, not on the `to` square.
        let wp = piece(Color::White, PieceType::Pawn);
        let bp = piece(Color::Black, PieceType::Pawn);
        let undo = make_uci(&mut board(EP_POS), "e5", "d6", None);
        assert_dirty(&undo, &[(wp, "d6")], &[(wp, "e5"), (bp, "d5")]);
    }

    #[test]
    fn dirty_piece_castles() {
        let wk = piece(Color::White, PieceType::King);
        let wr = piece(Color::White, PieceType::Rook);
        let bk = piece(Color::Black, PieceType::King);
        let br = piece(Color::Black, PieceType::Rook);
        const WHITE: &str = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
        const BLACK: &str = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1";

        let undo = make_uci(&mut board(WHITE), "e1", "g1", None); // O-O
        assert_dirty(&undo, &[(wk, "g1"), (wr, "f1")], &[(wk, "e1"), (wr, "h1")]);

        let undo = make_uci(&mut board(WHITE), "e1", "c1", None); // O-O-O
        assert_dirty(&undo, &[(wk, "c1"), (wr, "d1")], &[(wk, "e1"), (wr, "a1")]);

        let undo = make_uci(&mut board(BLACK), "e8", "g8", None); // ...O-O
        assert_dirty(&undo, &[(bk, "g8"), (br, "f8")], &[(bk, "e8"), (br, "h8")]);

        let undo = make_uci(&mut board(BLACK), "e8", "c8", None); // ...O-O-O
        assert_dirty(&undo, &[(bk, "c8"), (br, "d8")], &[(bk, "e8"), (br, "a8")]);
    }

    #[test]
    fn dirty_piece_promotion() {
        // a7-a8 promoting to each piece: the pawn leaves a7, the promoted piece
        // (not a pawn) arrives on a8.
        let wp = piece(Color::White, PieceType::Pawn);
        for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
            let promoted = piece(Color::White, pt);
            let undo = make_uci(&mut board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"), "a7", "a8", Some(pt));
            assert_dirty(&undo, &[(promoted, "a8")], &[(wp, "a7")]);
        }
    }

    #[test]
    fn dirty_piece_promotion_with_capture() {
        // a7xb8 capturing a knight and promoting: pawn leaves a7, victim leaves b8,
        // the promoted piece arrives on b8.
        let wp = piece(Color::White, PieceType::Pawn);
        let bn = piece(Color::Black, PieceType::Knight);
        for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
            let promoted = piece(Color::White, pt);
            let undo =
                make_uci(&mut board("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1"), "a7", "b8", Some(pt));
            assert_dirty(&undo, &[(promoted, "b8")], &[(wp, "a7"), (bn, "b8")]);
        }
    }

    #[test]
    fn dirty_piece_null_move_is_empty() {
        let undo = board(STARTPOS).make_null_move();
        assert!(undo.dirty.added().is_empty());
        assert!(undo.dirty.removed().is_empty());
    }

    // ── Zobrist hashing (issue #23) ─────────────────────────────────────────

    /// Play the (assumed legal) move from `from` to `to` on `board`.
    fn play(board: &mut Board, from: &str, to: &str) {
        let from = Square::from_str(from).unwrap();
        let to = Square::from_str(to).unwrap();
        let mv = generate_legal(board)
            .into_iter()
            .find(|m| m.from() == from && m.to() == to)
            .expect("a legal move between the given squares");
        board.make_move(mv);
    }

    #[test]
    fn transposed_move_orders_reach_the_same_hash() {
        // The canonical test: 1.e4 e5 2.Nf3 and 1.Nf3 e5 2.e4 reach the same
        // position, so they must share a Zobrist key. The two differ in their
        // halfmove clock (not hashed) and in a *non-capturable* ep square after
        // the e4-last order (not hashed, thanks to the capturable-only rule), so
        // the boards are not bit-for-bit equal — only the hashes are.
        let mut a = board(STARTPOS);
        play(&mut a, "e2", "e4");
        play(&mut a, "e7", "e5");
        play(&mut a, "g1", "f3");

        let mut b = board(STARTPOS);
        play(&mut b, "g1", "f3");
        play(&mut b, "e7", "e5");
        play(&mut b, "e2", "e4");

        assert_eq!(a.hash, b.hash, "transposed positions must share a hash");
    }

    #[test]
    fn incremental_hash_matches_from_scratch_across_perft() {
        // Walk the move tree and assert the maintained key equals a fresh
        // recomputation at every node. (The same check runs as a debug_assert
        // inside make_move; this makes it an explicit, named guarantee.)
        fn walk(board: &mut Board, depth: u32) {
            assert_eq!(board.hash, crate::zobrist::compute(board));
            if depth == 0 {
                return;
            }
            for mv in generate_legal(board) {
                let undo = board.make_move(mv);
                walk(board, depth - 1);
                board.unmake_move(mv, undo);
                // unmake must put the key back exactly.
                assert_eq!(board.hash, crate::zobrist::compute(board));
            }
        }
        for fen in [STARTPOS, KIWIPETE, POS3, POS4, POS5, EP_POS] {
            walk(&mut board(fen), 3);
        }
    }

    // ── Null move (issue #36) ───────────────────────────────────────────────

    #[test]
    fn null_move_hash_matches_a_fresh_parse() {
        // The forward-hash validator. A round-trip alone can't catch a bug in
        // `make_null_move`'s key math, because `unmake_null_move` restores the
        // pre-move snapshot verbatim — so we compare the *made* key against a
        // board parsed directly into the post-null position (opponent to move, ep
        // cleared). Equal keys prove the remove-old-ep / flip-side / add-new-ep
        // ordering is right.

        // No ep: only the side flips.
        let mut b = board(STARTPOS);
        b.make_null_move();
        assert_eq!(b.hash, board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1").hash);

        // Live ep: EP_POS's d6 is capturable by White's e5 pawn, so its ep key is
        // nonzero — a null move must XOR it back out. (A non-capturable ep would
        // hash to 0 and make this assertion vacuous.)
        assert_ne!(board(EP_POS).ep_zobrist(), 0, "EP_POS must have a live ep key");
        let mut b = board(EP_POS);
        b.make_null_move();
        assert_eq!(b.hash, board("4k3/8/8/3pP3/8/8/8/4K3 b - - 0 1").hash);
    }

    #[test]
    fn null_move_roundtrips() {
        // make then unmake restores the board bit-for-bit (Board derives Eq),
        // including the live-ep case where ep state must come back.
        for fen in [STARTPOS, KIWIPETE, POS4, EP_POS] {
            let original = board(fen);
            let mut b = original.clone();
            let undo = b.make_null_move();
            b.unmake_null_move(undo);
            assert_eq!(b, original, "null move did not round-trip for {fen}");
        }
    }
}
