//! chess-engine-nnue — library crate.
//!
//! Build order (see docs/03-roadmap.md): types → bitboard → board → movegen →
//! perft → search → eval → uci. We are in Phase 1 (search + eval + UCI).

pub mod bitboard;
pub mod board;
pub mod eval;
pub mod fen;
pub mod movegen;
pub mod moves;
pub mod perft;
pub mod search;
pub mod types;
pub mod uci;
