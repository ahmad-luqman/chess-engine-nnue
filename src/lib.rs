//! chess-engine-nnue — library crate.
//!
//! Build order (see docs/03-roadmap.md): types → bitboard → board → movegen →
//! perft → search → eval → uci. We are in Phase 0.

pub mod bitboard;
pub mod board;
pub mod fen;
pub mod movegen;
pub mod moves;
pub mod types;
