//! Criterion micro-benchmarks for the engine's hot paths (issue #30).
//!
//! These exist to turn "is this faster?" into a number. Run them in release
//! (criterion always builds optimized) and use baselines to catch regressions:
//!
//! ```text
//! cargo bench                              # run everything, print results
//! cargo bench --bench engine sliders      # one group (substring-filtered)
//! cargo bench -- --save-baseline v0.5.0    # stash a named baseline
//! cargo bench -- --baseline v0.5.0         # compare a later run against it
//! ```
//!
//! Every bench `black_box`es its inputs. That is load-bearing, not decoration:
//! with fat LTO + one codegen unit, constant inputs to a pure function (the
//! eval and slider leaves especially) would otherwise be constant-folded away
//! and we'd time an empty loop. `black_box` hides the inputs from the optimizer
//! so the call actually runs; returning the result lets criterion black-box the
//! output too.
//!
//! Benches reuse the published perft FENs (`engine::perft::{STARTPOS, …}`) so
//! they measure the exact positions the correctness tests pin.

use std::hint::black_box;
use std::str::FromStr;

use criterion::{criterion_group, criterion_main, Criterion};

use engine::board::Board;
use engine::eval::{Evaluator, Material};
use engine::movegen::{
    bishop_attacks, generate_legal, ray_attacks, rook_attacks, BISHOP_DIRS, ROOK_DIRS,
};
use engine::perft::{perft, KIWIPETE, POS3, POS4, STARTPOS};
use engine::types::Square;

fn board(fen: &str) -> Board {
    Board::from_str(fen).expect("benchmark FEN is valid")
}

/// End-to-end movegen + make/unmake loop, at the issue's depths. The clone-per-
/// move legality filter makes this the slowest group, so we trim the sample
/// count (criterion still times many iterations *within* each sample).
fn bench_perft(c: &mut Criterion) {
    let mut group = c.benchmark_group("perft");
    group.sample_size(10);
    for (name, fen, depth) in [("startpos/5", STARTPOS, 5), ("kiwipete/4", KIWIPETE, 4)] {
        group.bench_function(name, |b| {
            b.iter(|| perft(black_box(&mut board(fen)), black_box(depth)))
        });
    }
    group.finish();
}

/// Pseudo-legal generation + the copy-make legality filter, per position. The
/// returned `Vec<Move>` is a real allocation, so this also exercises the
/// move-list path the search hits at every node.
fn bench_movegen(c: &mut Criterion) {
    let mut group = c.benchmark_group("generate_legal");
    for (name, fen) in
        [("startpos", STARTPOS), ("kiwipete", KIWIPETE), ("pos3", POS3), ("pos4", POS4)]
    {
        let board = board(fen);
        group.bench_function(name, |b| b.iter(|| generate_legal(black_box(&board))));
    }
    group.finish();
}

/// The leaf evaluation (`Material::evaluate`): material + piece-square tables.
/// Called once per quiescence leaf, so its cost is multiplied across the tree.
fn bench_eval(c: &mut Criterion) {
    let evaluator = Material;
    let mut group = c.benchmark_group("eval");
    for (name, fen) in [("startpos", STARTPOS), ("kiwipete", KIWIPETE)] {
        let board = board(fen);
        group.bench_function(name, |b| b.iter(|| evaluator.evaluate(black_box(&board))));
    }
    group.finish();
}

/// make_move → unmake_move round-trip. We make and immediately unmake the same
/// move so the board is restored each iteration, isolating the cost of the
/// incremental update + undo that perft and search pay on every edge.
fn bench_make_unmake(c: &mut Criterion) {
    let mut group = c.benchmark_group("make_unmake");
    for (name, fen) in [("startpos", STARTPOS), ("kiwipete", KIWIPETE)] {
        let mut board = board(fen);
        // A real legal move from the position; first in generation order.
        let mv = generate_legal(&board)[0];
        group.bench_function(name, |b| {
            b.iter(|| {
                let undo = board.make_move(black_box(mv));
                board.unmake_move(black_box(mv), undo);
            })
        });
    }
    group.finish();
}

/// Sliding-attack lookup: magic bitboards vs the `ray_attacks` walk they
/// replaced (issue #27). Same squares and same occupancy for both, so the ratio
/// is the magic-bitboard win. Occupancy is taken from Kiwipete (a busy middle-
/// game position) rather than an empty board, which would flatter the ray walk.
fn bench_sliders(c: &mut Criterion) {
    let occupied = board(KIWIPETE).occupied();
    // A spread of squares: corner, centre, and edge.
    let squares = [Square(0), Square(27), Square(36), Square(63)];

    let mut group = c.benchmark_group("sliders");
    group.bench_function("rook/magic", |b| {
        b.iter(|| {
            for &sq in &squares {
                black_box(rook_attacks(black_box(sq), black_box(occupied)));
            }
        })
    });
    group.bench_function("rook/ray_oracle", |b| {
        b.iter(|| {
            for &sq in &squares {
                black_box(ray_attacks(black_box(sq), black_box(occupied), &ROOK_DIRS));
            }
        })
    });
    group.bench_function("bishop/magic", |b| {
        b.iter(|| {
            for &sq in &squares {
                black_box(bishop_attacks(black_box(sq), black_box(occupied)));
            }
        })
    });
    group.bench_function("bishop/ray_oracle", |b| {
        b.iter(|| {
            for &sq in &squares {
                black_box(ray_attacks(black_box(sq), black_box(occupied), &BISHOP_DIRS));
            }
        })
    });
    group.finish();
}

criterion_group!(benches, bench_perft, bench_movegen, bench_eval, bench_make_unmake, bench_sliders);
criterion_main!(benches);
