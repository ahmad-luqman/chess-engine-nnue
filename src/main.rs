//! Engine binary entry point.
//!
//! With no arguments (or `uci`) it runs the **UCI loop** — the engine's normal
//! mode, the way Cute Chess and other managers drive it. The `perft` subcommand
//! is kept as the Phase 0 correctness/benchmark tool. Per the iron rules, perft
//! numbers are only meaningful in `--release`.
//!
//! ```text
//! cargo run --release                            # UCI loop (stdin/stdout)
//! cargo run --release -- perft 6                 # startpos to depth 6
//! cargo run --release -- perft 5 "<FEN>"         # any position
//! ```

use std::process::ExitCode;
use std::str::FromStr;
use std::time::Instant;

use engine::board::Board;
use engine::perft::perft_divide;

const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("perft") => run_perft(&args[2..]),
        // Default mode (no args) and an explicit `uci` both enter the protocol
        // loop. A GUI launches the bare binary and speaks UCI immediately.
        None | Some("uci") => match engine::uci::run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("uci: I/O error: {err}");
                ExitCode::FAILURE
            }
        },
        Some(other) => {
            eprintln!("unknown subcommand {other:?}; expected `uci` or `perft`");
            ExitCode::FAILURE
        }
    }
}

/// Run `perft_divide` at the requested depth and print the per-move breakdown, a
/// total, and the nodes-per-second — the shape every chess engine's `perft`
/// command prints, so output can be diffed against a reference move by move.
fn run_perft(args: &[String]) -> ExitCode {
    let depth = match args.first().and_then(|d| d.parse::<u32>().ok()) {
        Some(d) if d >= 1 => d,
        _ => {
            eprintln!("perft: expected a depth >= 1, e.g. `perft 5`");
            return ExitCode::FAILURE;
        }
    };
    let fen = args.get(1).map(String::as_str).unwrap_or(STARTPOS);
    let mut board = match Board::from_str(fen) {
        Ok(board) => board,
        Err(err) => {
            eprintln!("perft: could not parse FEN: {err:?}");
            return ExitCode::FAILURE;
        }
    };

    let start = Instant::now();
    let breakdown = perft_divide(&mut board, depth);
    let elapsed = start.elapsed();

    let mut total = 0;
    for (mv, nodes) in &breakdown {
        println!("{mv}: {nodes}");
        total += nodes;
    }
    let secs = elapsed.as_secs_f64();
    let nps = if secs > 0.0 { (total as f64 / secs) as u64 } else { 0 };
    println!("\nperft({depth}) = {total}  ({:.3}s, {nps} nps)", secs);
    ExitCode::SUCCESS
}
