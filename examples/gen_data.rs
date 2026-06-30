//! gen_data — generate a labelled, quiet-filtered self-play dataset.
//!
//! Two output formats share one self-play harness (issues #42 and #44):
//!
//! - `texel` (default): one `FEN RESULT` line per **quiet** position, where
//!   `RESULT ∈ {1.0, 0.5, 0.0}` is the game outcome **from White's point of view**.
//!   The Texel tuner ([`examples/texel.rs`]) fits the eval so `sigmoid(K · eval_white)`
//!   predicts that label.
//! - `bullet`: one bulletformat text line per quiet position —
//!   `FEN | <score> | <result>` — for training the NNUE (#44 → bullet, #45). **Both
//!   the score (centipawns) and the result are White-relative**, which is the format
//!   bullet's `bulletformat` loader expects: it derives side-to-move from the FEN and
//!   flips to the stm perspective internally. Emitting stm-relative data here would
//!   double-flip and train a broken net.
//!
//! Each game starts from the standard position, plays a few **random** opening plies
//! (a seeded LCG — reproducible, and far more diverse than a fixed book) to spread the
//! data across openings, then self-plays the rest with a shallow fixed-depth search.
//! Positions are recorded only after the random opening, only when
//! [`engine::search::is_quiet`] holds (qsearch == static eval, side not in check), so
//! the recorded score is a meaningful evaluation of a stable position. Games end on
//! mate/stalemate, the fifty-move rule, threefold repetition, a decisive-score resign
//! adjudication, or a ply cap.
//!
//! Run (datasets are large — gitignored; commit the *tuned constants* / *trained net*,
//! not the data):
//!
//! ```text
//! cargo run --release --example gen_data -- <target_positions> <depth> <seed> [format]
//! #   texel:  ... -- 100000 6 1            > data/texel.txt
//! #   bullet: ... -- 100000000 6 1 bullet  > data/nnue.txt
//! ```
//!
//! Defaults: 100000 positions, depth 6, seed 1, format `texel`. Reproducible: same
//! args → same file (single-threaded, seeded; no wall-clock or RNG-crate dependence).

use std::collections::HashMap;
use std::env;
use std::io::{self, BufWriter, Write};
use std::str::FromStr;

use engine::board::Board;
use engine::movegen::{generate_legal, in_check};
use engine::perft::STARTPOS;
use engine::search::{is_quiet, search_with_tt, MATE};
use engine::tt::TranspositionTable;
use engine::types::Color;

/// Random opening plies per game before recording / search self-play begins.
const RANDOM_PLIES: usize = 8;
/// Hard cap on game length (plies) — reached games are scored a draw.
const MAX_PLIES: usize = 300;
/// Resign adjudication: if one side is ahead by at least this many centipawns for
/// [`RESIGN_PLIES`] consecutive plies, the game is awarded to it.
const RESIGN_CP: i32 = 1000;
const RESIGN_PLIES: u32 = 6;

/// Which line format to emit. See the module docs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    /// `FEN RESULT` (White-POV WDL) — the Texel tuner's input (#42).
    Texel,
    /// `FEN | score | result` (both White-relative) — bullet's input (#44).
    Bullet,
}

/// A tiny seeded LCG (PCG-style multiplier) — deterministic, no `rand` crate.
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Return a well-mixed value (xorshift on the high bits).
        let x = self.0;
        (x ^ (x >> 29)).wrapping_mul(0xbf58476d1ce4e5b9)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// The outcome of a self-play game, from White's point of view.
#[derive(Clone, Copy)]
enum Outcome {
    WhiteWin,
    Draw,
    BlackWin,
}

impl Outcome {
    /// White-POV result string, shared by both formats (`1.0` win / `0.5` draw /
    /// `0.0` loss).
    fn label(self) -> &'static str {
        match self {
            Outcome::WhiteWin => "1.0",
            Outcome::Draw => "0.5",
            Outcome::BlackWin => "0.0",
        }
    }
}

/// One recorded training sample: a quiet position's FEN and its White-relative
/// search score in centipawns. The game result is attached per game at write time.
struct Sample {
    fen: String,
    white_score: i32,
}

/// Running sanity statistics over the written dataset, logged at the end (#44
/// acceptance criterion).
#[derive(Default)]
struct Stats {
    considered: u64, // non-terminal plies eligible to record
    written: u64,    // quiet positions actually written
    wins: u64,       // written positions from White-win games
    draws: u64,
    losses: u64,
    score_min: i32,
    score_max: i32,
    score_sum: i64,
    /// Signed centipawn histogram; bin edges in [`Stats::BIN_EDGES`].
    hist: [u64; Stats::N_BINS],
}

impl Stats {
    /// Upper edges of the signed score histogram bins; the final bin is `> last`.
    const BIN_EDGES: [i32; 8] = [-1000, -400, -150, -50, 50, 150, 400, 1000];
    const N_BINS: usize = Self::BIN_EDGES.len() + 1;

    fn new() -> Stats {
        Stats { score_min: i32::MAX, score_max: i32::MIN, ..Stats::default() }
    }

    fn record(&mut self, white_score: i32, outcome: Outcome) {
        self.written += 1;
        self.score_min = self.score_min.min(white_score);
        self.score_max = self.score_max.max(white_score);
        self.score_sum += white_score as i64;
        let bin =
            Self::BIN_EDGES.iter().position(|&e| white_score <= e).unwrap_or(Self::N_BINS - 1);
        self.hist[bin] += 1;
        match outcome {
            Outcome::WhiteWin => self.wins += 1,
            Outcome::Draw => self.draws += 1,
            Outcome::BlackWin => self.losses += 1,
        }
    }

    fn report(&self, games: usize, depth: u32, seed: u64) {
        let pct_quiet = if self.considered > 0 {
            100.0 * self.written as f64 / self.considered as f64
        } else {
            0.0
        };
        let mean = if self.written > 0 { self.score_sum as f64 / self.written as f64 } else { 0.0 };
        eprintln!(
            "done: {} positions from {games} games (depth {depth}, seed {seed})",
            self.written
        );
        eprintln!("  quiet: {}/{} considered ({pct_quiet:.1}%)", self.written, self.considered);
        eprintln!(
            "  result: {} win / {} draw / {} loss (White POV)",
            self.wins, self.draws, self.losses
        );
        eprintln!(
            "  score (White cp): min {} mean {mean:.0} max {}",
            self.score_min, self.score_max
        );
        let labels =
            ["<-1000", "-400..", "-150..", "-50..", "-50..50", "..150", "..400", "..1000", ">1000"];
        let mut hist = String::from("  hist:");
        for (label, count) in labels.iter().zip(self.hist.iter()) {
            hist.push_str(&format!(" {label}={count}"));
        }
        eprintln!("{hist}");
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let target: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let depth: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let format = match args.get(4).map(String::as_str) {
        None | Some("texel") => Format::Texel,
        Some("bullet") => Format::Bullet,
        Some(other) => {
            eprintln!("gen_data: unknown format {other:?}; expected `texel` or `bullet`");
            std::process::exit(1);
        }
    };

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let mut stats = Stats::new();
    let mut games = 0usize;
    // One TT, reused across games (cleared per search) — realistic and cheaper.
    let mut tt = TranspositionTable::new(16);

    while stats.written < target as u64 {
        let mut rng = Lcg(seed.wrapping_add(games as u64).wrapping_mul(0x9e3779b97f4a7c15));
        let (samples, outcome, considered) = play_game(&mut rng, depth, &mut tt);
        stats.considered += considered;
        for sample in samples {
            // Stop exactly at the target so a run is fully determined by its args.
            if stats.written >= target as u64 {
                break;
            }
            write_sample(&mut out, format, &sample, outcome);
            stats.record(sample.white_score, outcome);
        }
        games += 1;
        if games.is_multiple_of(200) {
            eprintln!("games {games}, positions {}/{target}", stats.written);
        }
    }
    out.flush().expect("flush dataset");
    stats.report(games, depth, seed);
}

/// Write one sample line in the requested format. `outcome` is the game's White-POV
/// result; `sample.white_score` is the position's White-POV centipawn score.
fn write_sample<W: Write>(out: &mut W, format: Format, sample: &Sample, outcome: Outcome) {
    match format {
        Format::Texel => writeln!(out, "{} {}", sample.fen, outcome.label()),
        // bullet's binary `eval` is an i16; clamp defensively so any extreme (e.g. a
        // mate-adjacent quiet position) can never overflow the downstream conversion.
        Format::Bullet => {
            let score = sample.white_score.clamp(i16::MIN as i32, i16::MAX as i32);
            writeln!(out, "{} | {} | {}", sample.fen, score, outcome.label())
        }
    }
    .expect("write dataset line");
}

/// Play one self-play game. Return the quiet samples to record (FEN + White-POV
/// score), the game outcome (White's POV), and the count of positions *considered*
/// (non-terminal plies eligible to record — the denominator for the quiet fraction).
fn play_game(
    rng: &mut Lcg,
    depth: u32,
    tt: &mut TranspositionTable,
) -> (Vec<Sample>, Outcome, u64) {
    let mut board = Board::from_str(STARTPOS).expect("startpos FEN");
    let mut counts: HashMap<u64, u8> = HashMap::new();
    let mut samples = Vec::new();
    let mut considered = 0u64;

    // Random opening for diversity (not recorded).
    for _ in 0..RANDOM_PLIES {
        let legal = generate_legal(&board);
        if legal.is_empty() {
            return (samples, terminal_outcome(&board), considered);
        }
        let mv = legal[rng.below(legal.len())];
        board.make_move(mv);
    }

    let mut resign_leader: Option<Color> = None;
    let mut resign_count = 0u32;

    for _ in 0..MAX_PLIES {
        let legal = generate_legal(&board);
        if legal.is_empty() {
            return (samples, terminal_outcome(&board), considered);
        }
        if board.halfmove_clock >= 100 {
            return (samples, Outcome::Draw, considered); // fifty-move rule
        }
        let seen = counts.entry(board.hash).or_insert(0);
        *seen += 1;
        if *seen >= 3 {
            return (samples, Outcome::Draw, considered); // threefold repetition
        }

        considered += 1;
        // Evaluate the position once; the score both labels a recorded sample and
        // drives the decisive-score adjudication below. `is_quiet` is independent of
        // the search, so the recorded set matches the pre-#44 (record-before-search)
        // behaviour exactly — the Texel output is byte-identical.
        let quiet = is_quiet(&board);
        let result = search_with_tt(&board, depth, tt);
        let white_score =
            if board.side_to_move == Color::White { result.score } else { -result.score };
        if quiet {
            samples.push(Sample { fen: board.to_fen(), white_score });
        }

        if result.score >= MATE - 1000 {
            // The side to move has a forced mate (mate scores sit near ±MATE).
            return (samples, win_for(board.side_to_move), considered);
        }
        if white_score.abs() >= RESIGN_CP {
            let leader = if white_score > 0 { Color::White } else { Color::Black };
            if resign_leader == Some(leader) {
                resign_count += 1;
            } else {
                resign_leader = Some(leader);
                resign_count = 1;
            }
            if resign_count >= RESIGN_PLIES {
                return (samples, win_for(leader), considered);
            }
        } else {
            resign_leader = None;
            resign_count = 0;
        }

        board.make_move(result.best_move);
    }
    (samples, Outcome::Draw, considered) // ply cap
}

/// Outcome when the side to move has no legal move: checkmate (the side to move
/// loses) or stalemate (draw).
fn terminal_outcome(board: &Board) -> Outcome {
    if in_check(board, board.side_to_move) {
        win_for(board.side_to_move.flip()) // side to move is mated
    } else {
        Outcome::Draw
    }
}

fn win_for(c: Color) -> Outcome {
    match c {
        Color::White => Outcome::WhiteWin,
        Color::Black => Outcome::BlackWin,
    }
}
