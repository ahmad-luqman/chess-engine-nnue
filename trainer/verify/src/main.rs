//! Reference NNUE inference + sanity check for the first net (issue #45).
//!
//! Pure `std`, zero dependencies, no GPU — runnable on any machine. This file is
//! the **inference contract** that engine wiring (#46) copies into `src/`. Every
//! convention here (feature index, perspective flip, stm-first concat, int32
//! SCReLU, output-bucket formula, dequantisation order) must match what `#46`
//! implements, or the net will evaluate garbage.
//!
//! Net file = bullet's `quantised.bin`, little-endian i16, in `save_format` order
//! (see `bullet-train/src/main.rs` and ADR 0016):
//!   feature_weights : [768][HL]            column-major of (HL x 768)
//!   feature_bias    : [HL]
//!   output_weights  : [BUCKETS][2*HL]      bucket-major (l1w was `.transpose()`d)
//!   output_bias     : [BUCKETS]
//! followed by zero padding to a multiple of 64 bytes.
//!
//! Usage: `cargo run -r -p verify -- <path-to-net.bin>`

use std::process::ExitCode;

const HL: usize = 256;
const BUCKETS: usize = 8;
const QA: i32 = 255;
const QB: i32 = 64;
const SCALE: i32 = 400;

const N_FEATURES: usize = 768;

struct Network {
    feature_weights: Vec<[i16; HL]>, // indexed by feature (len 768)
    feature_bias: [i16; HL],
    output_weights: Vec<[i16; 2 * HL]>, // indexed by bucket (len 8); 0..HL = stm, HL.. = ntm
    output_bias: [i16; BUCKETS],
}

impl Network {
    fn load(bytes: &[u8]) -> Network {
        let need = (N_FEATURES * HL + HL + BUCKETS * 2 * HL + BUCKETS) * 2;
        assert!(bytes.len() >= need, "net file too small: {} < {need} bytes", bytes.len());

        let mut idx = 0usize;
        let mut next = || {
            let v = i16::from_le_bytes([bytes[idx], bytes[idx + 1]]);
            idx += 2;
            v
        };

        let mut feature_weights = Vec::with_capacity(N_FEATURES);
        for _ in 0..N_FEATURES {
            let mut col = [0i16; HL];
            for w in col.iter_mut() {
                *w = next();
            }
            feature_weights.push(col);
        }

        let mut feature_bias = [0i16; HL];
        for b in feature_bias.iter_mut() {
            *b = next();
        }

        let mut output_weights = Vec::with_capacity(BUCKETS);
        for _ in 0..BUCKETS {
            let mut row = [0i16; 2 * HL];
            for w in row.iter_mut() {
                *w = next();
            }
            output_weights.push(row);
        }

        let mut output_bias = [0i16; BUCKETS];
        for b in output_bias.iter_mut() {
            *b = next();
        }

        Network { feature_weights, feature_bias, output_weights, output_bias }
    }

    /// Evaluate a position, returning centipawns from the side-to-move's
    /// perspective (positive = stm is better) — matching the engine's
    /// `Evaluator::evaluate` convention.
    fn evaluate(&self, pieces: &[Piece], stm: usize) -> i32 {
        // Two accumulators, both initialised with the (shared) feature bias.
        let mut acc_stm = self.feature_bias;
        let mut acc_ntm = self.feature_bias;

        for p in pieces {
            let fi_stm = feature_index(stm, p.color, p.pt, p.sq);
            let fi_ntm = feature_index(stm ^ 1, p.color, p.pt, p.sq);
            let col_stm = &self.feature_weights[fi_stm];
            let col_ntm = &self.feature_weights[fi_ntm];
            for i in 0..HL {
                acc_stm[i] += col_stm[i];
                acc_ntm[i] += col_ntm[i];
            }
        }

        // Output bucket by piece count: MaterialCount<8> => (count - 2) / 4.
        let count = pieces.len() as i32;
        let bucket = (((count - 2) / 4).clamp(0, BUCKETS as i32 - 1)) as usize;
        let weights = &self.output_weights[bucket];

        // SCReLU MUST accumulate in i32 (clamp(x,0,QA)^2 reaches QA*QA = 65025,
        // which overflows i16). stm accumulator first, then opponent.
        let mut out: i32 = 0;
        for i in 0..HL {
            out += screlu(acc_stm[i]) * i32::from(weights[i]);
        }
        for i in 0..HL {
            out += screlu(acc_ntm[i]) * i32::from(weights[HL + i]);
        }

        // Dequantise: sum is in QA*QA*QB; reduce to QA*QB, add bias (QA*QB),
        // scale, then strip the final QA*QB. Order matches bullet's example.
        out /= QA;
        out += i32::from(self.output_bias[bucket]);
        out *= SCALE;
        out /= QA * QB;
        out
    }
}

#[inline]
fn screlu(x: i16) -> i32 {
    let v = i32::from(x).clamp(0, QA);
    v * v
}

/// The Chess768 feature index, replicated from bullet's `Chess768::map_features`.
///
/// `perspective`/`color`: 0 = white, 1 = black. `pt`: 0..=5 (P,N,B,R,Q,K).
/// `sq`: a1 = 0 .. h8 = 63 (engine convention). For black's perspective the
/// board is vertically mirrored (`sq ^ 56`) and friend/enemy colour is relative.
#[inline]
fn feature_index(perspective: usize, color: usize, pt: usize, sq: usize) -> usize {
    let relative_color = usize::from(color != perspective); // 0 = friendly, 1 = enemy
    let relative_sq = if perspective == 0 { sq } else { sq ^ 56 };
    [0, 384][relative_color] + 64 * pt + relative_sq
}

#[derive(Clone, Copy)]
struct Piece {
    color: usize, // 0 white, 1 black
    pt: usize,    // 0..=5
    sq: usize,    // a1=0..h8=63
}

/// Parse the piece-placement + side-to-move fields of a FEN into a piece list.
fn parse_fen(fen: &str) -> (Vec<Piece>, usize) {
    let mut fields = fen.split_whitespace();
    let placement = fields.next().expect("FEN placement field");
    let stm = match fields.next() {
        Some("b") => 1,
        _ => 0,
    };

    let mut pieces = Vec::new();
    // FEN ranks run 8 -> 1; squares are a1=0, so rank r (1-based) at file f maps
    // to (r-1)*8 + f. The first FEN rank is rank 8.
    let mut rank = 7i32;
    let mut file = 0i32;
    for ch in placement.chars() {
        match ch {
            '/' => {
                rank -= 1;
                file = 0;
            }
            '1'..='8' => file += (ch as u8 - b'0') as i32,
            _ => {
                let color = if ch.is_ascii_uppercase() { 0 } else { 1 };
                let pt = match ch.to_ascii_lowercase() {
                    'p' => 0,
                    'n' => 1,
                    'b' => 2,
                    'r' => 3,
                    'q' => 4,
                    'k' => 5,
                    other => panic!("bad FEN piece char: {other}"),
                };
                let sq = (rank * 8 + file) as usize;
                pieces.push(Piece { color, pt, sq });
                file += 1;
            }
        }
    }
    (pieces, stm)
}

/// Vertically mirror + colour-swap a position and flip the side to move. A
/// correct perspective net must evaluate the mirror identically (up to rounding):
/// the side to move sees the exact same relative features.
fn mirror(pieces: &[Piece], stm: usize) -> (Vec<Piece>, usize) {
    let mirrored =
        pieces.iter().map(|p| Piece { color: p.color ^ 1, pt: p.pt, sq: p.sq ^ 56 }).collect();
    (mirrored, stm ^ 1)
}

fn main() -> ExitCode {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: verify <path-to-net.bin>");
            return ExitCode::FAILURE;
        }
    };
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let net = Network::load(&bytes);
    println!("loaded {} ({} bytes)", path, bytes.len());

    // Reference positions. Material deltas are built from startpos so the checks
    // depend only on the mapping, not on how well the net is calibrated.
    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    // White's queen removed -> Black is up a queen (white to move).
    const W_NO_Q: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1";
    // Black's queen removed -> White is up a queen (white to move).
    const B_NO_Q: &str = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    let eval_fen = |fen: &str| {
        let (pieces, stm) = parse_fen(fen);
        net.evaluate(&pieces, stm)
    };

    let start = eval_fen(START);
    let white_up_q = eval_fen(B_NO_Q); // white to move, white up a queen
    let white_down_q = eval_fen(W_NO_Q); // white to move, white down a queen

    println!("  startpos (stm=white)        eval = {start:>6} cp");
    println!("  white up a queen  (w2m)     eval = {white_up_q:>6} cp");
    println!("  white down a queen (w2m)    eval = {white_down_q:>6} cp");

    // Perspective-flip check: mirror "white up a queen" -> "black up a queen,
    // black to move". A correct net evaluates both near-identically.
    let (pieces, stm) = parse_fen(B_NO_Q);
    let (mpieces, mstm) = mirror(&pieces, stm);
    let mirrored = net.evaluate(&mpieces, mstm);
    println!("  mirror of up-a-queen        eval = {mirrored:>6} cp (should ~= {white_up_q})");

    let all = [start, white_up_q, white_down_q, mirrored];
    let mut ok = true;

    // 1. All finite (i32 is always finite; guard against absurd magnitudes that
    //    would signal a layout/overflow bug).
    for (label, v) in ["start", "up_q", "down_q", "mirror"].iter().zip(all) {
        if v.abs() > 30_000 {
            eprintln!("FAIL: {label} eval {v} is implausibly large (layout/overflow bug?)");
            ok = false;
        }
    }
    // 2. Material monotonicity: up a queen >> startpos >> down a queen.
    if !(white_up_q > start + 200) {
        eprintln!("FAIL: up-a-queen ({white_up_q}) not >> startpos ({start})");
        ok = false;
    }
    if !(white_down_q < start - 200) {
        eprintln!("FAIL: down-a-queen ({white_down_q}) not << startpos ({start})");
        ok = false;
    }
    // 3. Perspective flip: mirror equals original up to quantisation rounding.
    //    This is THE mapping-correctness check — independent of how (or how well,
    //    or on what data) the net was trained. A wrong feature flip or concat
    //    order breaks it; nothing else here does.
    if (mirrored - white_up_q).abs() > 5 {
        eprintln!("FAIL: mirror {mirrored} != up-a-queen {white_up_q} (perspective bug)");
        ok = false;
    }
    // NOTE (not a failure): startpos magnitude is a *calibration* property, not a
    // mapping one. Bootstrap datasets filter out openings (ply >= 16) and may be
    // skewed toward decisive positions, so an untuned net can read startpos well
    // off zero. Calibration/strength is #48/#49's job, not #45's.
    if start.abs() > 150 {
        println!("  note: startpos eval {start} cp is off-balanced — expected for a");
        println!("        bootstrap net on opening-filtered data; calibration is #48.");
    }

    if ok {
        println!("\nOK: net loads and produces finite, sane, perspective-consistent evals.");
        ExitCode::SUCCESS
    } else {
        eprintln!("\nSANITY CHECK FAILED");
        ExitCode::FAILURE
    }
}
