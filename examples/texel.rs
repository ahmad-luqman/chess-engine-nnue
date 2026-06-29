//! texel — offline Texel tuner for the hand-crafted eval weights (issue #42).
//!
//! Fits the `(mg, eg)` eval weights by minimising the logistic prediction error of
//! the static eval against game-outcome-labelled quiet positions (from
//! [`examples/gen_data.rs`]):
//!
//! ```text
//! minimise  Σ (result − sigmoid(K · eval_white))²
//! ```
//!
//! The eval is **linear in its weights**, so each position is reduced once to a
//! sparse *feature trace* — the white-relative coefficient of every weight, split
//! into a middlegame and endgame count — and the fit is gradient descent (Adam)
//! over the weight vector. `PIECE_VALUE` is held fixed (pawn = 100 anchors the
//! `K` scaling and sidesteps the move-ordering coupling; the PST absorbs material
//! drift). The scaling constant `K` is solved first.
//!
//! A **consistency check** reconstructs the engine's integer eval from the trace
//! with the *default* weights and asserts it equals [`Material::evaluate`] exactly
//! on every position — the guard that the trace faithfully mirrors `eval`.
//!
//! Run (after generating a dataset):
//!
//! ```text
//! cargo run --release --example texel -- data/texel.txt
//! ```
//!
//! Prints the tuned constants as Rust, ready to paste into `src/eval.rs`.
//! Deterministic: same dataset file → same output (full-batch, fixed split/seed).

use std::env;
use std::fs;
use std::str::FromStr;

use engine::board::Board;
use engine::eval::{
    Evaluator, Material, ADJACENT_FILE_MASKS, BISHOP_PAIR, DOUBLED_PAWN, FILE_MASKS, ISOLATED_PAWN,
    KING_ATTACK_WEIGHT, KING_EG, KING_MG, MOBILITY, PASSED_MASKS, PASSED_PAWN_EG, PASSED_PAWN_MG,
    PAWN_SHIELD_HOLE, PIECE_SQUARE_TABLE, PIECE_VALUE, ROOK_OPEN_FILE, ROOK_SEMI_OPEN_FILE,
};
use engine::movegen::{
    bishop_attacks, king_attacks, king_square, knight_attacks, queen_attacks, rook_attacks,
};
use engine::types::{Color, PieceType, Square};

const PHASE_MAX: i32 = 24;

// ── Flat parameter layout ────────────────────────────────────────────────────
// Every tunable weight occupies a slot in a flat `theta` vector. Score pairs take
// two slots (mg, eg); phase-independent PST entries take one slot used by both
// phases; mg-only terms (king tables split, king safety) put their coefficient in
// only one of the two accumulators.
const PST: usize = 0; // 5 pieces × 64 squares
const KMG: usize = 320; // 64 (king middlegame table)
const KEG: usize = 384; // 64 (king endgame table)
const BP: usize = 448; // bishop pair (mg, eg)
const RO: usize = 450; // rook open file (mg, eg)
const RS: usize = 452; // rook semi-open file (mg, eg)
const MOB: usize = 454; // mobility N,B,R,Q × (mg, eg) = 8
const DBL: usize = 462; // doubled pawn (mg, eg)
const ISO: usize = 464; // isolated pawn (mg, eg)
const PMG: usize = 466; // passed pawn mg by rank (8)
const PEG: usize = 474; // passed pawn eg by rank (8)
const KAW: usize = 482; // king attack weight N,B,R,Q (4, mg-only)
const SHIELD: usize = 486; // pawn shield hole (1, mg-only)
const P: usize = 487;

/// Local index 0..4 (N,B,R,Q) for a sliding/attacking piece type.
fn nbrq_local(pt: PieceType) -> usize {
    pt.index() - 1 // Knight=1 → 0 … Queen=4 → 3
}

/// Replica of `eval::pst_index`: rank-8-first layout, Black reads mirrored.
fn pst_index(color: Color, sq: Square) -> usize {
    (match color {
        Color::White => sq.0 ^ 56,
        Color::Black => sq.0,
    }) as usize
}

/// A position reduced to its feature trace plus its game label.
struct Sample {
    result: f64, // White's POV: 1.0 win, 0.5 draw, 0.0 loss
    phase: i32,
    mat: i32,                     // white-relative material balance (fixed offset)
    coeffs: Vec<(u16, i16, i16)>, // (param index, mg coeff, eg coeff), sparse
}

fn count(board: &Board, pt: PieceType, c: Color) -> i32 {
    board.pieces(pt).intersect(board.color(c)).count() as i32
}

fn game_phase(board: &Board) -> i32 {
    let p = 4 * count(board, PieceType::Queen, Color::White)
        + 4 * count(board, PieceType::Queen, Color::Black)
        + 2 * count(board, PieceType::Rook, Color::White)
        + 2 * count(board, PieceType::Rook, Color::Black)
        + count(board, PieceType::Bishop, Color::White)
        + count(board, PieceType::Bishop, Color::Black)
        + count(board, PieceType::Knight, Color::White)
        + count(board, PieceType::Knight, Color::Black);
    p.min(PHASE_MAX)
}

/// Rook open/semi-open file counts for `color`: `(open, semi)`.
fn rook_counts(board: &Board, color: Color) -> (i32, i32) {
    let own_pawns = board.pieces(PieceType::Pawn).intersect(board.color(color));
    let enemy_pawns = board.pieces(PieceType::Pawn).intersect(board.color(color.flip()));
    let (mut open, mut semi) = (0, 0);
    let mut rooks = board.pieces(PieceType::Rook).intersect(board.color(color));
    while let Some(sq) = rooks.pop_lsb() {
        let f = FILE_MASKS[sq.file() as usize];
        if own_pawns.intersect(f).is_empty() {
            if enemy_pawns.intersect(f).is_empty() {
                open += 1;
            } else {
                semi += 1;
            }
        }
    }
    (open, semi)
}

/// Mobility square counts for `color`, per N,B,R,Q (attacked squares minus own).
fn mobility_counts(board: &Board, color: Color) -> [i32; 4] {
    let own = board.color(color);
    let occ = board.occupied();
    let mut out = [0i32; 4];
    for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
        let mut bb = board.pieces(pt).intersect(own);
        let mut sum = 0;
        while let Some(sq) = bb.pop_lsb() {
            let attacks = match pt {
                PieceType::Knight => knight_attacks(sq),
                PieceType::Bishop => bishop_attacks(sq, occ),
                PieceType::Rook => rook_attacks(sq, occ),
                PieceType::Queen => queen_attacks(sq, occ),
                _ => unreachable!(),
            };
            sum += attacks.minus(own).count() as i32;
        }
        out[nbrq_local(pt)] = sum;
    }
    out
}

/// Pawn-structure counts for `color`: `(doubled, isolated, passed_by_relative_rank)`.
fn pawn_counts(board: &Board, color: Color) -> (i32, i32, [i32; 8]) {
    let own = board.pieces(PieceType::Pawn).intersect(board.color(color));
    let enemy = board.pieces(PieceType::Pawn).intersect(board.color(color.flip()));
    let mut doubled = 0;
    for file in FILE_MASKS {
        let n = own.intersect(file).count() as i32;
        if n > 1 {
            doubled += n - 1;
        }
    }
    let (mut isolated, mut passed) = (0, [0i32; 8]);
    let mut pawns = own;
    while let Some(sq) = pawns.pop_lsb() {
        if own.intersect(ADJACENT_FILE_MASKS[sq.file() as usize]).is_empty() {
            isolated += 1;
        }
        if enemy.intersect(PASSED_MASKS[color.index()][sq.0 as usize]).is_empty() {
            let rel = match color {
                Color::White => sq.rank() as usize,
                Color::Black => 7 - sq.rank() as usize,
            };
            passed[rel] += 1;
        }
    }
    (doubled, isolated, passed)
}

/// King-safety counts for `color`'s king: `(enemy attackers per N,B,R,Q, shield holes)`.
fn king_safety_counts(board: &Board, color: Color) -> ([i32; 4], i32) {
    let ksq = king_square(board, color);
    let zone = king_attacks(ksq).with(ksq);
    let occ = board.occupied();
    let enemy = color.flip();
    let mut att = [0i32; 4];
    for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
        let mut bb = board.pieces(pt).intersect(board.color(enemy));
        while let Some(sq) = bb.pop_lsb() {
            let attacks = match pt {
                PieceType::Knight => knight_attacks(sq),
                PieceType::Bishop => bishop_attacks(sq, occ),
                PieceType::Rook => rook_attacks(sq, occ),
                PieceType::Queen => queen_attacks(sq, occ),
                _ => unreachable!(),
            };
            if !attacks.intersect(zone).is_empty() {
                att[nbrq_local(pt)] += 1;
            }
        }
    }
    let own_pawns = board.pieces(PieceType::Pawn).intersect(board.color(color));
    let kf = ksq.file() as usize;
    let mut holes = 0;
    for file in &FILE_MASKS[kf.saturating_sub(1)..=(kf + 1).min(7)] {
        if own_pawns.intersect(*file).is_empty() {
            holes += 1;
        }
    }
    (att, holes)
}

/// Build the white-relative feature trace of `board`: phase, fixed material
/// balance, and the dense mg/eg coefficient of every tunable weight.
fn trace(board: &Board) -> (i32, i32, Vec<i32>, Vec<i32>) {
    let mut mg = vec![0i32; P];
    let mut eg = vec![0i32; P];

    // Material balance (fixed; contributes equally to mg and eg).
    let mut mat = 0;
    for pt in
        [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen]
    {
        let v = PIECE_VALUE[pt.index()];
        mat += v * (count(board, pt, Color::White) - count(board, pt, Color::Black));
    }

    // PST (non-king): white adds, black subtracts; same coeff in mg and eg.
    for pt in
        [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen]
    {
        let base = PST + pt.index() * 64;
        let mut wb = board.pieces(pt).intersect(board.color(Color::White));
        while let Some(sq) = wb.pop_lsb() {
            let i = base + pst_index(Color::White, sq);
            mg[i] += 1;
            eg[i] += 1;
        }
        let mut bb = board.pieces(pt).intersect(board.color(Color::Black));
        while let Some(sq) = bb.pop_lsb() {
            let i = base + pst_index(Color::Black, sq);
            mg[i] -= 1;
            eg[i] -= 1;
        }
    }

    // Kings: KING_MG into mg only, KING_EG into eg only.
    let wk = pst_index(Color::White, king_square(board, Color::White));
    let bk = pst_index(Color::Black, king_square(board, Color::Black));
    mg[KMG + wk] += 1;
    eg[KEG + wk] += 1;
    mg[KMG + bk] -= 1;
    eg[KEG + bk] -= 1;

    // Bishop pair.
    let bp = (count(board, PieceType::Bishop, Color::White) >= 2) as i32
        - (count(board, PieceType::Bishop, Color::Black) >= 2) as i32;
    mg[BP] += bp;
    eg[BP + 1] += bp;

    // Rook files.
    let (ow, sw) = rook_counts(board, Color::White);
    let (ob, sb) = rook_counts(board, Color::Black);
    mg[RO] += ow - ob;
    eg[RO + 1] += ow - ob;
    mg[RS] += sw - sb;
    eg[RS + 1] += sw - sb;

    // Mobility.
    let mw = mobility_counts(board, Color::White);
    let mb = mobility_counts(board, Color::Black);
    for i in 0..4 {
        let c = mw[i] - mb[i];
        mg[MOB + i * 2] += c;
        eg[MOB + i * 2 + 1] += c;
    }

    // Pawn structure.
    let (dw, iw, pw) = pawn_counts(board, Color::White);
    let (db, ib, pb) = pawn_counts(board, Color::Black);
    mg[DBL] += dw - db;
    eg[DBL + 1] += dw - db;
    mg[ISO] += iw - ib;
    eg[ISO + 1] += iw - ib;
    for r in 0..8 {
        let c = pw[r] - pb[r];
        mg[PMG + r] += c;
        eg[PEG + r] += c;
    }

    // King safety (mg only). White-relative coeff of KAW[pt] is
    // (white pieces attacking black king) − (black pieces attacking white king).
    let (aw, hw) = king_safety_counts(board, Color::White); // attackers on White king, White holes
    let (ab, hb) = king_safety_counts(board, Color::Black); // attackers on Black king, Black holes
    for i in 0..4 {
        mg[KAW + i] += ab[i] - aw[i];
    }
    mg[SHIELD] += hb - hw;

    (game_phase(board), mat, mg, eg)
}

fn default_theta() -> Vec<f64> {
    let mut t = vec![0.0f64; P];
    for (pt, table) in PIECE_SQUARE_TABLE.iter().enumerate() {
        for (sq, &v) in table.iter().enumerate() {
            t[PST + pt * 64 + sq] = v as f64;
        }
    }
    for sq in 0..64 {
        t[KMG + sq] = KING_MG[sq] as f64;
        t[KEG + sq] = KING_EG[sq] as f64;
    }
    t[BP] = BISHOP_PAIR.mg as f64;
    t[BP + 1] = BISHOP_PAIR.eg as f64;
    t[RO] = ROOK_OPEN_FILE.mg as f64;
    t[RO + 1] = ROOK_OPEN_FILE.eg as f64;
    t[RS] = ROOK_SEMI_OPEN_FILE.mg as f64;
    t[RS + 1] = ROOK_SEMI_OPEN_FILE.eg as f64;
    for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
        let i = nbrq_local(pt);
        t[MOB + i * 2] = MOBILITY[pt.index()].mg as f64;
        t[MOB + i * 2 + 1] = MOBILITY[pt.index()].eg as f64;
    }
    t[DBL] = DOUBLED_PAWN.mg as f64;
    t[DBL + 1] = DOUBLED_PAWN.eg as f64;
    t[ISO] = ISOLATED_PAWN.mg as f64;
    t[ISO + 1] = ISOLATED_PAWN.eg as f64;
    for r in 0..8 {
        t[PMG + r] = PASSED_PAWN_MG[r] as f64;
        t[PEG + r] = PASSED_PAWN_EG[r] as f64;
    }
    for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
        t[KAW + nbrq_local(pt)] = KING_ATTACK_WEIGHT[pt.index()] as f64;
    }
    t[SHIELD] = PAWN_SHIELD_HOLE as f64;
    t
}

fn eval_white_f(theta: &[f64], s: &Sample) -> f64 {
    let (mut mg, mut eg) = (s.mat as f64, s.mat as f64);
    for &(idx, mc, ec) in &s.coeffs {
        mg += theta[idx as usize] * mc as f64;
        eg += theta[idx as usize] * ec as f64;
    }
    (mg * s.phase as f64 + eg * (PHASE_MAX - s.phase) as f64) / PHASE_MAX as f64
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn mse(theta: &[f64], k: f64, samples: &[Sample]) -> f64 {
    let sum: f64 = samples
        .iter()
        .map(|s| {
            let e = sigmoid(k * eval_white_f(theta, s));
            (s.result - e).powi(2)
        })
        .sum();
    sum / samples.len() as f64
}

/// Golden-section-ish 1-D search for the K minimising MSE at fixed weights.
fn solve_k(theta: &[f64], samples: &[Sample]) -> f64 {
    let (mut lo, mut hi) = (0.0f64, 0.02f64);
    for _ in 0..40 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if mse(theta, m1, samples) < mse(theta, m2, samples) {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    (lo + hi) / 2.0
}

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| "data/texel.txt".to_string());
    // L2 regularisation strength, pulling each weight toward its default. Restrains
    // params with little data (king-table squares, corner PST) — whose sparse
    // gradients Adam would otherwise amplify to extremes — while leaving
    // well-supported weights free to move. 0 = off.
    let lambda: f64 = env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let raw = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));

    // Parse + trace every position.
    let mut samples: Vec<Sample> = Vec::new();
    let mut consistency_failures = 0usize;
    let theta0 = default_theta();
    let theta0_i: Vec<i32> = theta0.iter().map(|&v| v as i32).collect();
    let evaluator = Material;

    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Two accepted formats:
        //   * EPD (zurichess quiet-labeled): `<4 FEN fields> c9 "1-0|0-1|1/2-1/2";`
        //   * our generator: `<6-field FEN> <result float>`
        let (fen, result): (String, f64) = if let Some((head, tail)) = line.split_once("c9") {
            // EPD: take the four board fields, append a dummy move clock (the eval
            // ignores it), and map the game result to White's POV.
            let f: Vec<&str> = head.split_whitespace().collect();
            let fen = format!("{} {} {} {} 0 1", f[0], f[1], f[2], f[3]);
            let result = if tail.contains("1/2") {
                0.5
            } else if tail.contains("1-0") {
                1.0
            } else {
                0.0
            };
            (fen, result)
        } else {
            let (fen, result) = line.rsplit_once(' ').expect("`FEN RESULT`");
            (fen.to_string(), result.parse().expect("result is a float"))
        };
        let board = Board::from_str(&fen).unwrap_or_else(|_| panic!("bad FEN: {fen}"));

        let (phase, mat, mg, eg) = trace(&board);

        // Consistency check: reconstruct the engine's integer eval from the trace
        // with default weights and compare to Material::evaluate exactly.
        let (mut img, mut ieg) = (mat, mat);
        for i in 0..P {
            img += theta0_i[i] * mg[i];
            ieg += theta0_i[i] * eg[i];
        }
        let white_rel = (img * phase + ieg * (PHASE_MAX - phase)) / PHASE_MAX;
        let engine = evaluator.evaluate(&board);
        let engine_white = if board.side_to_move == Color::White { engine } else { -engine };
        if white_rel != engine_white {
            if consistency_failures < 5 {
                eprintln!(
                    "CONSISTENCY MISMATCH: trace {white_rel} vs engine {engine_white} for {fen}"
                );
            }
            consistency_failures += 1;
        }

        let coeffs: Vec<(u16, i16, i16)> = (0..P)
            .filter(|&i| mg[i] != 0 || eg[i] != 0)
            .map(|i| (i as u16, mg[i] as i16, eg[i] as i16))
            .collect();
        samples.push(Sample { result, phase, mat, coeffs });
    }

    assert_eq!(
        consistency_failures, 0,
        "feature trace does not match engine eval on {consistency_failures} positions — \
         the tuner would optimise the wrong function"
    );
    eprintln!("loaded {} positions; consistency check passed (exact)", samples.len());

    // Train/validation split: every 10th position is held out.
    let (mut train, mut val): (Vec<Sample>, Vec<Sample>) = (Vec::new(), Vec::new());
    for (i, s) in samples.into_iter().enumerate() {
        if i % 10 == 0 {
            val.push(s);
        } else {
            train.push(s);
        }
    }

    let mut theta = default_theta();
    let k = solve_k(&theta, &train);
    eprintln!(
        "K = {k:.6}; start MSE train {:.6} val {:.6}",
        mse(&theta, k, &train),
        mse(&theta, k, &val)
    );

    // Adam gradient descent, full-batch (deterministic). The fixed material slots
    // have no coeff so they never move; passed-rank 0/7 likewise stay at default.
    let (b1, b2, eps, lr) = (0.9f64, 0.999f64, 1e-8f64, 1.0f64);
    let mut m = vec![0.0f64; P];
    let mut v = vec![0.0f64; P];
    let epochs = 4000;
    let n = train.len() as f64;
    for epoch in 1..=epochs {
        let mut grad = vec![0.0f64; P];
        for s in &train {
            let e = eval_white_f(&theta, s);
            let sig = sigmoid(k * e);
            let factor = -2.0 * (s.result - sig) * sig * (1.0 - sig) * k;
            let phase = s.phase as f64;
            for &(idx, mc, ec) in &s.coeffs {
                let d_e =
                    (mc as f64 * phase + ec as f64 * (PHASE_MAX as f64 - phase)) / PHASE_MAX as f64;
                grad[idx as usize] += factor * d_e;
            }
        }
        for i in 0..P {
            // Freeze the king piece-square tables at their (sound) #40 defaults:
            // only ~2 kings per position, clustered on a few squares, so most
            // king-table entries have too little data and Adam drives them to
            // ±1000 — a real blunder risk if the king ever visits such a square.
            // The other terms (many pieces per position) have ample data.
            if (KMG..KEG + 64).contains(&i) {
                continue;
            }
            // Data gradient + L2 pull toward the default weight.
            let g = grad[i] / n + 2.0 * lambda * (theta[i] - theta0[i]);
            m[i] = b1 * m[i] + (1.0 - b1) * g;
            v[i] = b2 * v[i] + (1.0 - b2) * g * g;
            let mhat = m[i] / (1.0 - b1.powi(epoch));
            let vhat = v[i] / (1.0 - b2.powi(epoch));
            theta[i] -= lr * mhat / (vhat.sqrt() + eps);
        }
        if epoch % 500 == 0 {
            eprintln!(
                "epoch {epoch}: MSE train {:.6} val {:.6}",
                mse(&theta, k, &train),
                mse(&theta, k, &val)
            );
        }
    }
    eprintln!("final MSE train {:.6} val {:.6}", mse(&theta, k, &train), mse(&theta, k, &val));

    emit(&theta);
}

/// Print the tuned weights as Rust constants, ready to paste into `src/eval.rs`.
fn emit(theta: &[f64]) {
    let r = |x: f64| x.round() as i32;
    let grid = |name: &str, base: usize| {
        println!("#[rustfmt::skip]\npub const {name}: [i32; 64] = [");
        for rank in 0..8 {
            print!("   ");
            for file in 0..8 {
                print!(" {:>4},", r(theta[base + rank * 8 + file]));
            }
            println!();
        }
        println!("];");
    };

    println!("// ===== tuned eval constants (issue #42) — paste into src/eval.rs =====");
    println!("#[rustfmt::skip]\npub const PIECE_SQUARE_TABLE: [[i32; 64]; 5] = [");
    for pt in 0..5 {
        println!("  [");
        for rank in 0..8 {
            print!("   ");
            for file in 0..8 {
                print!(" {:>4},", r(theta[PST + pt * 64 + rank * 8 + file]));
            }
            println!();
        }
        println!("  ],");
    }
    println!("];");
    grid("KING_MG", KMG);
    grid("KING_EG", KEG);

    let score = |name: &str, base: usize| {
        println!(
            "pub const {name}: Score = Score {{ mg: {}, eg: {} }};",
            r(theta[base]),
            r(theta[base + 1])
        );
    };
    score("BISHOP_PAIR", BP);
    score("ROOK_OPEN_FILE", RO);
    score("ROOK_SEMI_OPEN_FILE", RS);
    println!("pub const MOBILITY: [Score; 6] = [");
    println!("    Score {{ mg: 0, eg: 0 }}, // Pawn");
    for (label, pt) in [
        ("Knight", PieceType::Knight),
        ("Bishop", PieceType::Bishop),
        ("Rook", PieceType::Rook),
        ("Queen", PieceType::Queen),
    ] {
        let i = nbrq_local(pt);
        println!(
            "    Score {{ mg: {}, eg: {} }}, // {label}",
            r(theta[MOB + i * 2]),
            r(theta[MOB + i * 2 + 1])
        );
    }
    println!("    Score {{ mg: 0, eg: 0 }}, // King");
    println!("];");
    score("DOUBLED_PAWN", DBL);
    score("ISOLATED_PAWN", ISO);
    print!("pub const PASSED_PAWN_MG: [i32; 8] = [");
    for r8 in 0..8 {
        print!("{}, ", r(theta[PMG + r8]));
    }
    println!("];");
    print!("pub const PASSED_PAWN_EG: [i32; 8] = [");
    for r8 in 0..8 {
        print!("{}, ", r(theta[PEG + r8]));
    }
    println!("];");
    print!("pub const KING_ATTACK_WEIGHT: [i32; 6] = [0, ");
    for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
        print!("{}, ", r(theta[KAW + nbrq_local(pt)]));
    }
    println!("0];");
    println!("pub const PAWN_SHIELD_HOLE: i32 = {};", r(theta[SHIELD]));
}
