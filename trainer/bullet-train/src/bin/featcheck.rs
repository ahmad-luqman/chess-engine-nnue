//! Cross-validate `verify`'s feature mapping against bullet's OWN `Chess768`.
//!
//! `verify`'s sanity checks (material monotonicity, colour-mirror equality) are
//! all *flip-direction-blind*: a wrong square-flip direction cancels in the mirror
//! and never moves a material sign, yet it scrambles positional value. So this
//! check is the ground truth — it compares the exact feature index *sets* that
//! `verify::feature_index` produces (working in absolute white-POV coords) against
//! what `bullet_lib::Chess768` produces (working on its stm-relative `ChessBoard`),
//! on black-to-move asymmetric positions where the flip actually matters.
//!
//! Run: `cargo run -r -p bullet-train --features metal --bin featcheck`
//! (the backend feature is irrelevant here, but the crate needs one to build).

use bullet_lib::game::inputs::{Chess768, SparseInputType};
use bulletformat::ChessBoard;

/// EXACT copy of `verify/src/main.rs::feature_index` — the contract under test.
fn feature_index(perspective: usize, color: usize, pt: usize, sq: usize) -> usize {
    let relative_color = usize::from(color != perspective);
    let relative_sq = if perspective == 0 { sq } else { sq ^ 56 };
    [0, 384][relative_color] + 64 * pt + relative_sq
}

/// EXACT copy of `verify/src/main.rs::parse_fen` (placement + stm only).
fn parse_fen(fen: &str) -> (Vec<(usize, usize, usize)>, usize) {
    let mut fields = fen.split_whitespace();
    let placement = fields.next().expect("placement");
    let stm = matches!(fields.next(), Some("b")) as usize;
    let mut pieces = Vec::new();
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
                    o => panic!("bad piece {o}"),
                };
                pieces.push((color, pt, (rank * 8 + file) as usize));
                file += 1;
            }
        }
    }
    (pieces, stm)
}

fn mine(fen: &str) -> (Vec<usize>, Vec<usize>) {
    let (pieces, stm) = parse_fen(fen);
    let mut s: Vec<usize> = pieces.iter().map(|&(c, pt, sq)| feature_index(stm, c, pt, sq)).collect();
    let mut n: Vec<usize> =
        pieces.iter().map(|&(c, pt, sq)| feature_index(stm ^ 1, c, pt, sq)).collect();
    s.sort_unstable();
    n.sort_unstable();
    (s, n)
}

fn bullets(fen: &str) -> (Vec<usize>, Vec<usize>) {
    let board: ChessBoard = format!("{fen} | 0 | 0.5").parse().expect("parse ChessBoard");
    let (mut s, mut n) = (Vec::new(), Vec::new());
    Chess768.map_features(&board, |stm_idx, ntm_idx| {
        s.push(stm_idx);
        n.push(ntm_idx);
    });
    s.sort_unstable();
    n.sort_unstable();
    (s, n)
}

fn main() {
    let fens = [
        // white to move (the easy case)
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        // BLACK to move, asymmetric — the case the mirror test cannot validate
        "r3k2r/pp1bbppp/2n2n2/1B1pp3/3PP3/2N2N2/PPP2PPP/R1BQ1RK1 b kq - 0 1",
        "8/2k5/8/8/3Q4/8/5K2/8 b - - 0 1",
        // white to move, asymmetric
        "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 1",
    ];
    let mut ok = true;
    for fen in fens {
        let (ms, mn) = mine(fen);
        let (bs, bn) = bullets(fen);
        let stm_ok = ms == bs;
        let ntm_ok = mn == bn;
        println!("stm={} ntm={}  {}", stm_ok, ntm_ok, fen);
        if !(stm_ok && ntm_ok) {
            ok = false;
            println!("  MINE stm {ms:?}\n  BULL stm {bs:?}\n  MINE ntm {mn:?}\n  BULL ntm {bn:?}");
        }
    }
    if ok {
        println!("\nOK: verify's feature mapping matches bullet's Chess768 exactly.");
    } else {
        eprintln!("\nMISMATCH: verify's mapping diverges from bullet — fix before committing.");
        std::process::exit(1);
    }
}
