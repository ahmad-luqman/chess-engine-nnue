//! Search: negamax with alpha-beta pruning.
//!
//! Search is where the engine actually *thinks*. Given a position it walks the
//! game tree to a fixed depth, scoring leaves with the [evaluator](crate::eval)
//! and backing those scores up to choose a move. Two ideas do the heavy lifting:
//!
//! **Negamax.** Instead of separate "maximize for White / minimize for Black"
//! logic, we exploit `score(me) == -score(opponent)`: every node maximizes from
//! the side-to-move's seat, and the recursion negates the child's score. The
//! evaluator already returns side-to-move-relative scores, so the whole tree
//! speaks one language.
//!
//! **Alpha-beta.** `alpha` is the best score the side to move has already
//! guaranteed; `beta` is the best the opponent will allow. When a move's score
//! reaches `beta`, the opponent would never let us get here — they had a better
//! option earlier — so we stop searching this node (a *cutoff*). This prunes
//! large parts of the tree without changing the result. The windows flip and
//! negate on the way down: `negamax(-beta, -alpha)`.
//!
//! Issue #19 is fixed-depth. Iterative deepening and time management arrive in
//! issue #21; transposition tables, move ordering, and quiescence in Phase 2.

use crate::board::Board;
use crate::eval::{Evaluator, Material};
use crate::movegen::{generate_legal, in_check};
use crate::moves::Move;

/// A score larger than any real evaluation — used as the initial alpha/beta
/// bounds (an open window). Must exceed every score the tree can produce,
/// including mate scores, so it never clips a real value.
pub const INF: i32 = 32_000;

/// Score of being checkmated *at this node*. A mate found `n` plies deeper scores
/// `MATE - n`, so shallower (faster) mates score higher — the engine prefers to
/// deliver mate sooner and to delay being mated. `INF > MATE` guarantees mate
/// scores sit inside the search window.
pub const MATE: i32 = 31_000;

/// The outcome of a search: the move to play and why.
#[derive(Clone, Copy, Debug)]
pub struct SearchResult {
    /// The best move found, or [`Move::NONE`] when the position is terminal
    /// (checkmate or stalemate — there is nothing to play).
    pub best_move: Move,
    /// Score of `best_move`, in centipawns, from the side-to-move's perspective.
    /// Mate scores are near ±[`MATE`].
    pub score: i32,
    /// The depth this result was searched to.
    pub depth: u32,
    /// Nodes visited — the unit of search cost, and what `nps` is computed from.
    pub nodes: u64,
}

/// Search `board` to a fixed `depth` and return the best move with its score.
///
/// Non-destructive: it searches a clone, so the caller's board is untouched.
/// `depth` is clamped to at least 1 (a depth-0 "search" would just be a static
/// eval with no move to return).
pub fn search(board: &Board, depth: u32) -> SearchResult {
    let depth = depth.max(1);
    let evaluator = Material;
    let mut board = board.clone();
    let mut nodes: u64 = 0;

    let moves = generate_legal(&board);
    if moves.is_empty() {
        // Terminal at the root: no move to make. Report the mate/stalemate score
        // from the side-to-move's seat (negative = the mover is the one mated).
        let score = terminal_score(&board, 0);
        return SearchResult { best_move: Move::NONE, score, depth, nodes: 1 };
    }

    // Root is an open-window negamax that also remembers which move achieved
    // alpha. We don't take beta cutoffs at the root: there's no parent to cut to,
    // and we want the genuinely best move, not merely one that beats a bound.
    let mut best_move = Move::NONE;
    let mut alpha = -INF;
    for mv in moves {
        let undo = board.make_move(mv);
        let score = -negamax(&mut board, &evaluator, depth - 1, -INF, -alpha, 1, &mut nodes);
        board.unmake_move(mv, undo);

        if best_move == Move::NONE || score > alpha {
            alpha = score;
            best_move = mv;
        }
    }

    SearchResult { best_move, score: alpha, depth, nodes }
}

/// Negamax with fail-hard alpha-beta. Returns the position's score from the
/// side-to-move's perspective, searched to `depth` plies. `ply` is the distance
/// from the root, used to make mate scores prefer faster mates.
fn negamax(
    board: &mut Board,
    evaluator: &Material,
    depth: u32,
    mut alpha: i32,
    beta: i32,
    ply: i32,
    nodes: &mut u64,
) -> i32 {
    *nodes += 1;

    // Generate first so we can distinguish terminal nodes (no legal moves) from
    // a quiet leaf. This must come before the depth check: a checkmate at depth
    // 0 is a mate, not whatever the static eval happens to say.
    let moves = generate_legal(board);
    if moves.is_empty() {
        return terminal_score(board, ply);
    }

    if depth == 0 {
        return evaluator.evaluate(board);
    }

    for mv in moves {
        let undo = board.make_move(mv);
        let score = -negamax(board, evaluator, depth - 1, -beta, -alpha, ply + 1, nodes);
        board.unmake_move(mv, undo);

        if score >= beta {
            // The opponent has a refutation good enough that they'd avoid this
            // whole line; no need to look further (fail-hard cutoff).
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}

/// Score of a node that has no legal moves: checkmate if the side to move is in
/// check (loss, adjusted by distance so faster mates score higher), else
/// stalemate (a draw). Always from the side-to-move's perspective.
fn terminal_score(board: &Board, ply: i32) -> i32 {
    if in_check(board, board.side_to_move) {
        -MATE + ply
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::str::FromStr;

    fn board(fen: &str) -> Board {
        Board::from_str(fen).unwrap()
    }

    #[test]
    fn finds_back_rank_mate_in_one() {
        // White rook delivers Ra8#: the Black king on g8 is boxed in by its own
        // f7/g7/h7 pawns and the rook covers the 8th rank.
        let b = board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1");
        let result = search(&b, 1);
        assert_eq!(result.best_move.to_string(), "a1a8");
        // Mate-in-1 is delivered at ply 1, so the score is exactly MATE - 1.
        assert_eq!(result.score, MATE - 1);
    }

    #[test]
    fn grabs_a_hanging_queen() {
        // The black queen on d4 is defended by nothing; the rook on d1 takes it.
        let b = board("4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1");
        let result = search(&b, 1);
        assert_eq!(result.best_move.to_string(), "d1d4");
        // After RxQ the resulting balance is White's lone rook vs nothing: +500.
        // (The capture nets the queen, but White already had the rook.)
        assert_eq!(result.score, 500);
    }

    #[test]
    fn stalemate_is_a_draw_with_no_move() {
        // Classic Q+K vs K stalemate: Black to move, king on h8 not in check but
        // with no legal move.
        let b = board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
        let result = search(&b, 3);
        assert_eq!(result.best_move, Move::NONE);
        assert_eq!(result.score, 0);
    }

    #[test]
    fn checkmated_side_reports_no_move_and_a_mate_score() {
        // Fool's mate reached: White is checkmated, Black queen on h4 + pawns.
        let b = board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
        let result = search(&b, 2);
        assert_eq!(result.best_move, Move::NONE);
        // Side to move (White) is mated: a large negative score.
        assert!(result.score <= -(MATE - 10), "expected a mate score, got {}", result.score);
    }

    #[test]
    fn deeper_search_visits_more_nodes() {
        let b = board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let shallow = search(&b, 2).nodes;
        let deep = search(&b, 3).nodes;
        assert!(deep > shallow, "depth 3 ({deep}) should visit more than depth 2 ({shallow})");
    }

    #[test]
    fn prefers_the_faster_mate() {
        // Mate-distance encoding means the nearest mate scores highest. Searching
        // a mate-in-1 position deeper (depth 3) still reports MATE - 1 — the
        // immediate mate — rather than a slower MATE - 3 line, because the
        // shallower mate wins the max.
        let b = board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1");
        assert_eq!(search(&b, 3).score, MATE - 1);
    }
}
