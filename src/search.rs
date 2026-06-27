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
//! **Iterative deepening (issue #21).** Rather than searching straight to a
//! target depth, [`search_timed`] searches depth 1, then 2, then 3, … keeping
//! the best move from the last *completed* depth. This sounds wasteful but isn't
//! — each depth is far cheaper than the next — and it is what makes time
//! management possible: when the clock runs out we always have a fully-searched
//! move in hand, and we simply stop before starting a depth we can't finish.
//!
//! Transposition tables, move ordering, and quiescence come in Phase 2.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::board::Board;
use crate::eval::{Evaluator, Material};
use crate::movegen::{generate_legal, in_check};
use crate::moves::Move;
use crate::timeman::Budget;

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

/// Mutable state threaded through a single search: the evaluator, the node
/// counter, and the two ways a search can be cut short — a wall-clock deadline
/// and a stop flag.
struct SearchContext {
    evaluator: Material,
    /// When set, the search aborts once `Instant::now()` passes this point.
    /// `None` means "no time limit" (depth-limited or unbounded search).
    deadline: Option<Instant>,
    /// Cooperative stop signal. Phase 1 search is synchronous, so nothing sets
    /// this mid-search yet — it is the seam a future search thread (and UCI
    /// `stop`) will use. We still honour it so the mechanism is testable.
    stop: Arc<AtomicBool>,
    nodes: u64,
    /// Latched once a deadline or stop is observed: every node above unwinds
    /// without doing more work, and the partial iteration is discarded.
    aborted: bool,
}

impl SearchContext {
    /// A context with no limits: used by the fixed-depth [`search`] and as the
    /// guaranteed-move fallback. Its stop flag is never raised.
    fn unbounded() -> SearchContext {
        SearchContext {
            evaluator: Material,
            deadline: None,
            stop: Arc::new(AtomicBool::new(false)),
            nodes: 0,
            aborted: false,
        }
    }

    /// Should the search stop now? The stop flag is a cheap atomic load, checked
    /// every node; the deadline check calls `Instant::now()`, which is costlier,
    /// so it is only sampled every 2048 nodes. Once either fires, `aborted`
    /// stays latched.
    fn should_stop(&mut self) -> bool {
        if self.aborted {
            return true;
        }
        if self.stop.load(Ordering::Relaxed) {
            self.aborted = true;
            return true;
        }
        if self.nodes & 2047 == 0 {
            if let Some(deadline) = self.deadline {
                if Instant::now() >= deadline {
                    self.aborted = true;
                }
            }
        }
        self.aborted
    }
}

/// Search `board` to a fixed `depth`, ignoring the clock, and return the best
/// move with its score. Non-destructive (it searches a clone). `depth` is
/// clamped to at least 1. This is the fixed-depth entry used by tests and as the
/// fallback that guarantees [`search_timed`] always returns a legal move.
pub fn search(board: &Board, depth: u32) -> SearchResult {
    let mut board = board.clone();
    let mut ctx = SearchContext::unbounded();
    run_root(&mut board, &mut ctx, depth.max(1))
}

/// Iterative-deepening search under a time [`Budget`]. Searches depth 1, 2, 3,
/// … updating the result only when an iteration *completes*; an iteration cut
/// short by the deadline or stop flag is thrown away. `on_depth` is invoked
/// after each completed depth (the UCI layer turns it into an `info` line).
/// `start` is the search's origin instant, shared with the budget's deadline.
///
/// Guarantees a legal move whenever one exists: if the clock is too tight for
/// even one full iteration, it falls back to an unabortable depth-1 search.
pub fn search_timed(
    board: &Board,
    budget: &Budget,
    stop: Arc<AtomicBool>,
    start: Instant,
    on_depth: &mut dyn FnMut(&SearchResult, Duration),
) -> SearchResult {
    let mut board = board.clone();
    let mut ctx = SearchContext {
        evaluator: Material,
        deadline: budget.deadline,
        stop,
        nodes: 0,
        aborted: false,
    };

    let mut best = SearchResult { best_move: Move::NONE, score: 0, depth: 0, nodes: 0 };
    for depth in 1..=budget.max_depth {
        let result = run_root(&mut board, &mut ctx, depth);
        if ctx.aborted {
            break; // partial iteration — discard it, keep the last good `best`.
        }
        best = result;
        on_depth(&best, start.elapsed());

        // Soft stop: if we've already spent over half the budget, the next
        // iteration (which costs several times this one) almost certainly won't
        // finish, so stop now and save the remaining time.
        if let Some(deadline) = ctx.deadline {
            let total = deadline.saturating_duration_since(start);
            if start.elapsed() * 2 > total {
                break;
            }
        }
    }

    if best.best_move == Move::NONE {
        // Either the position is terminal, or the clock was too tight to finish
        // even depth 1. An unabortable depth-1 search resolves both: it returns a
        // legal move when one exists and Move::NONE only at checkmate/stalemate.
        best = search(&board, 1);
    }
    best
}

/// One fixed-depth search from the root. Like an interior negamax node but it
/// remembers which move achieved alpha and takes no beta cutoff (there is no
/// parent to cut to — we want the genuinely best move). Bails out cleanly if the
/// context aborts mid-iteration.
fn run_root(board: &mut Board, ctx: &mut SearchContext, depth: u32) -> SearchResult {
    let moves = generate_legal(board);
    if moves.is_empty() {
        let score = terminal_score(board, 0);
        return SearchResult { best_move: Move::NONE, score, depth, nodes: ctx.nodes };
    }

    let mut best_move = Move::NONE;
    let mut alpha = -INF;
    for mv in moves {
        let undo = board.make_move(mv);
        let score = -negamax(board, ctx, depth - 1, -INF, -alpha, 1);
        board.unmake_move(mv, undo);

        if ctx.aborted {
            break; // the score is from an interrupted subtree — don't trust it.
        }
        if best_move == Move::NONE || score > alpha {
            alpha = score;
            best_move = mv;
        }
    }

    SearchResult { best_move, score: alpha, depth, nodes: ctx.nodes }
}

/// Negamax with fail-hard alpha-beta. Returns the position's score from the
/// side-to-move's perspective, searched to `depth` plies. `ply` is the distance
/// from the root, used to make mate scores prefer faster mates.
fn negamax(board: &mut Board, ctx: &mut SearchContext, depth: u32, mut alpha: i32, beta: i32, ply: i32) -> i32 {
    ctx.nodes += 1;
    if ctx.should_stop() {
        return 0; // value ignored: the caller discards aborted iterations.
    }

    // Generate first so we can distinguish terminal nodes (no legal moves) from
    // a quiet leaf. This must come before the depth check: a checkmate at depth
    // 0 is a mate, not whatever the static eval happens to say.
    let moves = generate_legal(board);
    if moves.is_empty() {
        return terminal_score(board, ply);
    }

    if depth == 0 {
        return ctx.evaluator.evaluate(board);
    }

    for mv in moves {
        let undo = board.make_move(mv);
        let score = -negamax(board, ctx, depth - 1, -beta, -alpha, ply + 1);
        board.unmake_move(mv, undo);

        if ctx.aborted {
            return alpha; // unwind promptly; the result will be discarded.
        }
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
        // After RxQ White is up a rook's worth of material (≈+500); PSTs nudge
        // the exact number, so just assert a clearly-winning score.
        assert!(result.score > 400, "expected a winning score, got {}", result.score);
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

    /// All legal moves in `b`, as UCI strings — for "is the move legal?" checks.
    fn legal_strings(b: &Board) -> Vec<String> {
        generate_legal(b).iter().map(|m| m.to_string()).collect()
    }

    #[test]
    fn iterative_deepening_completes_each_depth_in_order() {
        let b = board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let now = Instant::now();
        let budget = Budget { deadline: None, max_depth: 3 };
        let mut depths = Vec::new();
        let result = search_timed(
            &b,
            &budget,
            Arc::new(AtomicBool::new(false)),
            now,
            &mut |r: &SearchResult, _| depths.push(r.depth),
        );
        assert_eq!(depths, vec![1, 2, 3], "should report depths 1..=3 as they complete");
        assert_eq!(result.depth, 3);
        assert!(legal_strings(&b).contains(&result.best_move.to_string()));
    }

    #[test]
    fn search_timed_respects_its_deadline() {
        // A 100ms budget on a normal middlegame must return a legal move without
        // running away. The tolerance is generous so it won't flake under load.
        let b = board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1");
        let now = Instant::now();
        let budget = Budget {
            deadline: Some(now + Duration::from_millis(100)),
            max_depth: 64,
        };
        let result = search_timed(&b, &budget, Arc::new(AtomicBool::new(false)), now, &mut |_, _| {});
        assert!(now.elapsed() < Duration::from_millis(2000), "overran budget: {:?}", now.elapsed());
        assert!(legal_strings(&b).contains(&result.best_move.to_string()));
    }

    #[test]
    fn a_preset_stop_still_returns_a_legal_move() {
        // The must-not-happen case: the search is told to stop before it does any
        // work (every iteration aborts), yet a legal move still exists. The
        // depth-1 fallback must produce one rather than emitting a null move.
        let b = board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let now = Instant::now();
        let budget = Budget { deadline: None, max_depth: 64 };
        let stop = Arc::new(AtomicBool::new(true)); // already stopped
        let result = search_timed(&b, &budget, stop, now, &mut |_, _| {});
        assert_ne!(result.best_move, Move::NONE, "must not forfeit when moves exist");
        assert!(legal_strings(&b).contains(&result.best_move.to_string()));
    }
}
