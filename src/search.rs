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
//! **Transposition table (issue #24).** Searched positions are cached in a
//! [`TranspositionTable`] keyed by Zobrist hash: a position reached again by a
//! different move order can reuse the stored score (an immediate cutoff when it
//! was searched deep enough) instead of being re-searched. The table is borrowed
//! by the search so it persists across iterative-deepening iterations and, via
//! UCI, across moves.
//!
//! **Move ordering (issue #25).** Moves are searched best-first — TT move, then
//! captures by MVV-LVA, killers, and history — so beta cutoffs fire early; this
//! is the single biggest practical search speedup.
//!
//! **Quiescence (issue #26).** At the leaves [`qsearch`] keeps resolving captures
//! until the position is quiet before evaluating, removing the horizon effect.
//!
//! **Draw detection (issue #28).** Before searching a node the engine scores
//! threefold repetition and the fifty-move rule as draws, so it stops shuffling
//! in dead-drawn endgames and recognizes a line is not winning. The repetition
//! key stack lives on the search context (seeded from the game history), not on
//! the `Board`, which stays a pure value.

use std::cmp::Reverse;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use crate::board::Board;
use crate::eval::{Evaluator, Material, PIECE_VALUE};
use crate::movegen::{generate_legal, in_check};
use crate::moves::Move;
use crate::timeman::Budget;
use crate::tt::{Bound, TranspositionTable};
use crate::types::{Color, PieceType};

/// A score larger than any real evaluation — used as the initial alpha/beta
/// bounds (an open window). Must exceed every score the tree can produce,
/// including mate scores, so it never clips a real value.
pub const INF: i32 = 32_000;

/// Score of being checkmated *at this node*. A mate found `n` plies deeper scores
/// `MATE - n`, so shallower (faster) mates score higher — the engine prefers to
/// deliver mate sooner and to delay being mated. `INF > MATE` guarantees mate
/// scores sit inside the search window.
pub const MATE: i32 = 31_000;

/// Default transposition-table size in MB when none is set via UCI.
pub const DEFAULT_TT_MB: usize = 16;

/// A draw — repetition or the fifty-move rule — scores zero (issue #28).
const DRAW: i32 = 0;

/// Whether the current position (`hash`) has occurred earlier in the line. Only
/// the last `halfmove` plies can hold a repeat — a pawn move or capture is
/// irreversible and resets the clock, so no earlier position can recur across one
/// — so we scan back only that far. A single in-tree repeat is treated as a draw:
/// it's enough to stop the search wasting effort shuffling. `rep` holds the keys
/// of every ancestor position (game history + the search path above this node).
fn is_repetition(hash: u64, halfmove: u16, rep: &[u64]) -> bool {
    let window = halfmove as usize;
    let start = rep.len().saturating_sub(window);
    rep[start..].contains(&hash)
}

/// Any score at or beyond this magnitude encodes a forced mate (`MATE - plies`).
/// The margin comfortably exceeds the deepest reachable ply, so it never
/// misclassifies a positional score as a mate.
const MATE_BOUND: i32 = MATE - 1000;

/// Adjust a mate score for *storage* in the TT. A mate score is "mate in N plies
/// **from this node**", but the same position can be probed at a different
/// distance from the root, so we store it relative to the node (add `ply` going
/// in) and undo that on the way out — otherwise a cached mate would claim the
/// wrong distance. Non-mate scores pass through unchanged.
fn score_to_tt(score: i32, ply: i32) -> i32 {
    if score >= MATE_BOUND {
        score + ply
    } else if score <= -MATE_BOUND {
        score - ply
    } else {
        score
    }
}

/// Inverse of [`score_to_tt`]: re-anchor a stored mate score to the probing node.
fn score_from_tt(score: i32, ply: i32) -> i32 {
    if score >= MATE_BOUND {
        score - ply
    } else if score <= -MATE_BOUND {
        score + ply
    } else {
        score
    }
}

// ── Move ordering (issue #25) ───────────────────────────────────────────────
//
// Alpha-beta's pruning power comes almost entirely from searching the best move
// first: a good first guess collapses the effective branching factor. We score
// each move and search highest first. The bands, in priority order:
//
//   TT move  >  captures/promotions (MVV-LVA)  >  killers  >  history (quiets)
//
// The bands are spaced far enough apart that no in-band score can cross into a
// neighbour (history is bounded well below the killer band — see the cap below).

/// Maximum search ply we track killer moves for; deeper plies simply skip them.
const MAX_PLY: usize = 128;

/// The transposition-table best move — the single strongest ordering signal.
const TT_BONUS: i32 = 1 << 28;
/// Base for captures and promotions, lifted above all quiet moves.
const CAPTURE_BONUS: i32 = 1 << 24;
/// The two killer slots, just below captures and above history.
const KILLER_BONUS: [i32; 2] = [1 << 23, (1 << 23) - 1];
/// History scores are clamped below the killer band so bands never overlap.
const HISTORY_MAX: i32 = (1 << 23) - 2;

/// Butterfly history: `[from][side][to]` cutoff counters for quiet moves. Boxed
/// because it is 32 KiB — too big to sit inline in a stack-allocated context.
type History = [[[i32; 64]; 2]; 64];

/// Killer moves: two quiet moves per ply that recently caused a beta cutoff.
type Killers = [[Move; 2]; MAX_PLY];

/// Order score for `mv` (higher searched first). Reads the position for MVV-LVA
/// victim/attacker values and the search's killer/history tables for quiets.
fn move_score(
    board: &Board,
    mv: Move,
    tt_move: Move,
    killers: &[Move; 2],
    history: &History,
    side: Color,
) -> i32 {
    if mv == tt_move {
        return TT_BONUS;
    }

    let promo = mv.promotion_piece().map_or(0, |pt| PIECE_VALUE[pt.index()]);
    if mv.is_capture() || promo != 0 {
        // MVV-LVA: most valuable victim first, least valuable attacker breaking
        // ties (so PxQ outranks QxQ). An en-passant victim is always a pawn.
        let victim = if mv.is_en_passant() {
            PIECE_VALUE[0]
        } else if mv.is_capture() {
            board.piece_on(mv.to()).map_or(0, |p| PIECE_VALUE[p.piece_type.index()])
        } else {
            0
        };
        let attacker = board.piece_on(mv.from()).map_or(0, |p| PIECE_VALUE[p.piece_type.index()]);
        return CAPTURE_BONUS + victim * 16 - attacker + promo;
    }

    // Quiet move: killers first, then the history score.
    if mv == killers[0] {
        KILLER_BONUS[0]
    } else if mv == killers[1] {
        KILLER_BONUS[1]
    } else {
        history[mv.from().0 as usize][side.index()][mv.to().0 as usize]
    }
}

/// Record a quiet beta-cutoff move as a killer for `ply`, keeping the two most
/// recent distinct ones (most recent in slot 0).
fn store_killer(killers: &mut [Move; 2], mv: Move) {
    if killers[0] != mv {
        killers[1] = killers[0];
        killers[0] = mv;
    }
}

/// Reward a quiet move that caused a beta cutoff, by depth² (deeper cutoffs are
/// more valuable), clamped so the score stays within its ordering band.
fn update_history(history: &mut History, side: Color, mv: Move, depth: u32) {
    let slot = &mut history[mv.from().0 as usize][side.index()][mv.to().0 as usize];
    *slot = (*slot + (depth * depth) as i32).min(HISTORY_MAX);
}

/// Sort `moves` highest-score first (see [`move_score`]). `sort_by_cached_key`
/// evaluates each move's score once, so the position lookups aren't repeated.
fn order_moves(
    moves: &mut [Move],
    board: &Board,
    tt_move: Move,
    killers: &[Move; 2],
    history: &History,
    side: Color,
) {
    moves.sort_by_cached_key(|&mv| Reverse(move_score(board, mv, tt_move, killers, history, side)));
}

/// Whether `mv` is a capture or promotion — a "tactical" move, excluded from the
/// killer/history heuristics, which track *quiet* cutoffs only.
fn is_tactical(mv: Move) -> bool {
    mv.is_capture() || mv.promotion_piece().is_some()
}

/// Whether `side` has any non-pawn, non-king material (a knight, bishop, rook, or
/// queen). This is the zugzwang guard for null-move pruning (#36): in king-and-pawn
/// endings passing can be *better* than any legal move, so a null move there would
/// prune lines that are actually losing. With a real piece on the board that
/// pathology is vanishingly rare, so NMP is only attempted when this holds.
fn has_non_pawn_material(board: &Board, side: Color) -> bool {
    let ours = board.color(side);
    let non_pawn = board
        .pieces(PieceType::Knight)
        .union(board.pieces(PieceType::Bishop))
        .union(board.pieces(PieceType::Rook))
        .union(board.pieces(PieceType::Queen));
    !non_pawn.intersect(ours).is_empty()
}

/// Late-move-reduction table (issue #37): plies to shave off a late, quiet move's
/// search, indexed `[depth][move_index]` (both clamped to 63). Reductions grow
/// with depth and with how late the move is — `R = floor(0.75 + ln(d)·ln(i)/2)`.
/// Built once at startup via [`init`].
static LMR_TABLE: OnceLock<[[u8; 64]; 64]> = OnceLock::new();

fn lmr_table() -> &'static [[u8; 64]; 64] {
    LMR_TABLE.get_or_init(|| {
        let mut t = [[0u8; 64]; 64];
        for (depth, row) in t.iter_mut().enumerate().skip(1) {
            for (moves, cell) in row.iter_mut().enumerate().skip(1) {
                let r = 0.75 + (depth as f64).ln() * (moves as f64).ln() / 2.0;
                *cell = r as u8; // floor for the positive value
            }
        }
        t
    })
}

/// Base LMR reduction (plies) for a late quiet move at this `depth` / `move_index`,
/// before the caller clamps it to keep the reduced depth ≥ 1.
fn lmr_reduction(depth: u32, move_index: usize) -> u32 {
    lmr_table()[(depth as usize).min(63)][move_index.min(63)] as u32
}

/// Force the one-time table builds (LMR here; magics in [`crate::magic`]) at
/// process startup, so the first search doesn't pay them on its own clock — see
/// the magic-init lesson in `src/magic.rs`. Idempotent; called from `main`.
pub fn init() {
    lmr_table();
    crate::magic::init();
}

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
struct SearchContext<'a> {
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
    /// Shared transposition table, borrowed so it persists across iterative-
    /// deepening iterations and (via UCI) across moves.
    tt: &'a mut TranspositionTable,
    /// Killer moves per ply (issue #25), fresh each search.
    killers: Killers,
    /// Butterfly history counters (issue #25), accumulated across this search's
    /// iterative-deepening iterations.
    history: Box<History>,
    /// Zobrist keys of every position above the current node — the game history
    /// (from `position … moves`) followed by the keys pushed along the search
    /// path. Used for repetition detection (issue #28).
    rep: Vec<u64>,
    /// PVS diagnostics (issue #34): how many null-window scout searches we ran,
    /// and how many of those failed high and forced a full-window re-search. A
    /// low `researches / scouts` ratio is the signal that move ordering is doing
    /// its job — the scout proves most moves worse without a re-search. Counts
    /// only; they never affect the result.
    pvs_scouts: u64,
    pvs_researches: u64,
    /// LMR diagnostics (issue #37): how many late quiet moves we searched at
    /// reduced depth, and how many of those failed high and forced a full-depth
    /// re-search. A low `researches / reductions` ratio means the reductions are
    /// safe (rarely wrong); a high one means we're over-reducing. Counts only.
    lmr_reductions: u64,
    lmr_researches: u64,
    /// NMP diagnostics (issue #36): how many nodes attempted a null move, and how
    /// many of those failed high (`>= beta`) and pruned the node. A high
    /// `cutoffs / attempts` ratio is the signal that null-move pruning is paying
    /// off. Counts only; they never affect the result.
    null_attempts: u64,
    null_cutoffs: u64,
}

impl<'a> SearchContext<'a> {
    /// A context with no time/stop limits, borrowing `tt`: used by the
    /// fixed-depth [`search`] and as the guaranteed-move fallback.
    fn unbounded(tt: &'a mut TranspositionTable) -> SearchContext<'a> {
        SearchContext {
            evaluator: Material,
            deadline: None,
            stop: Arc::new(AtomicBool::new(false)),
            nodes: 0,
            aborted: false,
            tt,
            killers: [[Move::NONE; 2]; MAX_PLY],
            history: Box::new([[[0; 64]; 2]; 64]),
            rep: Vec::new(),
            pvs_scouts: 0,
            pvs_researches: 0,
            lmr_reductions: 0,
            lmr_researches: 0,
            null_attempts: 0,
            null_cutoffs: 0,
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
    let mut tt = TranspositionTable::new(DEFAULT_TT_MB);
    search_with_tt(board, depth, &mut tt)
}

/// Fixed-depth search reusing a caller-owned transposition table. Lets tests
/// search a position twice (reusing the populated TT) to show the TT changes
/// speed, not the chosen move, and to run a TT-free reference via
/// [`TranspositionTable::disabled`].
pub fn search_with_tt(board: &Board, depth: u32, tt: &mut TranspositionTable) -> SearchResult {
    let mut board = board.clone();
    tt.new_search();
    let mut ctx = SearchContext::unbounded(tt);
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
    tt: &mut TranspositionTable,
    game_history: &[u64],
    on_depth: &mut dyn FnMut(&SearchResult, Duration),
) -> SearchResult {
    let mut board = board.clone();
    tt.new_search();
    // Seed repetition history with the keys of the positions the game already
    // passed through (so a draw the game is about to repeat is seen pre-search).
    let mut rep = Vec::with_capacity(game_history.len() + 64);
    rep.extend_from_slice(game_history);
    let mut ctx = SearchContext {
        evaluator: Material,
        deadline: budget.deadline,
        stop,
        nodes: 0,
        aborted: false,
        tt,
        killers: [[Move::NONE; 2]; MAX_PLY],
        history: Box::new([[[0; 64]; 2]; 64]),
        rep,
        pvs_scouts: 0,
        pvs_researches: 0,
        lmr_reductions: 0,
        lmr_researches: 0,
        null_attempts: 0,
        null_cutoffs: 0,
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
        // Reuse the context (and its TT) but clear the abort latch and limits so
        // this last-resort iteration always completes.
        ctx.aborted = false;
        ctx.deadline = None;
        ctx.stop = Arc::new(AtomicBool::new(false));
        best = run_root(&mut board, &mut ctx, 1);
    }
    best
}

/// One fixed-depth search from the root. Like an interior negamax node but it
/// remembers which move achieved alpha and takes no beta cutoff (there is no
/// parent to cut to — we want the genuinely best move). Bails out cleanly if the
/// context aborts mid-iteration.
fn run_root(board: &mut Board, ctx: &mut SearchContext<'_>, depth: u32) -> SearchResult {
    let mut moves = generate_legal(board);
    if moves.is_empty() {
        let score = terminal_score(board, 0);
        return SearchResult { best_move: Move::NONE, score, depth, nodes: ctx.nodes };
    }

    // Search the previous iteration's best move (the TT move) first — the single
    // biggest iterative-deepening speedup — then captures, killers, history.
    let side = board.side_to_move;
    let tt_move = ctx.tt.probe(board.hash).map_or(Move::NONE, |e| e.best_move);
    order_moves(&mut moves, board, tt_move, &ctx.killers[0], &ctx.history, side);

    let mut best_move = Move::NONE;
    let mut alpha = -INF;
    for mv in moves {
        ctx.rep.push(board.hash); // this (root) position becomes an ancestor below
        let undo = board.make_move(mv);
        let score = if best_move == Move::NONE {
            // First (best-ordered) move: a full-window search establishes the PV
            // baseline. (-INF, -alpha) is the original root window.
            -negamax(board, ctx, depth - 1, -INF, -alpha, 1, true)
        } else {
            // PVS scout: prove later moves worse than the PV with a null window.
            // The root takes no beta cutoff (its beta is +INF), so any fail-high
            // is a genuine new best — re-search it full-width for an exact score.
            ctx.pvs_scouts += 1;
            let s = -negamax(board, ctx, depth - 1, -alpha - 1, -alpha, 1, true);
            if !ctx.aborted && s > alpha {
                ctx.pvs_researches += 1;
                -negamax(board, ctx, depth - 1, -INF, -alpha, 1, true)
            } else {
                s
            }
        };
        board.unmake_move(mv, undo);
        ctx.rep.pop();

        if ctx.aborted {
            break; // the score is from an interrupted subtree — don't trust it.
        }
        if best_move == Move::NONE || score > alpha {
            alpha = score;
            best_move = mv;
        }
    }

    // Cache the root result (a full-window search, so exact) for the next
    // iteration and future moves. Never store a partial result from an aborted
    // iteration — its score is untrustworthy.
    if !ctx.aborted && best_move != Move::NONE {
        ctx.tt.store(board.hash, best_move, score_to_tt(alpha, 0), depth as u8, Bound::Exact);
    }

    SearchResult { best_move, score: alpha, depth, nodes: ctx.nodes }
}

/// Negamax with fail-hard alpha-beta. Returns the position's score from the
/// side-to-move's perspective, searched to `depth` plies. `ply` is the distance
/// from the root, used to make mate scores prefer faster mates.
fn negamax(
    board: &mut Board,
    ctx: &mut SearchContext<'_>,
    depth: u32,
    mut alpha: i32,
    beta: i32,
    ply: i32,
    null_ok: bool,
) -> i32 {
    ctx.nodes += 1;
    if ctx.should_stop() {
        return 0; // value ignored: the caller discards aborted iterations.
    }

    // Repetition is a draw, and is path-dependent (not a property of the position
    // alone), so it must be checked before the TT — a TT entry stored for this
    // key on a non-repeating path would otherwise mask it. A repeated position
    // can never be checkmate (the earlier identical one wasn't terminal), so it's
    // safe to return before generating moves.
    if is_repetition(board.hash, board.halfmove_clock, &ctx.rep) {
        return DRAW;
    }

    let alpha_orig = alpha;

    // Transposition probe. An entry searched at least as deep as we need lets us
    // cut immediately, how depending on its bound: an exact score returns
    // directly; a lower bound can only fail us high (≥ beta), an upper bound low
    // (≤ alpha). We mirror the fail-hard returns below so a probe cut is
    // indistinguishable from searching the node. Its best move drives ordering
    // (below) even when too shallow to cut. A deeper entry probed at a shallower
    // node returns the deeper score — a known, accepted fixed-depth instability.
    //
    // Suppress the *cut* at the fifty-move boundary: the clock isn't part of the
    // key, so a cached score from a lower clock could hide a draw we owe to the
    // rule below. (The stored move is still fine for ordering.)
    let mut tt_move = Move::NONE;
    if let Some(entry) = ctx.tt.probe(board.hash) {
        tt_move = entry.best_move;
        if board.halfmove_clock < 100 && entry.depth as u32 >= depth {
            let score = score_from_tt(entry.score, ply);
            match entry.bound {
                Bound::Exact => return score,
                Bound::Lower if score >= beta => return beta,
                Bound::Upper if score <= alpha => return alpha,
                _ => {}
            }
        }
    }

    // Generate first so we can distinguish terminal nodes (no legal moves) from
    // a quiet leaf. This must come before the depth check: a checkmate at depth
    // 0 is a mate, not whatever the static eval happens to say.
    let mut moves = generate_legal(board);
    if moves.is_empty() {
        return terminal_score(board, ply);
    }

    // Fifty-move rule — but only now that we know there's a legal move:
    // checkmate delivered on the 50th move is a mate, not a draw.
    if board.halfmove_clock >= 100 {
        return DRAW;
    }

    if depth == 0 {
        // Resolve pending captures before evaluating, so the leaf is quiet.
        return qsearch(board, ctx, alpha, beta, ply);
    }

    let side = board.side_to_move;
    let ply_idx = (ply as usize).min(MAX_PLY - 1);

    // Whether the side to move is in check — null-move pruning (#36) and late-move
    // reductions (#37) are both suppressed in check (the node is tactical). Only
    // computed at `depth >= 3`, the only depths either can fire, so shallow nodes
    // skip the attack scan.
    let in_check_node = depth >= 3 && in_check(board, side);

    // ── Null-move pruning (issue #36) ────────────────────────────────────────
    // Hand the opponent a *free* move (a null move); if a shallower search of the
    // resulting position still fails high, the real moves would too — so prune the
    // node without searching them. Gated against the known failure modes: never in
    // check, never when `beta` is a mate bound (we return `beta`, so a mate score
    // can't be fabricated), never in king-and-pawn endings where passing can beat
    // every legal move (zugzwang — the `has_non_pawn_material` guard), and never
    // twice in a row (`null_ok`, false only in the null child below). The
    // `eval >= beta` gate restricts pruning to nodes that already look winning.
    if null_ok
        && depth >= 3
        && !in_check_node
        && beta < MATE_BOUND
        && has_non_pawn_material(board, side)
        && ctx.evaluator.evaluate(board) >= beta
    {
        ctx.null_attempts += 1;
        let r = if depth >= 6 { 3 } else { 2 };
        let reduced = depth.saturating_sub(1 + r);
        ctx.rep.push(board.hash); // the pre-null position becomes an ancestor
        let undo = board.make_null_move();
        let null_score = -negamax(board, ctx, reduced, -beta, -beta + 1, ply + 1, false);
        board.unmake_null_move(undo);
        ctx.rep.pop();
        if !ctx.aborted && null_score >= beta {
            ctx.null_cutoffs += 1;
            return beta; // fail-hard cutoff; `beta < MATE_BOUND` keeps it honest
        }
    }

    order_moves(&mut moves, board, tt_move, &ctx.killers[ply_idx], &ctx.history, side);

    let mut best_move = Move::NONE;
    for (i, mv) in moves.into_iter().enumerate() {
        ctx.rep.push(board.hash); // this position becomes an ancestor of the child
        let undo = board.make_move(mv);
        let score = if i == 0 {
            // The first (best-ordered) move is the PV candidate: search it with
            // the full window so its exact score sets alpha for the scouts below.
            -negamax(board, ctx, depth - 1, -beta, -alpha, ply + 1, true)
        } else {
            // PVS scout (issue #34): a null window (alpha, alpha+1) only asks "is
            // this move worse than the PV?". A fail-low costs far less than a full
            // search.
            ctx.pvs_scouts += 1;

            // LMR (issue #37): a late, quiet, non-checking move is unlikely to be
            // best, so search it *shallower* first. `r == 0` (any non-eligible
            // move) leaves this identical to a plain PVS scout. The `gives_check`
            // probe (`in_check` on the post-make board) is a non-incremental scan,
            // so it sits last in the `&&` chain — only run once the cheap tests
            // pass.
            let mut r = 0u32;
            if depth >= 3
                && i >= 3
                && !is_tactical(mv)
                && !in_check_node
                && mv != ctx.killers[ply_idx][0]
                && mv != ctx.killers[ply_idx][1]
                && !in_check(board, board.side_to_move)
            {
                // Clamp so the reduced depth (`depth - 1 - r`) stays ≥ 1.
                r = lmr_reduction(depth, i).min(depth - 2);
            }

            let mut s = if r > 0 {
                ctx.lmr_reductions += 1;
                -negamax(board, ctx, depth - 1 - r, -alpha - 1, -alpha, ply + 1, true)
            } else {
                -negamax(board, ctx, depth - 1, -alpha - 1, -alpha, ply + 1, true)
            };
            // A reduced scout that beats alpha may have been over-reduced — verify
            // at full depth (still the null window) before trusting it.
            if !ctx.aborted && r > 0 && s > alpha {
                ctx.lmr_researches += 1;
                s = -negamax(board, ctx, depth - 1, -alpha - 1, -alpha, ply + 1, true);
            }
            // PVS: a full-depth null-window scout that beats alpha inside a *wide*
            // window (`s < beta`, else the scout already is the full window) may be
            // a new PV — re-search it full-width for its true score.
            if !ctx.aborted && s > alpha && s < beta {
                ctx.pvs_researches += 1;
                -negamax(board, ctx, depth - 1, -beta, -alpha, ply + 1, true)
            } else {
                s
            }
        };
        board.unmake_move(mv, undo);
        ctx.rep.pop();

        if ctx.aborted {
            return alpha; // unwind promptly; the result will be discarded.
        }
        if score >= beta {
            // The opponent has a refutation good enough that they'd avoid this
            // whole line; no need to look further (fail-hard cutoff). Cache it as
            // a lower bound, and reward a *quiet* refutation as a killer/history
            // move so siblings try it earlier.
            if !is_tactical(mv) {
                store_killer(&mut ctx.killers[ply_idx], mv);
                update_history(&mut ctx.history, side, mv, depth);
            }
            ctx.tt.store(board.hash, mv, score_to_tt(beta, ply), depth as u8, Bound::Lower);
            return beta;
        }
        if score > alpha {
            alpha = score;
            best_move = mv;
        }
    }

    // No cutoff: an Exact score if some move beat alpha (a PV node), otherwise an
    // Upper bound (every move failed low). Cache it with the best move found.
    let bound = if alpha > alpha_orig { Bound::Exact } else { Bound::Upper };
    ctx.tt.store(board.hash, best_move, score_to_tt(alpha, ply), depth as u8, bound);

    alpha
}

/// Quiescence search (issue #26): from a leaf, keep resolving captures and
/// promotions until the position is "quiet", then evaluate. Fixed-depth search
/// otherwise stops mid-exchange — the *horizon effect* — scoring a position as if
/// a half-finished capture sequence were over (v0.1.0's startpos score swings
/// ~100cp between even and odd depths for exactly this reason).
///
/// The key idea is the **stand-pat** baseline: the side to move is not obliged to
/// capture, so its static eval is a floor. If even that floor beats `beta` we cut;
/// otherwise we try captures, hoping to raise alpha. Because captures strictly
/// reduce material, the recursion is finite; a `ply` cap guards pathological lines.
fn qsearch(
    board: &mut Board,
    ctx: &mut SearchContext<'_>,
    mut alpha: i32,
    beta: i32,
    ply: i32,
) -> i32 {
    ctx.nodes += 1;
    if ctx.should_stop() {
        return 0; // value ignored: the caller discards aborted iterations.
    }

    // Stand-pat: not being forced to capture is the baseline. A fail-high here is
    // the common case (most positions are fine as they stand).
    let stand_pat = ctx.evaluator.evaluate(board);
    if stand_pat >= beta {
        return beta;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }
    // Safety cap: never recurse past the killer/ply tables' bound.
    if ply >= MAX_PLY as i32 {
        return alpha;
    }

    // Only captures and promotions — the moves that change material and so could
    // overturn the stand-pat score. Ordered by MVV-LVA (no TT move or killers in
    // quiescence; history is irrelevant to captures).
    let mut moves: Vec<Move> =
        generate_legal(board).into_iter().filter(|&m| is_tactical(m)).collect();
    order_moves(&mut moves, board, Move::NONE, &[Move::NONE; 2], &ctx.history, board.side_to_move);

    for mv in moves {
        let undo = board.make_move(mv);
        let score = -qsearch(board, ctx, -beta, -alpha, ply + 1);
        board.unmake_move(mv, undo);

        if ctx.aborted {
            return alpha;
        }
        if score >= beta {
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
        let mut tt = TranspositionTable::new(1);
        let result = search_timed(
            &b,
            &budget,
            Arc::new(AtomicBool::new(false)),
            now,
            &mut tt,
            &[],
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
        let budget = Budget { deadline: Some(now + Duration::from_millis(100)), max_depth: 64 };
        let mut tt = TranspositionTable::new(1);
        let result = search_timed(
            &b,
            &budget,
            Arc::new(AtomicBool::new(false)),
            now,
            &mut tt,
            &[],
            &mut |_, _| {},
        );
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
        let mut tt = TranspositionTable::new(1);
        let result = search_timed(&b, &budget, stop, now, &mut tt, &[], &mut |_, _| {});
        assert_ne!(result.best_move, Move::NONE, "must not forfeit when moves exist");
        assert!(legal_strings(&b).contains(&result.best_move.to_string()));
    }

    // ── Transposition table (issue #24) ─────────────────────────────────────

    #[test]
    fn tt_does_not_change_the_chosen_move() {
        // The correctness anchor: enabling the TT must not change which move the
        // search plays — only how fast it gets there. (We compare the move, not
        // the score: a depth-preferred TT can shift a fixed-depth score via the
        // depth-leak, which is expected, not a bug.)
        let positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
            "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 1",
        ];
        for fen in positions {
            let b = board(fen);
            for depth in 1..=3 {
                let mut off = TranspositionTable::disabled();
                let mut on = TranspositionTable::new(4);
                let without = search_with_tt(&b, depth, &mut off);
                let with = search_with_tt(&b, depth, &mut on);
                assert_eq!(
                    with.best_move, without.best_move,
                    "{fen} d{depth}: TT changed the chosen move"
                );
            }
        }
    }

    // ── Quiescence search (issue #26) ───────────────────────────────────────

    #[test]
    fn quiescence_keeps_the_startpos_score_stable_across_depths() {
        // The horizon effect: without quiescence, fixed-depth eval of the start
        // position swings ~100cp between even and odd depths, because a leaf can
        // land mid-pawn-trade. With qsearch every leaf is quiet, so the score
        // stays small and steady — no even/odd alternation.
        let b = board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        for depth in 2..=5 {
            let s = search(&b, depth).score;
            assert!(s.abs() < 60, "startpos score at depth {depth} should be near 0, got {s}");
        }
    }

    #[test]
    fn quiescence_does_not_grab_a_defended_pawn() {
        // Equal material (2 pawns each). White's only capture, bxc6, is met by
        // d7xc6 — an even trade. At depth 1 that recapture sits *past* the leaf,
        // so without quiescence white would score bxc6 as a won pawn (~+100).
        // Quiescence plays the recapture out, so the position stays ~even.
        let b = board("4k3/3p4/2p5/1P6/3P4/8/8/4K3 w - - 0 1");
        let s = search(&b, 1).score;
        assert!(s.abs() < 80, "a defended-pawn grab should not look winning, got {s}");
    }

    // ── Draw detection (issue #28) ──────────────────────────────────────────

    #[test]
    fn repetition_is_detected_only_within_the_halfmove_window() {
        // `rep` holds ancestor keys oldest-first. A key repeating an ancestor
        // inside the last `halfmove` plies is a draw; an older match is not (an
        // irreversible move reset the clock, so it can't actually recur).
        let rep = [10u64, 20, 30, 40];
        assert!(is_repetition(20, 4, &rep)); // 20 is within the window of 4
        assert!(!is_repetition(20, 2, &rep)); // window 2 scans only [30,40]
        assert!(!is_repetition(99, 4, &rep)); // never occurred
        assert!(!is_repetition(10, 0, &rep)); // window 0 scans nothing
    }

    #[test]
    fn fifty_move_rule_scores_a_won_position_as_a_draw() {
        // White is up a whole queen, but the halfmove clock is at the limit and
        // no capture or pawn move can reset it, so every continuation is a draw.
        // (Depth 1, so the search can't instead find a forced mate.)
        let b = board("8/8/8/8/4k3/8/8/Q6K w - - 100 1");
        let s = search(&b, 1).score;
        assert_eq!(s, 0, "fifty-move rule should score this drawn, got {s}");
    }

    #[test]
    fn checkmate_takes_precedence_over_the_fifty_move_rule() {
        // The clock is at 99, so Ra8# is the 100th ply — but checkmate ends the
        // game, so it's scored as a mate, not a fifty-move draw.
        let b = board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 99 1");
        let r = search(&b, 1);
        assert_eq!(r.best_move.to_string(), "a1a8");
        assert_eq!(r.score, MATE - 1);
    }

    #[test]
    fn a_seeded_repetition_lets_a_losing_side_claim_a_draw() {
        // White is down a whole queen — every line loses (~-900). But if a move
        // reaches a position already in the game history, that's a draw worth 0,
        // strictly better than losing. We seed the key of one move's result, so
        // the *only* way the root score is 0 (not deeply negative) is the
        // repetition being recognized — a discriminating test, unlike a winning
        // side that scores positive whether or not repetition works.
        let root = board("6k1/8/8/8/8/1q6/8/6K1 w - - 10 1");
        let saving = generate_legal(&root)[0];
        let mut after = root.clone();
        after.make_move(saving);
        let repeated_key = after.hash;

        // History long enough that the window (halfmove 10+) reaches the seed.
        let history = vec![repeated_key; 4];
        let mut tt = TranspositionTable::new(1);
        let budget = Budget { deadline: None, max_depth: 3 };
        let r = search_timed(
            &root,
            &budget,
            Arc::new(AtomicBool::new(false)),
            Instant::now(),
            &mut tt,
            &history,
            &mut |_, _| {},
        );
        assert_eq!(r.score, 0, "should claim the seeded repetition draw, got {}", r.score);
    }

    #[test]
    fn reusing_a_populated_tt_cuts_nodes_and_keeps_the_move() {
        // Searching the same position twice with the same table: the second pass
        // reuses cached results, so it visits far fewer nodes and plays the same
        // move — the TT changes speed, not the decision.
        let b = board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1");
        let mut tt = TranspositionTable::new(16);
        let cold = search_with_tt(&b, 4, &mut tt);
        let warm = search_with_tt(&b, 4, &mut tt);
        assert_eq!(cold.best_move, warm.best_move);
        assert!(
            warm.nodes < cold.nodes,
            "warm TT should visit fewer nodes: cold {} then warm {}",
            cold.nodes,
            warm.nodes
        );
    }

    /// PVS must be **result-invariant**: with the transposition table disabled it
    /// returns exactly the score (and, where the position has a unique answer, the
    /// move) that plain alpha-beta did. These golden numbers were captured from the
    /// pre-PVS engine with a disabled TT at depth 6. The TT is disabled on purpose:
    /// PVS stores null-window bounds a full-window search never would, so *with* a
    /// TT a fixed-depth score can legitimately differ (the accepted instability
    /// documented at the TT probe) — that would make this a flaky, wrong assertion.
    #[test]
    fn pvs_is_result_invariant_with_tt_disabled() {
        // (fen, golden score, unique best move or None when many moves tie)
        let cases: [(&str, i32, Option<&str>); 5] = [
            // Startpos scores 0 with many equal replies — assert score only.
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0, None),
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 35, Some("e2a6")),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 90, Some("b4f4")),
            ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 25, Some("c3d5")),
            ("4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1", 500, Some("d1d4")),
        ];
        for (fen, score, best) in cases {
            let b = board(fen);
            let mut tt = TranspositionTable::disabled();
            let r = search_with_tt(&b, 6, &mut tt);
            assert_eq!(r.score, score, "score changed for {fen}");
            if let Some(mv) = best {
                assert_eq!(r.best_move.to_string(), mv, "best move changed for {fen}");
            }
        }
    }

    /// Sanity check on the PVS scout: with TT + ordering doing their job, almost
    /// every scout proves a move worse without a full-width re-search. A high
    /// re-search rate would mean ordering is broken (or PVS is miswired) and the
    /// scout is pure overhead. We run a normal middlegame through iterative
    /// deepening (so the TT seeds ordering, as in a real search) and assert the
    /// re-search rate stays well under 20%.
    #[test]
    fn pvs_research_rate_stays_low() {
        let b = board("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
        let mut tt = TranspositionTable::new(16);
        tt.new_search();
        let mut ctx = SearchContext::unbounded(&mut tt);
        let mut bb = b.clone();
        for d in 1..=7 {
            run_root(&mut bb, &mut ctx, d);
        }
        assert!(ctx.pvs_scouts > 1000, "expected many scouts, got {}", ctx.pvs_scouts);
        assert!(
            ctx.pvs_researches * 5 < ctx.pvs_scouts,
            "re-search rate too high: {} researches / {} scouts",
            ctx.pvs_researches,
            ctx.pvs_scouts
        );
    }

    /// LMR must not drop a tactic the engine could previously find. These
    /// fixtures were captured from the pre-LMR engine (PVS) at depth 7 — where it
    /// finds each and LMR is genuinely engaged — and assert the *decisive* score
    /// survives (a mate, or clearly winning). Asserting the score rather than an
    /// exact move is robust to LMR's legitimate non-invariance.
    #[test]
    fn lmr_keeps_finding_known_tactics() {
        // (fen, depth, minimum acceptable score)
        let cases: [(&str, u32, i32); 4] = [
            ("2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1", 7, MATE_BOUND), // mate 2
            ("r2qrb2/p1pn1Qp1/1p4Nk/4PR2/3n4/7N/P5PP/R6K w - - 0 1", 7, MATE_BOUND),        // mate 2
            ("1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1", 7, MATE_BOUND),         // mate 3
            ("2r1nrk1/p4p1p/1p2p1pQ/nP6/2pNP3/P1B2q1P/2B2P2/R4RK1 w - - 0 1", 7, 800),       // +1150
        ];
        for (fen, depth, min_score) in cases {
            let r = search(&board(fen), depth);
            assert!(
                r.score >= min_score,
                "LMR dropped a tactic in {fen}: score {} < {min_score}",
                r.score
            );
        }
    }

    /// LMR is actually firing and its reductions are mostly safe: on a normal
    /// middlegame searched through iterative deepening (so the TT seeds ordering),
    /// reductions happen in bulk and only a minority fail high and force a
    /// full-depth re-search. A high re-search rate would mean we over-reduce.
    #[test]
    fn lmr_reductions_fire_and_research_rate_is_bounded() {
        let b = board("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
        let mut tt = TranspositionTable::new(16);
        tt.new_search();
        let mut ctx = SearchContext::unbounded(&mut tt);
        let mut bb = b.clone();
        for d in 1..=8 {
            run_root(&mut bb, &mut ctx, d);
        }
        assert!(ctx.lmr_reductions > 1000, "expected many reductions, got {}", ctx.lmr_reductions);
        assert!(
            ctx.lmr_researches * 2 < ctx.lmr_reductions,
            "LMR re-search rate too high (over-reducing): {} researches / {} reductions",
            ctx.lmr_researches,
            ctx.lmr_reductions
        );
    }

    // ── Null-move pruning (issue #36) ───────────────────────────────────────

    /// NMP must not drop a tactic the engine could previously find. Same fixtures
    /// (and depth) as [`lmr_keeps_finding_known_tactics`]: NMP is engaged at this
    /// depth, so a decisive score surviving proves the prune isn't cutting real
    /// lines. Asserting the score (not an exact move) is robust to NMP's
    /// legitimate non-invariance.
    #[test]
    fn nmp_keeps_finding_known_tactics() {
        // (fen, depth, minimum acceptable score)
        let cases: [(&str, u32, i32); 4] = [
            ("2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1", 7, MATE_BOUND), // mate 2
            ("r2qrb2/p1pn1Qp1/1p4Nk/4PR2/3n4/7N/P5PP/R6K w - - 0 1", 7, MATE_BOUND),        // mate 2
            ("1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1", 7, MATE_BOUND),         // mate 3
            ("2r1nrk1/p4p1p/1p2p1pQ/nP6/2pNP3/P1B2q1P/2B2P2/R4RK1 w - - 0 1", 7, 800),       // +1150
        ];
        for (fen, depth, min_score) in cases {
            let r = search(&board(fen), depth);
            assert!(
                r.score >= min_score,
                "NMP dropped a tactic in {fen}: score {} < {min_score}",
                r.score
            );
        }
    }

    /// NMP is actually firing and pruning: on a normal middlegame searched through
    /// iterative deepening (so the TT seeds ordering and eval often clears beta),
    /// null moves are attempted in bulk and a healthy fraction fail high and prune.
    #[test]
    fn nmp_attempts_and_cuts() {
        let b = board("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
        let mut tt = TranspositionTable::new(16);
        tt.new_search();
        let mut ctx = SearchContext::unbounded(&mut tt);
        let mut bb = b.clone();
        for d in 1..=8 {
            run_root(&mut bb, &mut ctx, d);
        }
        assert!(ctx.null_attempts > 100, "expected many null attempts, got {}", ctx.null_attempts);
        assert!(ctx.null_cutoffs > 0, "expected some null cutoffs, got {}", ctx.null_cutoffs);
    }

    /// Zugzwang guard: in a king-and-pawn ending neither side has non-pawn
    /// material, so NMP must never fire (passing can beat every legal move there).
    /// A pure pawn endgame searched deep should record *zero* null attempts —
    /// a direct check that the `has_non_pawn_material` guard holds.
    #[test]
    fn nmp_is_disabled_without_non_pawn_material() {
        let b = board("8/8/8/4k3/8/4K3/4P3/8 w - - 0 1"); // K+P vs K
        let mut tt = TranspositionTable::new(16);
        tt.new_search();
        let mut ctx = SearchContext::unbounded(&mut tt);
        let mut bb = b.clone();
        for d in 1..=8 {
            run_root(&mut bb, &mut ctx, d);
        }
        assert_eq!(
            ctx.null_attempts, 0,
            "NMP fired in a pawn-only endgame ({} attempts) — zugzwang guard failed",
            ctx.null_attempts
        );
    }
}
