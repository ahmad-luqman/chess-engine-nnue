//! Time management: turning a clock into a per-move deadline.
//!
//! In a real game the engine isn't told "search to depth 8" — it's told "you
//! have 3 minutes left, plus 2 seconds per move." This module turns those UCI
//! `go` parameters ([`Limits`]) into a concrete [`Budget`] the search can obey:
//! a wall-clock deadline and a maximum depth. [`search_timed`](crate::search)
//! then deepens until the deadline looms.
//!
//! The allocation policy here is deliberately simple — a fixed fraction of the
//! remaining clock plus the increment. It's enough to play sound games without
//! flagging (losing on time); smarter schemes (spending more in complex
//! middlegames, less when low) are a later refinement.
//!
//! ## A note on `stop` and `infinite`
//!
//! Phase 1 search is **synchronous**: while it runs, the UCI loop isn't reading
//! input, so a `stop` command can't be received mid-search. Time-controlled play
//! doesn't need it — the engine self-terminates at its deadline — but it means
//! we can't honour a true `infinite` "search until stopped" search. So `infinite`
//! (and a bare `go` with no parameters) get a modest fixed depth instead of an
//! unbounded one: harmless to a tournament manager (which always sends a clock),
//! and it can't hang if typed by hand. A real `stop`/`infinite` needs the search
//! to run on its own thread — deferred past Phase 1.

use std::time::{Duration, Instant};

use crate::types::Color;

/// Milliseconds we hold back as a safety margin against the cost of moving,
/// process scheduling, and GUI/network lag — so we send `bestmove` before the
/// flag actually falls.
const MOVE_OVERHEAD_MS: u64 = 40;

/// When the GUI doesn't say how many moves remain until the next time control,
/// we assume the game still has roughly this many moves to go, and spend that
/// fraction of the clock on this move.
const ASSUMED_MOVES_TO_GO: u64 = 20;

/// Depth ceiling for time-limited searches — a sanity bound; real games stop on
/// the clock long before this.
const MAX_DEPTH: u32 = 64;

/// Depth used when there is no clock and no explicit depth (a bare `go`, or
/// `infinite` which we can't truly honour while synchronous — see module docs).
const DEFAULT_DEPTH: u32 = 6;

/// Raw `go` parameters, in milliseconds, straight off the UCI command. All
/// optional: the UCI loop fills in whatever the GUI sent.
#[derive(Default, Clone, Debug)]
pub struct Limits {
    /// White's / Black's remaining time.
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    /// White's / Black's increment per move.
    pub winc: Option<u64>,
    pub binc: Option<u64>,
    /// Moves until the next time control, if a fixed control is in use.
    pub movestogo: Option<u32>,
    /// Fixed time for this move (`go movetime`): use exactly this, no estimate.
    pub movetime: Option<u64>,
    /// Fixed search depth (`go depth`): no time limit.
    pub depth: Option<u32>,
    /// `go infinite`: search "forever" (see module docs for the Phase 1 caveat).
    pub infinite: bool,
}

/// A resolved budget for one move: how long the search may run, and how deep.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    /// Stop once `Instant::now()` reaches here. `None` = no time limit (the
    /// search is bounded by `max_depth` instead).
    pub deadline: Option<Instant>,
    /// Don't start an iteration deeper than this.
    pub max_depth: u32,
}

/// Resolve `limits` for the side to move into a concrete [`Budget`]. `now` is
/// the search's start instant — passed in (rather than read here) so the
/// deadline shares an origin with the search's own `elapsed()` clock and so this
/// function is deterministic to test.
pub fn allocate(limits: &Limits, stm: Color, now: Instant) -> Budget {
    // 1. Fixed time for the move: spend it (minus the safety margin), exactly.
    if let Some(movetime) = limits.movetime {
        return Budget {
            deadline: Some(now + millis_minus_overhead(movetime)),
            max_depth: limits.depth.unwrap_or(MAX_DEPTH),
        };
    }

    // 2. Clock-based: spend a fraction of our remaining time plus the increment.
    let (time, inc) = match stm {
        Color::White => (limits.wtime, limits.winc),
        Color::Black => (limits.btime, limits.binc),
    };
    if let Some(time) = time {
        let inc = inc.unwrap_or(0);
        let alloc = match limits.movestogo {
            // Divide the clock across the moves until the control, keeping the
            // full increment (we get it back each move).
            Some(mtg) => time / u64::from(mtg).max(1) + inc,
            // Open-ended: assume a fixed horizon, and bank only half the
            // increment so a tiny increment can't make us greedy.
            None => time / ASSUMED_MOVES_TO_GO + inc / 2,
        };
        // Never allocate more than what's actually on the clock (minus margin).
        let capped = alloc.min(time.saturating_sub(MOVE_OVERHEAD_MS)).max(1);
        return Budget {
            deadline: Some(now + Duration::from_millis(capped)),
            max_depth: limits.depth.unwrap_or(MAX_DEPTH),
        };
    }

    // 3. Explicit fixed depth, no clock.
    if let Some(depth) = limits.depth {
        return Budget { deadline: None, max_depth: depth };
    }

    // 4. `infinite` or bare `go`: a finite default depth (see module docs).
    Budget { deadline: None, max_depth: DEFAULT_DEPTH }
}

/// `ms` as a `Duration`, less the safety margin, but never zero — even a tiny
/// `movetime` must leave us *some* time to search.
fn millis_minus_overhead(ms: u64) -> Duration {
    Duration::from_millis(ms.saturating_sub(MOVE_OVERHEAD_MS).max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn movetime_uses_that_time_minus_overhead() {
        let now = Instant::now();
        let limits = Limits { movetime: Some(1000), ..Default::default() };
        let budget = allocate(&limits, Color::White, now);
        let span = budget.deadline.unwrap().duration_since(now);
        // 1000ms minus the 40ms safety margin.
        assert_eq!(span, Duration::from_millis(960));
        assert_eq!(budget.max_depth, MAX_DEPTH);
    }

    #[test]
    fn fixed_depth_has_no_deadline() {
        let now = Instant::now();
        let limits = Limits { depth: Some(7), ..Default::default() };
        let budget = allocate(&limits, Color::White, now);
        assert!(budget.deadline.is_none());
        assert_eq!(budget.max_depth, 7);
    }

    #[test]
    fn clock_uses_the_side_to_moves_own_time() {
        let now = Instant::now();
        // White has lots of time, Black almost none. The allocation must read the
        // mover's clock — so White gets a big budget, Black a tiny one.
        let limits = Limits {
            wtime: Some(60_000),
            btime: Some(1_000),
            winc: Some(0),
            binc: Some(0),
            ..Default::default()
        };
        let white = allocate(&limits, Color::White, now).deadline.unwrap().duration_since(now);
        let black = allocate(&limits, Color::Black, now).deadline.unwrap().duration_since(now);
        // 60000/20 = 3000ms for White; 1000/20 = 50ms for Black.
        assert_eq!(white, Duration::from_millis(3000));
        assert_eq!(black, Duration::from_millis(50));
    }

    #[test]
    fn never_allocates_more_than_the_clock() {
        let now = Instant::now();
        // With almost no time left, the allocation must not exceed time - margin.
        let limits = Limits { wtime: Some(30), winc: Some(0), ..Default::default() };
        let budget = allocate(&limits, Color::White, now);
        let span = budget.deadline.unwrap().duration_since(now);
        // 30/20 = 1ms, and that's already under the clock; just assert it's tiny
        // and positive (never zero or negative).
        assert!(span >= Duration::from_millis(1));
        assert!(span <= Duration::from_millis(30));
    }

    #[test]
    fn movestogo_divides_the_clock() {
        let now = Instant::now();
        let limits = Limits {
            wtime: Some(60_000),
            winc: Some(0),
            movestogo: Some(10),
            ..Default::default()
        };
        let span = allocate(&limits, Color::White, now).deadline.unwrap().duration_since(now);
        // 60000/10 = 6000ms.
        assert_eq!(span, Duration::from_millis(6000));
    }

    #[test]
    fn bare_go_and_infinite_fall_back_to_a_finite_depth() {
        let now = Instant::now();
        let bare = allocate(&Limits::default(), Color::White, now);
        assert!(bare.deadline.is_none());
        assert_eq!(bare.max_depth, DEFAULT_DEPTH);

        let infinite = Limits { infinite: true, ..Default::default() };
        let budget = allocate(&infinite, Color::White, now);
        assert!(budget.deadline.is_none());
        assert_eq!(budget.max_depth, DEFAULT_DEPTH);
    }
}
