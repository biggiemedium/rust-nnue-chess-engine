//! The searcher, holds everything one search needs.
//!
//! The actual search is split across the folder:
//!   iterative_deepening.rs        -> search() entry point, depth 1, 2, 3 ... until time/nodes run out
//!   principal_variation_search.rs -> the main alpha-beta tree
//!   quiescence.rs                 -> keeps searching captures at the leaves
//!
//! Scores are negamax style -> always from the side to move's POV.
//! Good for me = bad for you, so each level just flips the sign: score = -child_score
//! saves writing every bit of the search twice (once for white maxing, once for black minimizing)

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::board::Move;
use crate::evaluation::evaluator::Evaluator;
use crate::search::move_ordering::history_heuristic::HistoryHeuristic;
use crate::search::move_ordering::killer_moves::KillerMoves;
use crate::search::transposition::transposition_table::TranspositionTable;

pub const INFINITY: i32 = 32_001;
pub const MATE: i32 = 32_000;
pub const MAX_PLY: usize = 128;

// any score past this is a forced mate, not a real eval
pub const MATE_BOUND: i32 = MATE - MAX_PLY as i32;

// how often (in nodes) we look at the clock, Instant::now() isn't free
const TIME_CHECK_INTERVAL: u64 = 2048;

pub const DEFAULT_HASH_MB: usize = 64;

#[derive(Clone, Copy, Debug, Default)]
pub struct SearchLimits {
    pub depth: Option<i32>,
    pub nodes: Option<u64>, // datagen uses this -> same strength no matter how fast the pc is
    pub movetime: Option<Duration>,
}

#[derive(Clone, Copy, Debug)]
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
}

pub struct Searcher {
    pub(super) evaluator: Evaluator,
    pub(super) limits: SearchLimits,
    pub(super) start: Instant,
    pub(super) nodes: u64,
    pub(super) stopped: bool,

    // zobrist keys of every position before the current one (game moves + search path)
    // Position doesn't keep history (copy-make would clone it every node) so we do it here
    pub(super) position_history: Vec<u64>,

    // these 3 survive between searches (until new_game)
    pub(super) tt: TranspositionTable,
    pub(super) killers: KillerMoves,
    pub(super) history: HistoryHeuristic,

    pub print_info: bool, // uci "info" lines, turned off for tests + datagen

    // uci "stop" comes in on the main thread while we search on another one -> shared flag
    stop_signal: Arc<AtomicBool>,
}

impl Searcher {

    pub fn new() -> Self {
        Self {
            evaluator: Evaluator::new(),
            limits: SearchLimits::default(),
            start: Instant::now(),
            nodes: 0,
            stopped: false,
            position_history: Vec::with_capacity(512),
            tt: TranspositionTable::new(DEFAULT_HASH_MB),
            killers: KillerMoves::new(),
            history: HistoryHeuristic::new(),
            print_info: true,
            stop_signal: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Keys of the positions already played in the game (oldest first), not including the current one.
    /// Without this the engine can't see it's walking into a repetition.
    pub fn set_game_history(&mut self, keys: &[u64]) {
        self.position_history.clear();
        self.position_history.extend_from_slice(keys);
    }

    pub fn stop_signal(&self) -> Arc<AtomicBool> {
        self.stop_signal.clone()
    }

    // uci "ucinewgame" -> nothing from the last game should leak into this one
    pub fn new_game(&mut self) {
        self.tt.clear();
        self.killers.clear();
        self.history.clear();
        self.position_history.clear();
    }

    // uci "setoption name Hash"
    pub fn resize_hash(&mut self, megabytes: usize) {
        self.tt = TranspositionTable::new(megabytes);
    }

    // called every node, true = drop everything and unwind
    pub(super) fn should_stop(&mut self) -> bool {
        if self.stopped {
            return true;
        }

        if let Some(max_nodes) = self.limits.nodes {
            if self.nodes >= max_nodes {
                self.stopped = true;
            }
        }

        if self.nodes % TIME_CHECK_INTERVAL == 0 {
            if self.stop_signal.load(Ordering::Relaxed) {
                self.stopped = true;
            }

            if let Some(movetime) = self.limits.movetime {
                if self.start.elapsed() >= movetime {
                    self.stopped = true;
                }
            }
        }

        self.stopped
    }

    pub(super) fn is_repetition(&self, key: u64, halfmove_clock: u32) -> bool {
        // only positions since the last capture / pawn move can repeat (halfmove clock)
        // and it has to be the same side to move -> every 2nd entry back
        // one repeat counts as a draw inside the search, if it's good to repeat once it's good to repeat 3 times
        let len = self.position_history.len();
        let window = (halfmove_clock as usize).min(len);

        self.position_history[len - window..]
            .iter()
            .rev()
            .skip(1)
            .step_by(2)
            .any(|&k| k == key)
    }
}

// Default state / restarted state
impl Default for Searcher {
    fn default() -> Self {
        Self::new()
    }
}

// mate in N from the side to move's POV, for printing
pub fn mate_in(score: i32) -> Option<i32> {
    if score > MATE_BOUND {
        Some((MATE - score + 1) / 2)
    } else if score < -MATE_BOUND {
        Some(-(MATE + score + 1) / 2)
    } else {
        None
    }
}
