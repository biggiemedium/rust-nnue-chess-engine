//! UCI (Universal Chess Interface)
//! http://wbec-ridderkerk.nl/html/UCIProtocol.html
//! https://www.chessprogramming.org/UCI
//!
//! The GUI (or lichess bot, or a match runner) talks to us over stdin / stdout:
//!   GUI -> uci                          us -> id ..., option ..., uciok
//!   GUI -> isready                      us -> readyok
//!   GUI -> position startpos moves e2e4
//!   GUI -> go wtime 60000 btime 60000   us -> info ... (every depth), then bestmove e7e5
//!
//! The search runs on its own thread so we can still read "stop" / "isready" while it thinks.
//! The Searcher gets moved onto that thread and handed back when the search finishes.

pub mod command_parser;
pub mod option_handler;
pub mod time_manager;
pub mod benchmark;

use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::board::Position;
use crate::evaluation::evaluator::Evaluator;
use crate::search::searcher::Searcher;
use command_parser::{Command, GoParams};
use option_handler::{EngineOptions, OptionChange};

const ENGINE_NAME: &str = "ChessProject";
const ENGINE_AUTHOR: &str = "James";

// recursion goes up to MAX_PLY deep + quiescence, the default 2MB thread stack is cutting it close
const SEARCH_STACK_SIZE: usize = 32 * 1024 * 1024;

pub struct UciEngine {
    position: Position,
    game_keys: Vec<u64>, // every position before the current one, for repetition detection
    options: EngineOptions,

    // either we're holding the searcher, or the search thread is
    searcher: Option<Searcher>,
    search_thread: Option<JoinHandle<Searcher>>,
    stop: Arc<AtomicBool>,
}

impl UciEngine {

    pub fn new() -> Self {
        let searcher = Searcher::new();
        let stop = searcher.stop_signal();

        Self {
            position: Position::startpos(),
            game_keys: Vec::new(),
            options: EngineOptions::new(),
            searcher: Some(searcher),
            search_thread: None,
            stop,
        }
    }

    // main loop, runs until "quit" or the GUI closes stdin
    pub fn run(&mut self) {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else {
                break;
            };
            if !self.handle(&line) {
                break;
            }
        }
        self.stop_search();
    }

    // false = quit
    pub fn handle(&mut self, line: &str) -> bool {
        match command_parser::parse(line) {
            Command::Uci => {
                println!("id name {ENGINE_NAME} {}", env!("CARGO_PKG_VERSION"));
                println!("id author {ENGINE_AUTHOR}");
                self.options.print();
                println!("uciok");
            }
            // answer straight away even mid search, GUIs use this as a ping
            Command::IsReady => println!("readyok"),
            Command::UciNewGame => {
                self.searcher().new_game();
                self.set_position(None, &[]);
            }
            Command::SetOption { name, value } => {
                match self.options.set(&name, value.as_deref()) {
                    OptionChange::ResizeHash(mb) => self.searcher().resize_hash(mb),
                    OptionChange::ClearHash => self.searcher().new_game(),
                    OptionChange::Nothing => {}
                }
            }
            Command::Position { fen, moves } => self.set_position(fen.as_deref(), &moves),
            Command::Go(go) => self.go(go),
            Command::Stop => self.stop_search(),
            Command::Quit => return false,

            Command::Bench(depth) => {
                self.wait_for_search();
                benchmark::bench(depth.unwrap_or(benchmark::DEFAULT_BENCH_DEPTH));
            }
            Command::Perft(depth) => {
                self.wait_for_search();
                benchmark::perft_divide(&self.position, depth);
            }
            Command::Display => println!("{}", self.position.to_fen()),
            Command::Eval => {
                let eval = Evaluator::new().evaluate(&self.position);
                println!("eval {eval} cp (side to move)");
            }
            Command::Unknown(text) => {
                if !text.is_empty() {
                    println!("info string unknown command {text}");
                }
            }
        }
        true
    }

    fn set_position(&mut self, fen: Option<&str>, moves: &[String]) {
        let start = match fen {
            Some(fen) => match Position::from_fen(fen) {
                Ok(pos) => pos,
                Err(e) => {
                    println!("info string {e}");
                    return;
                }
            },
            None => Position::startpos(),
        };

        // replay the moves, keeping every key on the way for repetition detection
        let mut pos = start;
        let mut keys = Vec::with_capacity(moves.len());

        for uci in moves {
            match pos.play_uci(uci) {
                Some(next) => {
                    keys.push(pos.key());
                    pos = next;
                }
                None => {
                    println!("info string illegal move {uci}");
                    break;
                }
            }
        }

        self.position = pos;
        self.game_keys = keys;
    }

    fn go(&mut self, go: GoParams) {
        self.wait_for_search(); // a "go" while already searching -> finish that one first

        let limits = time_manager::limits(&go, self.position.white_to_move(), self.options.move_overhead_ms);
        let mut searcher = self.searcher.take().expect("searcher missing");
        searcher.set_game_history(&self.game_keys);

        let pos = self.position.clone();
        let stop = self.stop.clone();
        let infinite = go.infinite;
        stop.store(false, Ordering::Relaxed);

        let handle = thread::Builder::new()
            .name("search".into())
            .stack_size(SEARCH_STACK_SIZE)
            .spawn(move || {
                let result = searcher.search(&pos, limits);

                // spec: in infinite mode bestmove only comes AFTER "stop", even if we found mate
                if infinite {
                    while !stop.load(Ordering::Relaxed) {
                        thread::sleep(Duration::from_millis(1));
                    }
                }

                // no legal moves -> "0000" is the uci null move
                let best = result.best_move.map(|m| m.to_uci_string()).unwrap_or_else(|| "0000".into());
                println!("bestmove {best}");
                searcher
            })
            .expect("couldn't start search thread");

        self.search_thread = Some(handle);
    }

    fn stop_search(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.wait_for_search();
    }

    // join the search thread (if any) and take the searcher back
    fn wait_for_search(&mut self) {
        if let Some(handle) = self.search_thread.take() {
            let searcher = handle.join().expect("search thread panicked");
            self.searcher = Some(searcher);
        }
    }

    // anything that touches the searcher has to wait until we own it again
    fn searcher(&mut self) -> &mut Searcher {
        self.wait_for_search();
        self.searcher.as_mut().expect("searcher missing")
    }
}

// Default state / restarted state
impl Default for UciEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_with_moves() {
        let mut engine = UciEngine::new();
        engine.handle("position startpos moves e2e4 e7e5 g1f3");

        assert_eq!(engine.position.to_fen(), "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");
        assert_eq!(engine.game_keys.len(), 3);
    }

    #[test]
    fn castling_both_notations() {
        // e1g1 (standard) and e1h1 (king takes rook, chess960 style) both have to work
        let fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
        for castle in ["e1g1", "e1h1"] {
            let mut engine = UciEngine::new();
            engine.handle(&format!("position fen {fen} moves {castle}"));
            assert_eq!(engine.position.to_fen(), "r3k2r/8/8/8/8/8/8/R4RK1 b kq - 1 1");
        }
    }

    #[test]
    fn go_then_stop_gives_searcher_back() {
        let mut engine = UciEngine::new();
        engine.handle("position startpos");
        engine.handle("go infinite");
        thread::sleep(Duration::from_millis(50));
        engine.handle("stop");

        assert!(engine.search_thread.is_none());
        assert!(engine.searcher.is_some());
    }

    #[test]
    fn go_depth_finishes_on_its_own() {
        let mut engine = UciEngine::new();
        engine.handle("position startpos");
        engine.handle("go depth 4");
        engine.wait_for_search();
        assert!(engine.searcher.is_some());
    }
}
