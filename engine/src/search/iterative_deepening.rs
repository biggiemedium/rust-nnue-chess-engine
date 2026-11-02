//! Iterative deepening
//! https://www.chessprogramming.org/Iterative_Deepening
//!
//! Search depth 1, then 2, then 3 ... until we run out of time.
//! Sounds wasteful but each depth is way cheaper than the next one (branching factor)
//! and the best move from the last depth gets searched first -> way more cutoffs.
//! Also means we always have a move ready when the clock runs out.

use std::time::Instant;

use crate::board::{Move, Position};
use crate::search::move_ordering::move_ordering::{order_moves, OrderContext};
use crate::search::searcher::{mate_in, SearchLimits, SearchResult, Searcher, MAX_PLY};

impl Searcher {

    pub fn search(&mut self, pos: &Position, limits: SearchLimits) -> SearchResult {
        self.limits = limits;
        self.start = Instant::now();
        self.nodes = 0;
        self.stopped = false;

        // killers are about this exact tree, last search's ones are useless
        // history is more general -> keep it but halve it so this search's info takes over
        self.killers.clear();
        self.history.age();

        let max_depth = limits.depth.unwrap_or(MAX_PLY as i32 - 1);

        let mut moves = pos.legal_moves();
        let mut result = SearchResult {
            best_move: moves.first().copied(), // something legal in case we get stopped instantly
            score: 0,
            depth: 0,
            nodes: 0,
        };

        if moves.is_empty() {
            return result; // game over -> checkmate / stalemate already happened
        }

        for depth in 1..=max_depth {
            // last depth's best move goes first
            let ctx = OrderContext {
                tt_move: result.best_move,
                killers: self.killers.get(0),
                history: &self.history,
            };
            order_moves(pos, &mut moves, &ctx);

            let (score, best_move) = self.search_root(pos, &moves, depth);

            // a half finished depth can't be trusted, keep the last full one
            if self.stopped {
                break;
            }

            result = SearchResult { best_move, score, depth, nodes: self.nodes };
            self.print_info_line(pos, &result);

            // found a mate, deeper won't change anything
            if mate_in(score).is_some() {
                break;
            }

            // next depth takes ~3-5x as long, don't start it if we probably can't finish it
            if let Some(movetime) = limits.movetime {
                if self.start.elapsed() > movetime / 2 {
                    break;
                }
            }
        }

        result.nodes = self.nodes;
        result
    }

    fn print_info_line(&self, pos: &Position, result: &SearchResult) {
        if !self.print_info {
            return;
        }

        let elapsed = self.start.elapsed();
        let nps = (self.nodes as f64 / elapsed.as_secs_f64().max(0.001)) as u64;

        let score = match mate_in(result.score) {
            Some(n) => format!("mate {n}"),
            None => format!("cp {}", result.score),
        };
        let pv: Vec<String> = self.pv_line(pos, result).iter().map(|m| m.to_uci_string()).collect();

        println!(
            "info depth {} score {} nodes {} nps {} hashfull {} time {} pv {}",
            result.depth, score, self.nodes, nps, self.tt.hashfull(), elapsed.as_millis(), pv.join(" ")
        );
    }

    // the "principal variation" -> the line both sides are expected to play
    // we don't store it anywhere, just follow the best move in the TT from position to position
    pub fn pv_line(&self, root: &Position, result: &SearchResult) -> Vec<Move> {
        let Some(best) = result.best_move else {
            return Vec::new();
        };

        let mut line = vec![best];
        let mut pos = root.play(best);
        let mut seen = vec![root.key(), pos.key()];

        while line.len() < result.depth.max(1) as usize {
            let Some(mv) = self.tt.probe(pos.key()).and_then(|e| e.best_move) else {
                break;
            };

            // 2 positions can share a slot -> make sure the move is actually legal here
            if !pos.legal_moves().contains(&mv) {
                break;
            }

            pos = pos.play(mv);
            line.push(mv);

            // repetition -> the line would loop forever
            if seen.contains(&pos.key()) {
                break;
            }
            seen.push(pos.key());
        }

        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn best_move(fen: &str, depth: i32) -> SearchResult {
        let mut searcher = Searcher::new();
        searcher.print_info = false;
        let limits = SearchLimits { depth: Some(depth), ..Default::default() };
        searcher.search(&Position::from_fen(fen).unwrap(), limits)
    }

    fn uci(result: &SearchResult) -> String {
        result.best_move.map(|m| m.to_uci_string()).unwrap_or_default()
    }

    #[test]
    fn finds_scholars_mate() {
        let result = best_move("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4", 3);
        assert_eq!(uci(&result), "f3f7");
        assert_eq!(mate_in(result.score), Some(1));
    }

    #[test]
    fn finds_back_rank_mate_as_black() {
        // checks the negamax sign flip works for black too
        let result = best_move("1r4k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1", 3);
        assert_eq!(uci(&result), "b8b1");
        assert_eq!(mate_in(result.score), Some(1));
    }

    #[test]
    fn takes_free_queen() {
        let result = best_move("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1", 3);
        assert_eq!(uci(&result), "e4d5");
    }

    #[test]
    fn finds_mate_in_two() {
        // Nf6+ gxf6 Bxf7# -> mate scores have to come out of the TT right for this to say mate 2
        let result = best_move("r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 1", 5);
        assert_eq!(uci(&result), "d5f6");
        assert_eq!(mate_in(result.score), Some(2));
    }

    #[test]
    fn stalemate_has_no_move() {
        let result = best_move("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 3);
        assert!(result.best_move.is_none());
    }

    #[test]
    fn respects_node_limit() {
        let mut searcher = Searcher::new();
        searcher.print_info = false;
        let limits = SearchLimits { nodes: Some(5_000), ..Default::default() };
        let result = searcher.search(&Position::startpos(), limits);

        assert!(result.best_move.is_some());
        assert!(result.nodes <= 5_001, "searched {} nodes", result.nodes);
    }

    // not a real test, just prints the info lines so you can eyeball depth / nps
    // cargo test --release startpos_info -- --ignored --nocapture
    #[test]
    #[ignore]
    fn startpos_info() {
        let mut searcher = Searcher::new();
        let limits = SearchLimits { movetime: Some(std::time::Duration::from_secs(3)), ..Default::default() };
        searcher.search(&Position::startpos(), limits);
    }
}
