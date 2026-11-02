//! Principal variation search (alpha-beta with a null window trick)
//! https://www.chessprogramming.org/Principal_Variation_Search
//! https://www.youtube.com/watch?v=l-hh51ncgDI
//!
//! alpha = the best score we're already guaranteed
//! beta  = the best score the opponent will allow, anything >= beta they'd never let happen
//! When alpha >= beta -> stop searching this node (pruning)
//!
//! PVS part: if move ordering is good the first move is usually the best one.
//! So the rest only get a cheap "null window" search (alpha, alpha + 1) that just
//! answers "is this better than alpha? yes / no". Only if it says yes do we re-search properly.

use crate::board::{Move, MoveList, Position};
use crate::search::move_ordering::move_ordering::{order_moves, OrderContext};
use crate::search::searcher::{Searcher, INFINITY, MATE, MAX_PLY};
use crate::search::transposition::transposition_table::{score_from_tt, Bound};

impl Searcher {

    // Root node of the tree, don't prune here -> we need an actual best move back not just a score
    pub(in crate::search) fn search_root(&mut self, pos: &Position, moves: &MoveList, depth: i32) -> (i32, Option<Move>) {
        let mut alpha = -INFINITY;
        let beta = INFINITY;
        let mut best_move = None;

        self.position_history.push(pos.key());

        for (i, &mv) in moves.iter().enumerate() {
            let child = pos.play(mv);

            let score = if i == 0 {
                -self.alpha_beta(&child, depth - 1, -beta, -alpha, 1)
            } else {
                self.null_window_then_full(&child, depth, alpha, beta, 1)
            };

            if self.stopped {
                break;
            }

            if score > alpha {
                alpha = score;
                best_move = Some(mv);
            }
        }

        self.position_history.pop();

        if !self.stopped {
            self.tt.store(pos.key(), best_move, alpha, depth, Bound::Exact, 0);
        }
        (alpha, best_move)
    }

    pub(in crate::search) fn alpha_beta(&mut self, pos: &Position, mut depth: i32, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.nodes += 1;
        if self.should_stop() {
            return 0; // garbage score, the root throws this depth away anyway
        }

        // draws, checked before anything else
        if pos.halfmove_clock() >= 100
            || pos.is_insufficient_material()
            || self.is_repetition(pos.key(), pos.halfmove_clock()) {
            return 0;
        }

        // check extension -> don't stop searching while in check, the leaf eval would be nonsense
        let in_check = pos.in_check();
        if in_check {
            depth += 1;
        }

        if depth <= 0 || ply >= MAX_PLY - 1 {
            //return self.evaluator.evaluate(pos);
            return self.quiescence(pos, alpha, beta, ply); // better search
        }

        // PV node = full window, this is the line we actually care about
        // null window nodes (beta = alpha + 1) are just yes/no checks
        let pv_node = beta - alpha > 1;
        let original_alpha = alpha; // this will cleanup flag logic

        let key = pos.key();
        let mut tt_move = None;

        if let Some(entry) = self.tt.probe(key) {
            tt_move = entry.best_move;

            // searched this deep (or deeper) before -> maybe we already know the answer
            // not on PV nodes though, cutting there leaves holes in the main line
            if !pv_node && entry.depth >= depth {
                let score = score_from_tt(entry.score, ply);
                match entry.bound {
                    Bound::Exact => return score,
                    Bound::Lower if score >= beta => return score,
                    Bound::Upper if score <= alpha => return score,
                    _ => {}
                }
            }
        }

        let mut moves = pos.legal_moves();
        if moves.is_empty() {
            if in_check {
                // + ply so a mate in 1 scores higher than a mate in 5 -> takes the fastest mate
                return -MATE + ply as i32;
            }
            return 0; // stalemate
        }

        // Filter for good moves first -> finds our cutoff thresholds earlier
        let ctx = OrderContext {
            tt_move,
            killers: self.killers.get(ply),
            history: &self.history,
        };
        order_moves(pos, &mut moves, &ctx);

        self.position_history.push(key);
        let mut best = -INFINITY;
        let mut best_move = None;

        for (i, &mv) in moves.iter().enumerate() {
            let child = pos.play(mv);

            let score = if i == 0 {
                -self.alpha_beta(&child, depth - 1, -beta, -alpha, ply + 1)
            } else {
                self.null_window_then_full(&child, depth, alpha, beta, ply + 1)
            };

            if self.stopped {
                break;
            }

            if score > best {
                best = score;
                best_move = Some(mv);
            }

            if score > alpha {
                alpha = score;
            }

            // opponent would never let us get here -> no point looking at the other moves
            if alpha >= beta {
                // quiet move caused it -> remember it (captures already get sorted first anyway)
                if !mv.is_capture() && !mv.is_promotion() {
                    self.killers.store(ply, mv);
                    self.history.update(pos.side_to_move(), mv, depth);
                }
                break;
            }
        }

        self.position_history.pop();

        if self.stopped {
            return 0; // half searched node, don't let it into the TT
        }

        let bound = if best >= beta {
            Bound::Lower
        } else if best > original_alpha {
            Bound::Exact
        } else {
            Bound::Upper
        };
        self.tt.store(key, best_move, best, depth, bound, ply);

        best
    }

    // PVS for every move after the first one
    fn null_window_then_full(&mut self, child: &Position, depth: i32, alpha: i32, beta: i32, ply: usize) -> i32 {
        // can this move beat alpha at all?
        let mut score = -self.alpha_beta(child, depth - 1, -alpha - 1, -alpha, ply);

        // yes and it might be the new best -> search again with the real window to get the actual score
        if score > alpha && score < beta {
            score = -self.alpha_beta(child, depth - 1, -beta, -alpha, ply);
        }
        score
    }
}
