//! Quiescence search
//! https://www.chessprogramming.org/Quiescence_Search
//!
//! Horizon effect: if the search stops at depth 0 right after QxP it thinks it's up a pawn,
//! it never sees the PxQ next move. So at the leaves we keep searching captures only
//! until the position goes "quiet", then evaluate.

use crate::board::{PieceType, Position};
use crate::evaluation::material::material::piece_value;
use crate::search::move_ordering::move_ordering::order_captures;
use crate::search::searcher::Searcher;
use crate::search::static_exchange_evaluation::see;

// safety margin on top of the biggest capture for delta pruning
const DELTA_MARGIN: i32 = 200;

impl Searcher {

    pub(in crate::search) fn quiescence(&mut self, pos: &Position, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.nodes += 1;
        if self.should_stop() {
            return 0;
        }

        // "stand pat" -> we don't HAVE to capture, so the static eval is the worst case for us
        let stand_pat = self.evaluator.evaluate(pos);

        // Position is already too good for us -> no need to explore
        if stand_pat >= beta {
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        // Delta pruning -> only continue if a capture could improve alpha
        // https://www.chessprogramming.org/Delta_Pruning
        // if(static_eval + Δ < α) return α
        // Δ = a whole queen + margin, if winning a queen can't save us nothing will
        // TODO: turn off in the endgame, promotions can be worth more than Δ there
        let big_delta = piece_value(PieceType::Queen) + DELTA_MARGIN;
        if stand_pat + big_delta < alpha {
            return alpha;
        }

        // TODO: when in check stand pat isn't allowed, every evasion needs searching (not just captures)

        let mut captures = pos.capture_moves();
        order_captures(&mut captures); // MVV-LVA

        for &mv in captures.iter() {
            // skip bad captures
            // "If I take this -> and it's obviously defended by something cheaper -> don't even try"
            if see(pos, mv) < 0 {
                continue;
            }

            let child = pos.play(mv);
            let score = -self.quiescence(&child, -beta, -alpha, ply + 1);

            if self.stopped {
                return 0;
            }

            // This is a good move but we anticipate that our opponent
            // won't let us take the piece
            if score >= beta {
                return score;
            }
            if score > alpha {
                alpha = score;
            }
        }

        alpha
    }
}
