// Move ordering, the better the order the more alpha-beta can prune
// best case alpha-beta only looks at ~sqrt(N) of the tree, worst case (best move last) it's plain minimax
//
// Order:
//   1. TT move (or last iteration's best at the root)
//   2. good captures (SEE >= 0), MVV-LVA between them
//   3. promotions
//   4. killer 1, killer 2
//   5. quiet moves by history score
//   6. losing captures (SEE < 0), low but not last... actually last, a quiet move is usually better than hanging a piece

use crate::board::{Move, MoveList, Position};
use crate::search::move_ordering::history_heuristic::HistoryHeuristic;
use crate::search::move_ordering::mvv_lva::mvv_lva;
use crate::search::static_exchange_evaluation::see;

const TT_MOVE: i32 = 1_000_000;
const GOOD_CAPTURE: i32 = 200_000;
const PROMOTION: i32 = 150_000;
const KILLER_1: i32 = 90_000;
const KILLER_2: i32 = 80_000; // history is capped under this (history_heuristic.rs)
const BAD_CAPTURE: i32 = -200_000;

pub struct OrderContext<'a> {
    pub tt_move: Option<Move>,
    pub killers: [Option<Move>; 2],
    pub history: &'a HistoryHeuristic,
}

pub fn order_moves(pos: &Position, moves: &mut MoveList, ctx: &OrderContext) {
    // higher score first
    // sort_by_cached_key -> scores each move once instead of every comparison (SEE isn't free)
    let side = pos.side_to_move();
    moves.sort_by_cached_key(|&mv| -score_move(pos, mv, ctx, side));
}

// quiescence only has captures, MVV-LVA is enough there (bad ones get skipped by SEE anyway)
pub fn order_captures(moves: &mut MoveList) {
    moves.sort_by_cached_key(|&mv| -mvv_lva(mv));
}

fn score_move(pos: &Position, mv: Move, ctx: &OrderContext, side: crate::board::Color) -> i32 {
    // if we found this was our best move last time -> try it first
    if Some(mv) == ctx.tt_move {
        return TT_MOVE;
    }

    if mv.is_capture() {
        // This is a losing capture e.g -> (QxP defended by pawn)
        if see(pos, mv) < 0 {
            return BAD_CAPTURE + mvv_lva(mv);
        }
        return GOOD_CAPTURE + mvv_lva(mv);
    }

    if mv.is_promotion() {
        return PROMOTION;
    }

    // "killer" moves
    if Some(mv) == ctx.killers[0] {
        return KILLER_1;
    }
    if Some(mv) == ctx.killers[1] {
        return KILLER_2;
    }

    ctx.history.get(side, mv)
}
