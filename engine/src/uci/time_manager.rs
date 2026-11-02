// Time management -> how long to think about this move
// https://www.chessprogramming.org/Time_Management
//
// Simple version: split the clock over the moves we expect are left, plus most of the increment
//   time_for_move = time_left / moves_left + increment * 3/4
// never more than half the clock in one go, and always keep the move overhead back
//
// the search itself won't start a new depth past half of this (iterative_deepening.rs)
// so most moves actually use way less than the full amount

use std::time::Duration;

use crate::search::searcher::SearchLimits;
use crate::uci::command_parser::GoParams;

// if the GUI doesn't say how many moves until the next time control, assume this many
const DEFAULT_MOVES_TO_GO: i64 = 30;

pub fn limits(go: &GoParams, white_to_move: bool, move_overhead_ms: i64) -> SearchLimits {
    let mut limits = SearchLimits {
        depth: go.depth,
        nodes: go.nodes,
        movetime: None,
    };

    // "go infinite" -> search until "stop", no clock at all
    if go.infinite {
        return limits;
    }

    if let Some(movetime) = go.movetime {
        limits.movetime = Some(ms(movetime - move_overhead_ms));
        return limits;
    }

    let (time, inc) = if white_to_move { (go.wtime, go.winc) } else { (go.btime, go.binc) };

    if let Some(time) = time {
        let usable = (time - move_overhead_ms).max(0);
        let moves_left = go.movestogo.unwrap_or(DEFAULT_MOVES_TO_GO).max(1);

        let target = usable / moves_left + inc.unwrap_or(0).max(0) * 3 / 4;
        limits.movetime = Some(ms(target.min(usable / 2)));
    }

    // no clock, no depth, no nodes ("go" on its own) -> stays unlimited until "stop"
    limits
}

fn ms(millis: i64) -> Duration {
    Duration::from_millis(millis.max(1) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn movetime_minus_overhead() {
        let go = GoParams { movetime: Some(1000), ..Default::default() };
        assert_eq!(limits(&go, true, 30).movetime, Some(Duration::from_millis(970)));
    }

    #[test]
    fn uses_own_clock() {
        // 60s + 1s inc for white, 30 moves left -> (60000 - 30) / 30 + 1000 * 3/4
        let go = GoParams { wtime: Some(60_000), btime: Some(1_000), winc: Some(1_000), binc: Some(1_000), ..Default::default() };
        assert_eq!(limits(&go, true, 30).movetime, Some(Duration::from_millis(59_970 / 30 + 750)));
        assert!(limits(&go, false, 30).movetime.unwrap() < Duration::from_millis(500));
    }

    #[test]
    fn never_more_than_half_the_clock() {
        // 1 move to go would mean "use it all", still capped at half
        let go = GoParams { wtime: Some(10_000), movestogo: Some(1), ..Default::default() };
        assert_eq!(limits(&go, true, 0).movetime, Some(Duration::from_millis(5_000)));
    }

    #[test]
    fn infinite_has_no_clock() {
        let go = GoParams { infinite: true, wtime: Some(10_000), ..Default::default() };
        assert_eq!(limits(&go, true, 0).movetime, None);
    }
}
