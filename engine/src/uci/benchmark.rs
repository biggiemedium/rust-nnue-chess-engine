// bench + perft
//
// bench -> search a fixed set of positions to a fixed depth, print total nodes + nps
//   same nodes every run = the search didn't change, faster nps = it got faster
//   "ChessProject bench" from the command line or "bench" inside uci
//
// perft -> count every legal move sequence to depth N, compare against known numbers
//   https://www.chessprogramming.org/Perft_Results
//   if these are off move generation is broken and nothing else matters

use std::time::Instant;

use crate::board::Position;
use crate::search::searcher::{SearchLimits, Searcher};

pub const DEFAULT_BENCH_DEPTH: i32 = 8;

// mix of openings, middlegames + endgames (most are the perft test positions)
const BENCH_FENS: [&str; 10] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP1B1PPP/R2QKB1R w KQ - 0 8",
    "2r3k1/pp3ppp/4p3/3p4/3P4/2P1P3/P4PPP/2R3K1 w - - 0 25",
    "8/8/4k3/3p4/3P4/4K3/8/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/1Q3PPP/6K1 w - - 0 1",
];

pub fn bench(depth: i32) {
    let mut searcher = Searcher::new();
    searcher.print_info = false;

    let start = Instant::now();
    let mut total_nodes = 0;

    for fen in BENCH_FENS {
        searcher.new_game(); // every position starts cold -> same result every run
        let pos = Position::from_fen(fen).expect("bad bench fen");
        let result = searcher.search(&pos, SearchLimits { depth: Some(depth), ..Default::default() });
        total_nodes += result.nodes;
    }

    let elapsed = start.elapsed().as_secs_f64().max(0.001);
    println!("{} nodes {} nps", total_nodes, (total_nodes as f64 / elapsed) as u64);
}

pub fn perft(pos: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let moves = pos.legal_moves();
    if depth == 1 {
        return moves.len() as u64; // bulk counting, no need to play the last move
    }

    moves.iter().map(|&mv| perft(&pos.play(mv), depth - 1)).sum()
}

// "go perft N" -> nodes under every root move, handy for finding which move is broken
pub fn perft_divide(pos: &Position, depth: u32) {
    let start = Instant::now();
    let mut total = 0;

    for mv in pos.legal_moves() {
        let nodes = perft(&pos.play(mv), depth.saturating_sub(1));
        println!("{}: {}", mv.to_uci_string(), nodes);
        total += nodes;
    }

    println!("\nNodes searched: {} ({} ms)", total, start.elapsed().as_millis());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perft_fen(fen: &str, depth: u32) -> u64 {
        perft(&Position::from_fen(fen).unwrap(), depth)
    }

    // numbers from https://www.chessprogramming.org/Perft_Results
    #[test]
    fn perft_startpos() {
        assert_eq!(perft(&Position::startpos(), 4), 197_281);
    }

    #[test]
    fn perft_kiwipete() {
        // castling, en passant, promotions, pins -> the one that catches most bugs
        assert_eq!(perft_fen(BENCH_FENS[1], 3), 97_862);
    }

    #[test]
    fn perft_position_3() {
        assert_eq!(perft_fen(BENCH_FENS[2], 4), 43_238);
    }

    #[test]
    fn perft_position_4() {
        assert_eq!(perft_fen(BENCH_FENS[3], 3), 9_467);
    }

    #[test]
    fn perft_position_5() {
        assert_eq!(perft_fen(BENCH_FENS[4], 3), 62_379);
    }
}
