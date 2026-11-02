pub mod board;
pub mod evaluation;
pub mod search;
pub mod nnue;
pub mod datagen;
pub mod uci;

use uci::benchmark;
use uci::UciEngine;

fn main() {
    // "ChessProject bench [depth]" -> run the bench and exit (OpenBench / CI style)
    // anything else -> normal uci mode
    let args: Vec<String> = std::env::args().collect();

    if args.get(1).map(String::as_str) == Some("bench") {
        let depth = args.get(2).and_then(|d| d.parse().ok()).unwrap_or(benchmark::DEFAULT_BENCH_DEPTH);
        benchmark::bench(depth);
        return;
    }

    UciEngine::new().run();
}
