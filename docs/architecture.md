# Engine architecture

## Board (`engine/src/board`)

- `shakmaty_adapter.rs` is the only file that uses the `shakmaty` crate. It exposes engine-owned types (`Move`, `Color`, `Square`, raw `u64` bitboards).
- `Position` wraps the adapter and caches the zobrist key. Moves use copy-make: `play()` returns a new position instead of mutating, which keeps search and threading simple.

## Search (`engine/src/search`)

Planned: iterative deepening → principal variation search → quiescence, with a transposition table, MVV-LVA / killer / history move ordering, static exchange evaluation for captures, and lazy SMP threading.

## Evaluation

- `engine/src/nnue`: the trained network. `features.rs` defines the input encoding shared with the trainer. `accumulator.rs` holds the incrementally updated first layer. `network.rs` loads the quantized weights and runs inference.
- `engine/src/evaluation`: material + piece-square tables only. It's used for generation-0 data and as the baseline every net is measured against.

## Data generation (`engine/src/datagen`)

Self-play writing 32-byte training records. See [training.md](training.md#data-format).

## UCI (`engine/src/uci`)

Command parsing, engine options, and a `bench` command for consistent nodes-per-second numbers.
