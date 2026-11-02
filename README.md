# Chess-Project

A chess engine written in Rust that learns its own evaluation.
The engine plays itself -> PyTorch trains an **NNUE** on those games -> the new net goes back into the engine -> repeat.
No Stockfish data, everything it knows it learned from playing itself.


## How it fits together


<img width="1542" height="896" alt="chess ai architecture drawio" src="https://github.com/user-attachments/assets/b351df16-5589-49a6-a52b-835d8e1d75fd" />


1. **Self play** -> the engine plays itself and writes every position, the search score and the final result to a binary file (32 bytes a position so python can memmap millions of them without parsing a single FEN)
2. **Train** -> PyTorch trains the net on that file. Every run saves its config, loss curves and best checkpoint so nothing gets lost
3. **Export** -> the net gets quantized to ints so the engine can run it with integer maths, and only the 2-4 pieces that moved get updated each move instead of the whole board (that's the "efficiently updatable" in NNUE)
4. **Repeat** -> the stronger engine makes better data for the next net

Gen 0 has no net yet so it starts from a plain material + piece-square table eval, after that it's nets all the way down.

## Results

Every new net plays the last one, see [docs/results.md](docs/results.md) for the full training history + loss curves.

| Net | Data | Val loss | Elo vs previous |
|---|---|---|---|
| baseline (material + PST) | — | — | — |

## What's in the engine

- **Search** -> iterative deepening, principal variation search, quiescence search (so it doesn't stop right after QxP and miss the PxQ)
- **Move ordering** -> transposition table move, MVV-LVA, static exchange evaluation, killer moves, history heuristic
- **Pruning** -> alpha-beta, delta pruning, SEE to skip captures that just hang material
- **UCI** -> works in any chess GUI, time management, `go infinite` / `stop`, perft + bench
- Move generation is checked against the known [perft numbers](https://www.chessprogramming.org/Perft_Results), if those are off nothing else matters

Speed right now: ~4-6M nodes/sec on one thread, depth 9 from the start position in about 1.5s.

## Repository layout

| Path | What |
|---|---|
| `engine/` | Rust engine -> board, search, NNUE inference, self play data generation, UCI |
| `training/` | PyTorch trainer, configs, every training run's history, tests |
| `bot/` | Lichess bot, runs the engine over UCI |
| `docs/` | [Architecture](docs/architecture.md), [how the net is trained](docs/training.md), [results](docs/results.md) |

## Running it

```bash
# engine
cd engine && cargo build --release && cargo test
./target/release/ChessProject bench        # same positions + depth every time -> prints nodes + nps

# training
cd training
pip install -e ".[tracking,dev]"
pytest
python -m nnue.train configs/net-001.toml
python -m nnue.report --plot               # table of every run + loss curves
```

## Sources

Most of what's in here I learned from these:
- https://www.chessprogramming.org/Main_Page
- https://www.chessprogramming.org/NNUE
- https://www.youtube.com/watch?v=l-hh51ncgDI
