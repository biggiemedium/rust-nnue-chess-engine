# Chess-Project

A chess engine written in Rust with an **NNUE** evaluation trained from scratch in **PyTorch** on the engine's own self-play games, plus a bot that plays it on Lichess.

> Status: early development. The engine core and the training pipeline are being built. See the [roadmap](#roadmap).

## How it fits together

```
 ┌─────────────── engine/ (Rust) ───────────────┐        ┌──────── training/ (Python) ────────┐
 │                                              │        │                                    │
 │  search ──► nnue eval (int, incremental)     │        │  data.py   memmap 32-byte records  │
 │    │              ▲                          │        │  model.py  (768→256)×2 → 1         │
 │    ▼              │ loads net-XXX.nnue ◄─────┼────────┤  train.py  + run tracking          │
 │  datagen (self play) ──► positions.bin ──────┼───────►│  export.py quantize → .nnue        │
 │                                              │        │                                    │
 └──────────────────────▲───────────────────────┘        └────────────────────────────────────┘
                        │ UCI
                  bot/ (Lichess)
```

1. The engine plays itself and writes positions, search scores and game results to a compact binary file.
2. PyTorch trains a network on that data. Every run's config, metrics and best checkpoint are recorded.
3. The trained net is quantized to integers and loaded by the engine, which updates it incrementally as moves are played.
4. The new engine generates better data for the next net, and the loop repeats.

## Results

Each net plays the previous one. See [docs/results.md](docs/results.md) for full training history and loss curves.

| Net | Data | Val loss | Elo vs previous |
|---|---|---|---|
| baseline (material + PST) | — | — | — |

## Repository layout

| Path | What |
|---|---|
| `engine/` | Rust engine: board, search, NNUE inference, self-play data generation, UCI |
| `training/` | PyTorch trainer, configs, training-run history, tests |
| `bot/` | Lichess bot that runs the engine over UCI |
| `docs/` | [Architecture](docs/architecture.md), [training write-up](docs/training.md), [results](docs/results.md) |

## Running it

```bash
# engine
cd engine && cargo build --release && cargo test
./target/release/ChessProject bench        # fixed-depth search, prints nodes + nps

# training
cd training
pip install -e ".[tracking,dev]"
pytest
python -m nnue.train configs/net-001.toml
python -m nnue.report --plot
```

## Roadmap

- [x] Board representation (shakmaty wrapper, copy-make)
- [x] Training pipeline: data format, model, loss, export, run tracking
- [x] Search: iterative deepening, PVS, quiescence, MVV-LVA ordering, draw detection
- [x] Search: transposition table, killer moves, history heuristic, SEE, delta pruning
- [ ] Search: lazy SMP, null move pruning, late move reductions
- [x] UCI protocol: time management, `go infinite`/`stop`, Hash / Move Overhead options, perft + bench
- [ ] Self-play data generation (gen 0 on material + PST)
- [ ] NNUE inference in Rust + cross-language test against PyTorch
- [ ] net-001, then iterate
- [ ] Lichess bot live
