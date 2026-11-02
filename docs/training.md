# Training the NNUE

## The loop

```
gen 0 engine (material + PST) ── self play ──► gen0.bin ──► net-001
net-001 engine                ── self play ──► gen1.bin ──► net-002
...
```

Each generation's data is produced by the best engine so far, so the network learns from progressively stronger play.

## Data format

Positions are fixed 32-byte records, so a file of any size can be memory mapped and read without parsing. The layout is defined in `engine/src/datagen/format.rs` and read by `training/nnue/data.py`. A golden-bytes test on both sides keeps the two in sync.

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | occupancy bitboard (u64 LE, a1 = bit 0) |
| 8 | 16 | one 4-bit nibble per piece in occupancy order: `colour << 3 \| type` |
| 24 | 2 | search score, centipawns, side to move's point of view (i16 LE) |
| 26 | 1 | game result, side to move's point of view: 0 loss, 1 draw, 2 win |
| 27 | 1 | side to move: 0 white, 1 black |
| 28 | 4 | reserved |

Filtered out during generation: positions in check, positions whose best move is a capture, and the first few plies of each game.

## Network

```
768 inputs ── feature transformer (768 → 256) ──┐  side to move
768 inputs ── same weights                   ───┤  opponent
                                                ▼
                               concat (512) → SCReLU → linear → 1
```

- **Inputs:** 2 sides × 6 piece types × 64 squares, seen from each side's own perspective (black's board is flipped). This lets one set of weights serve both colours.
- **SCReLU:** `clamp(x, 0, 1)²`.
- **Output:** 1.0 = 400 centipawns.

## Loss

MSE in win-probability space. The target blends the search score with the actual game result:

```
target = λ · result + (1 − λ) · sigmoid(score / 400)
loss   = (sigmoid(output) − target)²
```

`wdl_lambda` (λ) is set per config.

## Quantization

The engine runs the net in integer maths:

| Tensor | Scale | Type |
|---|---|---|
| feature transformer weights + bias | × 255 (QA) | i16 |
| output weights | × 64 (QB) | i16 |
| output bias | × 255 × 64 | i32 |

Output weights are clipped to ±127/64 during training, so they stay well inside i16 after scaling. `tests/test_export.py` checks that the integer forward pass matches the float model to within a few centipawns.

## Tracking runs

Every run writes to `training/runs/<name>/`:

| File | Committed | Contents |
|---|---|---|
| `config.toml` | yes | exact config used |
| `meta.json` | yes | git commit, device, dataset size, parameter count |
| `metrics.csv` | yes | train / val loss, learning rate, positions per second, over time |
| `summary.json` | yes | best val loss and its epoch, updated every epoch |
| `checkpoints/` | no | `last.pt` (for `--resume`) and `best.pt` |
| `tensorboard/` | no | live curves: `tensorboard --logdir training/runs` |
| `<name>.nnue` | no | exported net, published in GitHub Releases |

```bash
python -m nnue.train configs/net-002.toml            # new run
python -m nnue.train configs/net-002.toml --resume   # carry on after a stop
python -m nnue.report                                # table of every run
python -m nnue.report --markdown --plot              # for docs/results.md
```
