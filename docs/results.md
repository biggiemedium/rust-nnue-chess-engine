# Results

## Training runs

Generated with `python -m nnue.report --markdown`.

| run | commit | train pos | epochs | best val loss | best epoch | time |
|---|---|---|---|---|---|---|

## Playing strength

Each net plays the previous best. Elo is measured with SPRT (a sequential test that stops once the result is statistically clear).

| Net | Opponent | Games | Elo | ± |
|---|---|---|---|---|

## Loss curves

`python -m nnue.report --plot` writes `training/runs/loss_curves.png`. Copy it to `docs/` to show it here.
