# runs a tiny real training job end to end, catches anything broken in the pipeline + tracking

import csv
import json

from nnue.train import train

CONFIG = """
name = "smoke"

[data]
path = "{data}"
val_fraction = 0.1

[model]
hidden = 32

[train]
epochs = {epochs}
batch_size = 256
lr = 1e-3
weight_decay = 0.0
wdl_lambda = 0.3
seed = 1
log_every = 5
"""


def write_config(tmp_path, dataset, epochs):
    path = tmp_path / "smoke.toml"
    path.write_text(CONFIG.format(data=dataset.as_posix(), epochs=epochs))
    return path


def test_train_writes_progress(tmp_path, dataset):
    run = train(write_config(tmp_path, dataset, epochs=2), runs_dir=tmp_path / "runs")

    for name in ["config.toml", "meta.json", "metrics.csv", "summary.json",
                 "checkpoints/last.pt", "checkpoints/best.pt", "smoke.nnue"]:
        assert (run / name).exists(), name

    with open(run / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    assert len([r for r in rows if r["val_loss"]]) == 2  # one val row per epoch

    summary = json.loads((run / "summary.json").read_text())
    assert summary["epochs_done"] == 2 and summary["best_val_loss"] is not None


def test_resume_carries_on(tmp_path, dataset):
    runs = tmp_path / "runs"
    train(write_config(tmp_path, dataset, epochs=1), runs_dir=runs)
    run = train(write_config(tmp_path, dataset, epochs=2), resume=True, runs_dir=runs)

    summary = json.loads((run / "summary.json").read_text())
    assert summary["epochs_done"] == 2
