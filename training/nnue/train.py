# Trains one net from a config file
#
#   python -m nnue.train configs/net-001.toml
#   python -m nnue.train configs/net-001.toml --resume     (carries on from runs/net-001/checkpoints/last.pt)
#
# when it finishes the best checkpoint is exported to runs/<name>/<name>.nnue for the engine

import argparse
import math
import tomllib
from pathlib import Path

import numpy as np
import torch

from . import data, export
from .loss import wdl_loss
from .model import NNUE
from .tracking import RunTracker

ROOT = Path(__file__).resolve().parent.parent  # training/


def evaluate(model, records, batch_size, wdl_lambda, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for stm, nstm, score, result in data.batches(records, batch_size, shuffle=False):
            out = model(stm.to(device), nstm.to(device))
            n = len(score)
            total += wdl_loss(out, score.to(device), result.to(device), wdl_lambda).item() * n
            count += n
    model.train()
    return total / max(count, 1)


def train(config_path, resume=False, runs_dir=None):
    config = tomllib.loads(Path(config_path).read_text())
    name = config["name"]
    cfg_t, cfg_m = config["train"], config["model"]

    torch.manual_seed(cfg_t["seed"])
    rng = np.random.default_rng(cfg_t["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data path is relative to training/ so configs work from anywhere
    data_path = Path(config["data"]["path"])
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    train_set, val_set = data.split(data.load(data_path), config["data"]["val_fraction"])

    model = NNUE(hidden=cfg_m["hidden"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_t["lr"], weight_decay=cfg_t["weight_decay"])

    # cosine decay down to 1% of the starting lr, one step per epoch
    epochs = cfg_t["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * e / epochs)))

    run_dir = Path(runs_dir or ROOT / "runs") / name
    tracker = RunTracker(run_dir, config_path, resume=resume)

    start_epoch, step, positions = 1, 0, 0
    if resume and (tracker.checkpoints / "last.pt").exists():
        ckpt = torch.load(tracker.checkpoints / "last.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, step, positions = ckpt["epoch"] + 1, ckpt["step"], ckpt["positions"]
        print(f"resuming {name} from epoch {start_epoch}")
    else:
        tracker.write_meta(
            device=device,
            data=str(data_path),
            train_positions=len(train_set),
            val_positions=len(val_set),
            parameters=model.parameter_count(),
        )

    print(f"{name}: {len(train_set):,} train / {len(val_set):,} val positions, "
          f"{model.parameter_count():,} params on {device}")

    batch_size, wdl_lambda = cfg_t["batch_size"], cfg_t["wdl_lambda"]
    log_every = cfg_t["log_every"]

    for epoch in range(start_epoch, epochs + 1):
        running, running_n = 0.0, 0

        for stm, nstm, score, result in data.batches(train_set, batch_size, rng=rng):
            out = model(stm.to(device), nstm.to(device))
            loss = wdl_loss(out, score.to(device), result.to(device), wdl_lambda)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.clip_weights()

            step += 1
            positions += len(score)
            running += loss.item()
            running_n += 1

            if step % log_every == 0:
                tracker.log(epoch, step, positions, running / running_n, scheduler.get_last_lr()[0])
                running, running_n = 0.0, 0

        # end of epoch, val loss is the number that decides "best"
        val_loss = evaluate(model, val_set, batch_size, wdl_lambda, device)
        train_loss = running / running_n if running_n else float("nan")
        tracker.log(epoch, step, positions, train_loss, scheduler.get_last_lr()[0], val_loss=val_loss)

        scheduler.step()
        is_best = tracker.end_epoch(epoch, val_loss, {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "epoch": epoch,
            "step": step,
            "positions": positions,
        })
        if is_best:
            print(f"  new best val loss {val_loss:.5f}")

    tracker.close()

    net_path = run_dir / f"{name}.nnue"
    export.write(export.load_checkpoint(tracker.checkpoints / "best.pt"), net_path)
    print(f"done, best epoch {tracker.summary['best_epoch']} -> {net_path}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="train an NNUE net")
    parser.add_argument("config")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
