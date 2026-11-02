# Keeps a record of every training run so progress is never lost
#
# runs/<name>/
#   config.toml       exact config the run used
#   meta.json         git commit, device, dataset size, param count, start time
#   metrics.csv       one row per log step (loss, speed, lr ...), this is what the plots read
#   summary.json      best val loss + which epoch, written as the run goes
#   checkpoints/      last.pt every epoch, best.pt on a new best val loss  (gitignored)
#   tensorboard/      only if tensorboard is installed                     (gitignored)
#
# metrics.csv + summary.json are small so they get committed, that way the github repo
# shows the full history of every net even though the weights live in Releases

import csv
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

FIELDS = ["epoch", "step", "positions", "train_loss", "val_loss", "lr", "pos_per_sec", "elapsed_s"]


def git_commit():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip() != ""
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class RunTracker:

    def __init__(self, run_dir, config_path, resume=False):
        self.dir = Path(run_dir)
        self.checkpoints = self.dir / "checkpoints"
        self.metrics_path = self.dir / "metrics.csv"
        self.summary_path = self.dir / "summary.json"

        if self.dir.exists() and not resume and self.metrics_path.exists():
            raise FileExistsError(f"{self.dir} already has a run, pass --resume or pick a new name")

        self.checkpoints.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config_path, self.dir / "config.toml")

        self.summary = {"best_val_loss": None, "best_epoch": None, "epochs_done": 0}
        if resume and self.summary_path.exists():
            self.summary = json.loads(self.summary_path.read_text())

        if not self.metrics_path.exists():
            with open(self.metrics_path, "w", newline="") as f:
                csv.writer(f).writerow(FIELDS)

        # tensorboard is nice to have, not required
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(self.dir / "tensorboard")
        except ImportError:
            pass

        self.start = time.time() - self.summary.get("elapsed_s", 0)
        self.last_log_time = time.time()
        self.last_log_positions = 0

    def write_meta(self, **info):
        meta = {
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "torch": torch.__version__,
            **info,
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))

    def log(self, epoch, step, positions, train_loss, lr, val_loss=None):
        now = time.time()
        speed = (positions - self.last_log_positions) / max(now - self.last_log_time, 1e-9)
        self.last_log_time, self.last_log_positions = now, positions
        elapsed = now - self.start

        row = [epoch, step, positions, f"{train_loss:.6f}", "" if val_loss is None else f"{val_loss:.6f}",
               f"{lr:.2e}", f"{speed:.0f}", f"{elapsed:.1f}"]
        with open(self.metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        if self.tb:
            self.tb.add_scalar("loss/train", train_loss, positions)
            self.tb.add_scalar("lr", lr, positions)
            self.tb.add_scalar("speed/pos_per_sec", speed, positions)
            if val_loss is not None:
                self.tb.add_scalar("loss/val", val_loss, positions)

        val = "" if val_loss is None else f"  val {val_loss:.5f}"
        print(f"epoch {epoch:3d}  step {step:7d}  train {train_loss:.5f}{val}  {speed:,.0f} pos/s  {elapsed / 60:.1f} min")

    def end_epoch(self, epoch, val_loss, state):
        # state = everything needed to resume (model, optimizer, scheduler, config, epoch)
        torch.save(state, self.checkpoints / "last.pt")

        best = self.summary["best_val_loss"]
        is_best = best is None or val_loss < best
        if is_best:
            torch.save(state, self.checkpoints / "best.pt")
            self.summary["best_val_loss"] = val_loss
            self.summary["best_epoch"] = epoch

        self.summary["epochs_done"] = epoch
        self.summary["elapsed_s"] = round(time.time() - self.start, 1)
        self.summary_path.write_text(json.dumps(self.summary, indent=2))
        return is_best

    def close(self):
        if self.tb:
            self.tb.close()
