# Progress across every run, reads runs/*/summary.json + metrics.csv
#
#   python -m nnue.report               table of all runs in the terminal
#   python -m nnue.report --markdown    same table as markdown, paste into docs/results.md
#   python -m nnue.report --plot        loss curves -> runs/loss_curves.png (needs matplotlib)

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = ["run", "commit", "train pos", "epochs", "best val loss", "best epoch", "time"]


def collect(runs_dir):
    rows = []
    for run in sorted(Path(runs_dir).iterdir()):
        summary_path = run / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
        best = summary.get("best_val_loss")
        rows.append([
            run.name,
            meta.get("git_commit", "?"),
            f"{meta.get('train_positions', 0):,}",
            str(summary.get("epochs_done", 0)),
            "" if best is None else f"{best:.5f}",
            str(summary.get("best_epoch") or ""),
            f"{summary.get('elapsed_s', 0) / 60:.1f} min",
        ])
    return rows


def table(rows, markdown=False):
    if markdown:
        lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(lines)

    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c) for i, c in enumerate(COLUMNS)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    return "\n".join([fmt.format(*COLUMNS)] + [fmt.format(*r) for r in rows])


def plot(runs_dir, out):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for run in sorted(Path(runs_dir).iterdir()):
        metrics = run / "metrics.csv"
        if not metrics.exists():
            continue
        with open(metrics) as f:
            val = [(int(r["positions"]), float(r["val_loss"])) for r in csv.DictReader(f) if r["val_loss"]]
        if val:
            ax.plot(*zip(*val), marker="o", label=run.name)

    ax.set_xlabel("positions seen")
    ax.set_ylabel("validation loss")
    ax.set_title("NNUE training progress")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description="summarise every training run")
    parser.add_argument("--runs", default=ROOT / "runs")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if not Path(args.runs).exists():
        print("no runs yet")
        return
    print(table(collect(args.runs), markdown=args.markdown))
    if args.plot:
        plot(args.runs, Path(args.runs) / "loss_curves.png")


if __name__ == "__main__":
    main()
