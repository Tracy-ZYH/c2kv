from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_history(checkpoint_dir: Path) -> List[Dict[str, Any]]:
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.exists():
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint-*/trainer_state.json"),
            key=lambda path: int(path.parent.name.split("-")[-1]) if path.parent.name.split("-")[-1].isdigit() else -1,
        )
        if not checkpoints:
            raise FileNotFoundError(f"No trainer_state.json found under {checkpoint_dir}")
        state_path = checkpoints[-1]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    history = state.get("log_history", [])
    if not isinstance(history, list):
        raise ValueError(f"Invalid log_history in {state_path}")
    return [row for row in history if isinstance(row, dict)]


def _write_csv(path: Path, history: List[Dict[str, Any]]) -> None:
    keys = [
        "step",
        "epoch",
        "loss",
        "eval_loss",
        "learning_rate",
        "grad_norm",
        "label_tokens",
        "eval_runtime",
        "eval_samples_per_second",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in keys})


def _series(history: List[Dict[str, Any]], key: str) -> tuple[List[float], List[float]]:
    xs, ys = [], []
    for index, row in enumerate(history):
        if key not in row:
            continue
        x = row.get("step", index)
        y = row.get(key)
        try:
            xs.append(float(x))
            ys.append(float(y))
        except (TypeError, ValueError):
            continue
    return xs, ys


def _plot(path: Path, history: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("matplotlib is required for plotting. Install it or use --csv_only True.") from error

    metrics = [
        ("loss", "train loss"),
        ("eval_loss", "eval loss"),
        ("learning_rate", "learning rate"),
        ("grad_norm", "grad norm"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (key, title) in zip(axes.flat, metrics):
        xs, ys = _series(history, key)
        axis.plot(xs, ys, marker="o", markersize=2, linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("step")
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Hugging Face Trainer log_history from trainer_state.json.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_png")
    parser.add_argument("--output_csv")
    parser.add_argument("--csv_only", type=lambda x: str(x).lower() == "true", default=False)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    history = _load_history(checkpoint_dir)
    output_png = Path(args.output_png or checkpoint_dir / "trainer_curves.png")
    output_csv = Path(args.output_csv or checkpoint_dir / "trainer_log_history.csv")
    _write_csv(output_csv, history)
    if not args.csv_only:
        _plot(output_png, history)
    print(json.dumps({
        "checkpoint_dir": str(checkpoint_dir),
        "num_log_rows": len(history),
        "output_csv": str(output_csv),
        "output_png": None if args.csv_only else str(output_png),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
