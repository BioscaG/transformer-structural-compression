import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def extract_series(rows: list[dict[str, Any]]):
    """
    HuggingFace Trainer log_history has mixed entries:
    - training logs: {"loss": ..., "learning_rate": ..., "grad_norm": ..., "epoch": ..., "step": ...}
    - eval logs: {"eval_loss": ..., "eval_f1_micro": ..., "eval_f1_macro": ..., "epoch": ...}
    We extract series for plotting.
    """
    # training
    train_steps, train_loss = [], []
    lr_steps, lr_vals = [], []
    gn_steps, gn_vals = [], []

    # eval (per epoch)
    eval_epoch, eval_loss = [], []
    eval_f1_micro, eval_f1_macro = [], []
    eval_avg_pred = []

    for r in rows:
        step = r.get("step", None)
        epoch = r.get("epoch", None)

        # training loss
        if "loss" in r and step is not None:
            v = safe_float(r["loss"])
            if v is not None:
                train_steps.append(int(step))
                train_loss.append(v)

        # lr
        if "learning_rate" in r and step is not None:
            v = safe_float(r["learning_rate"])
            if v is not None:
                lr_steps.append(int(step))
                lr_vals.append(v)

        # grad norm
        if "grad_norm" in r and step is not None:
            v = safe_float(r["grad_norm"])
            if v is not None:
                gn_steps.append(int(step))
                gn_vals.append(v)

        # eval metrics
        if "eval_loss" in r and epoch is not None:
            e = safe_float(epoch)
            if e is None:
                continue
            eval_epoch.append(e)
            eval_loss.append(float(r["eval_loss"]))

            if "eval_f1_micro" in r:
                eval_f1_micro.append(float(r["eval_f1_micro"]))
            else:
                eval_f1_micro.append(None)

            if "eval_f1_macro" in r:
                eval_f1_macro.append(float(r["eval_f1_macro"]))
            else:
                eval_f1_macro.append(None)

            if "eval_avg_pred_labels" in r:
                eval_avg_pred.append(float(r["eval_avg_pred_labels"]))
            else:
                eval_avg_pred.append(None)

    return {
        "train": {"steps": train_steps, "loss": train_loss},
        "lr": {"steps": lr_steps, "vals": lr_vals},
        "grad_norm": {"steps": gn_steps, "vals": gn_vals},
        "eval": {
            "epoch": eval_epoch,
            "loss": eval_loss,
            "f1_micro": eval_f1_micro,
            "f1_macro": eval_f1_macro,
            "avg_pred": eval_avg_pred,
        },
    }


def plot_and_save(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot training curves from HF Trainer log_history.jsonl.")
    ap.add_argument(
        "--run_dir",
        required=True,
        help="Path to a single run directory (e.g., outputs/runs/<name>__<timestamp>)",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Where to save plots (default: <run_dir>/plots)",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively (useful in local dev).",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    log_path = run_dir / "logs" / "log_history.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"Could not find {log_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(log_path)
    series = extract_series(rows)

    # 1) Train loss vs step
    if series["train"]["steps"]:
        fig = plt.figure()
        plt.plot(series["train"]["steps"], series["train"]["loss"])
        plt.xlabel("Step")
        plt.ylabel("Train loss")
        plt.title("Train loss vs step")
        if args.show:
            plt.show()
        plot_and_save(fig, out_dir / "train_loss_vs_step.png")

    # 2) Eval loss vs epoch
    if series["eval"]["epoch"]:
        fig = plt.figure()
        plt.plot(series["eval"]["epoch"], series["eval"]["loss"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Eval loss")
        plt.title("Eval loss vs epoch")
        if args.show:
            plt.show()
        plot_and_save(fig, out_dir / "eval_loss_vs_epoch.png")

    # 3) Eval F1 micro/macro vs epoch
    if series["eval"]["epoch"] and any(v is not None for v in series["eval"]["f1_micro"]):
        fig = plt.figure()
        # filter None safely by plotting only where not None
        ep = series["eval"]["epoch"]
        f1m = series["eval"]["f1_micro"]
        f1M = series["eval"]["f1_macro"]

        if any(v is not None for v in f1m):
            xs = [ep[i] for i, v in enumerate(f1m) if v is not None]
            ys = [v for v in f1m if v is not None]
            plt.plot(xs, ys, marker="o", label="F1 micro")

        if any(v is not None for v in f1M):
            xs = [ep[i] for i, v in enumerate(f1M) if v is not None]
            ys = [v for v in f1M if v is not None]
            plt.plot(xs, ys, marker="o", label="F1 macro")

        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.title("Eval F1 vs epoch")
        plt.legend()
        if args.show:
            plt.show()
        plot_and_save(fig, out_dir / "eval_f1_vs_epoch.png")

    # 4) Avg predicted labels vs epoch (sanity)
    if series["eval"]["epoch"] and any(v is not None for v in series["eval"]["avg_pred"]):
        fig = plt.figure()
        ep = series["eval"]["epoch"]
        apred = series["eval"]["avg_pred"]
        xs = [ep[i] for i, v in enumerate(apred) if v is not None]
        ys = [v for v in apred if v is not None]
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Avg predicted labels")
        plt.title("Avg predicted labels vs epoch")
        if args.show:
            plt.show()
        plot_and_save(fig, out_dir / "avg_pred_labels_vs_epoch.png")

    # 5) LR vs step (optional)
    if series["lr"]["steps"]:
        fig = plt.figure()
        plt.plot(series["lr"]["steps"], series["lr"]["vals"])
        plt.xlabel("Step")
        plt.ylabel("Learning rate")
        plt.title("Learning rate vs step")
        if args.show:
            plt.show()
        plot_and_save(fig, out_dir / "lr_vs_step.png")

    # 6) Grad norm vs step (optional)
    if series["grad_norm"]["steps"]:
        fig = plt.figure()
        plt.plot(series["grad_norm"]["steps"], series["grad_norm"]["vals"])
        plt.xlabel("Step")
        plt.ylabel("Grad norm")
        plt.title("Grad norm vs step")
        if args.show:
            plt.show()
        plot_and_save(fig, out_dir / "grad_norm_vs_step.png")

    print(f"✅ Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()