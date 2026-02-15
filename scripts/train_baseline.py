import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from tsc.utils.seed import set_seed
from tsc.data.goemotions import load_goemotions, preprocess_goemotions
from tsc.eval.metrics import multilabel_f1_metrics


def make_run_dir(base: str, run_name: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base) / f"{run_name}__{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def save_log_history_jsonl(path: Path, log_history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in log_history:
            f.write(json.dumps(row) + "\n")


def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["lr"] = float(cfg["lr"])
    cfg["weight_decay"] = float(cfg["weight_decay"])

    # ---- Run/versioning ----
    # You can set these in YAML too if you prefer:
    base_runs_dir = cfg.get("runs_dir", "outputs/runs")
    run_name = cfg.get("run_name", "baseline")
    run_dir = make_run_dir(base_runs_dir, run_name)

    # Save full config used for this run
    save_json(run_dir / "run_config.json", cfg)

    set_seed(cfg["seed"])

    # ---- Data ----
    ds = load_goemotions(cfg["dataset_config"])

    train = ds["train"].shuffle(seed=cfg["seed"]).select(range(cfg["train_subset"]))
    val = ds["validation"].shuffle(seed=cfg["seed"]).select(range(cfg["val_subset"]))

    tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)

    train_tok, num_labels, label_names = preprocess_goemotions(train, tok, cfg["max_length"])
    val_tok, _, _ = preprocess_goemotions(val, tok, cfg["max_length"])

    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    threshold = float(cfg.get("threshold", 0.2))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return multilabel_f1_metrics(logits, labels, threshold=threshold)

    # ---- Trainer args ----
    # Keep checkpoints separate from the exported final model
    checkpoints_dir = run_dir / "checkpoints"

    args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        num_train_epochs=cfg["epochs"],
        weight_decay=cfg["weight_decay"],
        logging_strategy="steps",
        logging_steps=int(cfg.get("logging_steps", 50)),
        report_to=["tensorboard"] if cfg.get("tensorboard", False) else "none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )

    # ---- Train ----
    trainer.train()

    # ---- Evaluate best model ----
    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)

    # ---- Save logs for plots ----
    logs_dir = run_dir / "logs"
    save_log_history_jsonl(logs_dir / "log_history.jsonl", trainer.state.log_history)

    # Save final metrics + a short "best summary"
    summary = {
        "run_name": run_name,
        "threshold": threshold,
        "metrics": metrics,
    }
    save_json(logs_dir / "metrics_final.json", summary)

    # ---- Export best model to a clean folder ----
    final_dir = run_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)

    # Optional: save label names explicitly too (handy for external tools)
    save_json(final_dir / "label_names.json", label_names)

    print(f"\n✅ Run saved to: {run_dir}")
    print(f"✅ Best model exported to: {final_dir}")
    print(f"✅ Logs exported to: {logs_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    main(args.config)