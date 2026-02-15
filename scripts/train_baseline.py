import argparse
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


def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["lr"] = float(cfg["lr"])
    cfg["weight_decay"] = float(cfg["weight_decay"])

    set_seed(cfg["seed"])

    # Load dataset
    ds = load_goemotions(cfg["dataset_config"])

    train = ds["train"].shuffle(seed=cfg["seed"]).select(range(cfg["train_subset"]))
    val = ds["validation"].shuffle(seed=cfg["seed"]).select(range(cfg["val_subset"]))

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)

    train_tok, num_labels, label_names = preprocess_goemotions(train, tok, cfg["max_length"])
    val_tok, _, _ = preprocess_goemotions(val, tok, cfg["max_length"])

    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return multilabel_f1_metrics(logits, labels)

    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        num_train_epochs=cfg["epochs"],
        weight_decay=cfg["weight_decay"],
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
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

    trainer.train()

    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)

    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    main(args.config)