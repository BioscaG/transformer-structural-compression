import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _load_label_names(model, model_path: Path):
    """
    Best-effort label names:
    1) If model.config.id2label looks meaningful, use it.
    2) Else, if model_path/label_names.json exists, use it.
    3) Else fallback to LABEL_{i}.
    """
    id2label = getattr(model.config, "id2label", None)
    if isinstance(id2label, dict) and len(id2label) > 0:
        # HF sometimes stores keys as strings
        labels = [id2label.get(str(i), id2label.get(i, f"LABEL_{i}")) for i in range(len(id2label))]
        # If they look like LABEL_0, LABEL_1..., they're not meaningful
        if not all(str(l).startswith("LABEL_") for l in labels):
            return labels

    label_file = model_path / "label_names.json"
    if label_file.exists():
        return json.loads(label_file.read_text())

    # fallback
    num_labels = getattr(model.config, "num_labels", None) or 0
    return [f"LABEL_{i}" for i in range(num_labels)]


@torch.no_grad()
def predict(
    model,
    tokenizer,
    text: str,
    threshold: float,
    topk: int,
    device: torch.device,
    label_names: list[str],
):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).squeeze(0).detach().cpu()

    # top-k
    topk = min(topk, probs.numel())
    top_vals, top_idx = torch.topk(probs, k=topk)
    top = [(label_names[int(i)], float(v)) for v, i in zip(top_vals, top_idx)]

    # thresholded labels
    active_idx = (probs >= threshold).nonzero(as_tuple=False).squeeze(-1).tolist()
    active = sorted([(label_names[i], float(probs[i])) for i in active_idx], key=lambda x: x[1], reverse=True)

    return top, active


def main():
    ap = argparse.ArgumentParser(description="Quick inference for multi-label emotion models.")
    ap.add_argument("--model_path", required=True, help="Path to a saved model folder (e.g., outputs/models/baseline)")
    ap.add_argument("--text", default=None, help="Text to classify. If omitted, interactive mode starts.")
    ap.add_argument("--threshold", type=float, default=0.2, help="Sigmoid threshold for active labels.")
    ap.add_argument("--topk", type=int, default=8, help="Show top-k labels by probability.")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA/MPS available.")
    args = ap.parse_args()

    model_path = Path(args.model_path)

    device = torch.device("cpu")
    if not args.cpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    label_names = _load_label_names(model, model_path)

    def run_once(t: str):
        top, active = predict(
            model=model,
            tokenizer=tokenizer,
            text=t,
            threshold=args.threshold,
            topk=args.topk,
            device=device,
            label_names=label_names,
        )

        print(f"\nDevice: {device}")
        print(f"Threshold: {args.threshold}")
        print("\nTop-k:")
        for name, p in top:
            print(f"  - {name:20s}  {p:.3f}")

        print("\nActive (>= threshold):")
        if not active:
            print("  (none)")
        else:
            for name, p in active:
                print(f"  - {name:20s}  {p:.3f}")

    if args.text is not None:
        run_once(args.text)
        return

    # interactive
    print("Interactive mode. Type text and press Enter. Empty line to quit.\n")
    while True:
        t = input("> ").strip()
        if not t:
            break
        run_once(t)


if __name__ == "__main__":
    main()