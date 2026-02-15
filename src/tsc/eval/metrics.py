import numpy as np
from sklearn.metrics import f1_score

def multilabel_f1_metrics(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.1):
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    avg_pred = preds.sum(axis=1).mean()

    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "avg_pred_labels": float(avg_pred),
    }