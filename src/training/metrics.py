from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


def compute_classification_metrics(y_true, y_pred) -> dict:
    """
    Compute key metrics for DR severity classification.

    y_true, y_pred:
      - list/np.array of integer labels (0..4)

    Returns:
      dict with accuracy, macro_f1, qwk
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "qwk": float(qwk),
    }