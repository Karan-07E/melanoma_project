"""Evaluation metrics: accuracy, per-class P/R/F1, concept MAE, ECE, AUC.

All metrics work with PyTorch tensors or numpy arrays.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(y_true, y_pred, y_score, num_classes=7, class_names=None):
    """Compute classification metrics.

    Args:
        y_true: (N,) ground truth class indices.
        y_pred: (N,) predicted class indices.
        y_score: (N, C) softmax probabilities.
        num_classes: Number of classes.
        class_names: Optional list of class name strings.

    Returns:
        Dict with metrics.
    """
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0
    )

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(macro_p),
        "recall_macro": float(macro_r),
        "f1_macro": float(macro_f1),
        "precision_weighted": float(weighted_p),
        "recall_weighted": float(weighted_r),
        "f1_weighted": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def compute_auc(y_true, y_score, malignant_indices=(0, 2, 3)):
    """Compute ROC-AUC for malignancy risk score.

    y_true: (N,) class indices.
    y_score: (N,) risk scores or (N, C) softmax probabilities.

    If y_score is 2D, sums probabilities over malignant_indices.
    """
    if y_score.ndim == 2:
        p_malignant = y_score[:, list(malignant_indices)].sum(axis=-1)
    else:
        p_malignant = y_score

    y_binary = np.isin(y_true, list(malignant_indices)).astype(np.float32)

    if len(np.unique(y_binary)) < 2:
        return 0.5  # Cannot compute AUC with only one class

    try:
        return float(roc_auc_score(y_binary, p_malignant))
    except ValueError:
        return 0.5


def compute_ece(y_true, y_score, n_bins=10, malignant_indices=(0, 2, 3)):
    """Compute Expected Calibration Error.

    y_score: (N,) risk scores (or summed malignant probs from softmax).
    """
    if y_score.ndim == 2:
        y_score = y_score[:, list(malignant_indices)].sum(axis=-1)

    y_binary = np.isin(y_true, list(malignant_indices)).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_score, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() > 0:
            avg_conf = y_score[mask].mean()
            avg_acc = y_binary[mask].mean()
            ece += (np.abs(avg_conf - avg_acc) * mask.sum()) / len(y_score)

    return float(ece)
