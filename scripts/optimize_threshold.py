#!/usr/bin/env python3
"""Melanoma threshold optimization and temperature scaling script.

Finds the optimal probability threshold for melanoma classification
and fits temperature scaling, using the VALIDATION SET ONLY.

Usage:
  python scripts/optimize_threshold.py --checkpoint models/best.pt --data data/ham10000

Outputs:
  - Best threshold for melanoma (maximises melanoma F1)
  - Precision/recall/F1 at each threshold
  - Temperature calibration factor
  - ECE and Brier score before/after calibration
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datasets import load_synthetic_dataset, get_dataloader
from src.models.cbm_model import CBMModel
from src.utils.temperature_scaling import TemperatureScaler


CLASS_NAMES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
MEL_IDX = 0
MALIGNANT_INDICES = (0, 2, 3)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_val_predictions(model, dataloader, device):
    """Collect logits and labels from validation set."""
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting val preds"):
            images = batch["image"].to(device)
            labels = batch["label"]

            outputs = model(images)
            all_logits.append(outputs["class_logits"].cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return logits, labels


def compute_metrics_at_threshold(probs, labels, threshold, mel_idx=MEL_IDX):
    """Compute precision, recall, F1 for melanoma at a given threshold."""
    pred_binary = (probs[:, mel_idx] >= threshold).astype(int)
    true_binary = (labels == mel_idx).astype(int)

    tp = ((pred_binary == 1) & (true_binary == 1)).sum()
    fp = ((pred_binary == 1) & (true_binary == 0)).sum()
    fn = ((pred_binary == 0) & (true_binary == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    balanced_acc = (recall + (tp + tn_est(tp, fp, fn, labels)) / len(labels)) / 2

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_acc),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def tn_est(tp, fp, fn, labels):
    tn = (labels != MEL_IDX).sum() - fp
    return max(tn, 0)


def optimize_threshold(probs, labels, min_precision=0.3):
    """Search thresholds 0.05 to 0.95, find best melanoma F1.

    Args:
        probs: (N, C) softmax probabilities.
        labels: (N,) ground truth class indices.
        min_precision: Minimum melanoma precision constraint.

    Returns:
        best_metrics, all_results list.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    results = []
    best_f1 = -1
    best_result = None

    for t in thresholds:
        metrics = compute_metrics_at_threshold(probs, labels, t)
        results.append(metrics)

        if metrics["f1"] > best_f1 and metrics["precision"] >= min_precision:
            best_f1 = metrics["f1"]
            best_result = metrics

    return best_result, results


def compute_ece(y_true, y_score, n_bins=10):
    """Compute Expected Calibration Error for binary classification."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_score, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() > 0:
            ece += (np.abs(y_score[mask].mean() - y_true[mask].mean()) * mask.sum()) / len(y_score)
    return float(ece)


def compute_brier(y_true, y_score):
    """Compute Brier score for binary classification."""
    return float(np.mean((y_score - y_true) ** 2))


def main():
    parser = argparse.ArgumentParser(description="Melanoma threshold + temperature optimization")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data", required=True, help="Path to dataset directory")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="threshold_optimization.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    print(f"Device: {device}")

    model_cfg = cfg["model"]
    model_cfg["img_size"] = cfg["data"]["img_size"]

    model = CBMModel(model_cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded from {args.checkpoint}")

    val_dataset = load_synthetic_dataset(
        args.data, mode="val",
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        seed=cfg["seed"],
    )
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Validation set: {len(val_dataset)} samples")

    logits, labels = collect_val_predictions(model, val_loader, device)
    probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()

    print(f"\n{'='*60}")
    print("THRESHOLD OPTIMIZATION (Melanoma class)")
    print(f"{'='*60}")

    best_result, all_results = optimize_threshold(probs, labels, min_precision=0.3)

    print(f"\nBest threshold (max melanoma F1, P≥0.3):")
    print(f"  Threshold: {best_result['threshold']:.3f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall:    {best_result['recall']:.4f}")
    print(f"  F1:        {best_result['f1']:.4f}")
    print(f"  Bal. Acc:  {best_result['balanced_accuracy']:.4f}")

    print(f"\n{'='*60}")
    print("TEMPERATURE SCALING")
    print(f"{'='*60}")

    y_binary = (labels == MEL_IDX).astype(np.float32)
    p_mel_uncal = probs[:, MEL_IDX]

    ece_before = compute_ece(y_binary, p_mel_uncal)
    brier_before = compute_brier(y_binary, p_mel_uncal)
    print(f"\nBefore calibration:")
    print(f"  ECE:        {ece_before:.4f}")
    print(f"  Brier:      {brier_before:.4f}")

    scaler = TemperatureScaler()
    scaler.fit(torch.from_numpy(logits), torch.from_numpy(labels))

    cal_logits = scaler.calibrate(torch.from_numpy(logits))
    cal_probs = F.softmax(cal_logits, dim=-1).numpy()
    p_mel_cal = cal_probs[:, MEL_IDX]

    ece_after = compute_ece(y_binary, p_mel_cal)
    brier_after = compute_brier(y_binary, p_mel_cal)
    print(f"\nAfter calibration (T={scaler.temperature.item():.4f}):")
    print(f"  ECE:        {ece_after:.4f}")
    print(f"  Brier:      {brier_after:.4f}")

    report = {
        "threshold_optimization": {
            "best_threshold": best_result["threshold"],
            "best_mel_precision": best_result["precision"],
            "best_mel_recall": best_result["recall"],
            "best_mel_f1": best_result["f1"],
            "results": [{k: v for k, v in r.items() if k not in ("tp", "fp", "fn")}
                         for r in all_results[::10]],
        },
        "temperature_scaling": {
            "temperature": float(scaler.temperature.item()),
            "ece_before": ece_before,
            "ece_after": ece_after,
            "brier_before": brier_before,
            "brier_after": brier_after,
        },
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
