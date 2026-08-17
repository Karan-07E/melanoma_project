#!/usr/bin/env python3
"""Evaluation script for the CBM melanoma model.

Produces a JSON report with:
  - Accuracy (in-domain + cross-domain)
  - Per-class precision, recall, F1
  - Concept prediction MAE (per concept + overall)
  - ECE and AUC for risk score
  - Constraint violation rates

Supports Test-Time Augmentation (Strategy 2):
  Averages predictions over multiple augmented views (flips, rotations,
  brightness shifts) for more robust evaluation.

Usage:
  python src/evaluate.py --checkpoint models/best.pt --data data/ham10000 --tta
  python src/evaluate.py --checkpoint models/best.pt --data data/pad_ufes20 --tta
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

from src.data.datasets import load_dataset, get_dataloader
from src.models.cbm_model import CBMModel
from src.utils.metrics import (
    compute_metrics,
    compute_auc,
    compute_ece,
)
from src.utils.viz import export_prediction_visualization


CLASS_NAMES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tta_predict(model, images, device, tta_transforms=None, num_aug=4):
    """Test-Time Augmentation (TTA): average predictions over augmented views.

    Strategy 2 from the domain generalization roadmap.

    Args:
        model: CBMModel instance.
        images: (B, 3, H, W) normalized input tensor.
        device: torch device.
        tta_transforms: List of Albumentations transforms (or None for simple flips).
        num_aug: Number of augmentation views (default 4).

    Returns:
        dict with averaged class_logits, risk_score, concepts.
    """
    model.eval()
    B = images.size(0)

    all_logits = []
    all_risk = []
    all_concepts = []
    all_diameter = []

    tta_views = []
    for i in range(min(num_aug, 8)):
        if i == 0:
            tta_views.append(images)
        elif i == 1:
            tta_views.append(torch.flip(images, dims=[-1]))
        elif i == 2:
            tta_views.append(torch.flip(images, dims=[-2]))
        elif i == 3:
            tta_views.append(torch.rot90(images, k=1, dims=[-2, -1]))
        elif i == 4:
            tta_views.append(torch.flip(torch.rot90(images, k=1, dims=[-2, -1]), dims=[-1]))
        elif i == 5:
            tta_views.append(images + torch.randn_like(images) * 0.01)
        elif i == 6:
            tta_views.append(torch.flip(images, dims=[-1, -2]))
        elif i == 7:
            tta_views.append(torch.rot90(images, k=2, dims=[-2, -1]))

    with torch.no_grad():
        for view in tta_views:
            out = model(view.to(device))
            all_logits.append(out["class_logits"].cpu())
            all_risk.append(out["risk_score"].cpu())
            all_concepts.append(out["concepts"].cpu())
            all_diameter.append(out["diameter_mm"].cpu())

    avg_logits = torch.stack(all_logits).mean(dim=0)
    avg_risk = torch.stack(all_risk).mean(dim=0)
    avg_concepts = torch.stack(all_concepts).mean(dim=0)
    avg_diameter = torch.stack(all_diameter).mean(dim=0)

    return avg_logits, avg_risk, avg_concepts, avg_diameter


def evaluate(model, dataloader, cfg, device, use_tta=False, tta_aug=4):
    model.eval()

    all_labels = []
    all_class_probs = []
    all_risk_scores = []
    all_concepts = []
    all_targets = []
    all_logits_list = []

    violated_1_count = 0
    violated_2_count = 0
    violated_3_count = 0
    total_samples = 0

    constraint_cfg = cfg.get("constraints", {})
    concept_high_threshold = constraint_cfg.get("concept_high", 0.6)
    concept_low_threshold = constraint_cfg.get("concept_low", 0.3)
    diameter_mm_threshold = constraint_cfg.get("diameter_mm_threshold", 6.0)

    malignant_indices = cfg.get("malignant_indices", (0, 2, 3))

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["label"]

            if use_tta:
                avg_logits, avg_risk, avg_concepts, avg_diameter = tta_predict(
                    model, images, device, num_aug=tta_aug,
                )
                outputs = {
                    "class_logits": avg_logits.to(device),
                    "risk_score": avg_risk.to(device),
                    "concepts": avg_concepts.to(device),
                    "diameter_mm": avg_diameter.to(device),
                }
                class_probs = F.softmax(avg_logits.to(device), dim=-1)
            else:
                outputs = model(images)
                class_probs = F.softmax(outputs["class_logits"], dim=-1)

            all_labels.append(labels.numpy())
            all_class_probs.append(class_probs.cpu().numpy())
            all_risk_scores.append(outputs["risk_score"].cpu().numpy().flatten())
            all_concepts.append(outputs["concepts"].cpu().numpy())
            all_targets.append(batch["abcd_targets"].cpu().numpy())

            class_logits = outputs["class_logits"]
            risk_score = outputs["risk_score"].squeeze(-1)
            concepts = outputs["concepts"]
            diameter_mm = outputs["diameter_mm"]

            A, B, C = concepts[:, 0], concepts[:, 1], concepts[:, 2]
            p_mal_from_probs = class_probs[:, list(malignant_indices)].sum(dim=-1)

            all_high = (A > concept_high_threshold) & (B > concept_high_threshold) & (C > concept_high_threshold)
            all_low = (A < concept_low_threshold) & (B < concept_low_threshold) & (C < concept_low_threshold)
            large_diameter = diameter_mm >= diameter_mm_threshold

            violated_1_count += (all_high & (p_mal_from_probs < 0.5)).sum().item()
            violated_2_count += (large_diameter & (p_mal_from_probs < 0.5)).sum().item()
            violated_3_count += (all_low & (p_mal_from_probs > 0.7)).sum().item()
            total_samples += images.size(0)

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_class_probs)
    y_pred = y_score.argmax(axis=-1)
    risk_scores = np.concatenate(all_risk_scores)
    pred_concepts = np.concatenate(all_concepts)
    target_concepts = np.concatenate(all_targets)

    metrics = compute_metrics(y_true, y_pred, y_score, num_classes=7, class_names=CLASS_NAMES)

    auc = compute_auc(y_true, risk_scores, malignant_indices=malignant_indices)
    ece = compute_ece(y_true, risk_scores, n_bins=cfg.get("ece_n_bins", 10), malignant_indices=malignant_indices)

    names = ["asymmetry", "border", "color", "normalized_lesion_area"]
    concept_mae = {}
    for i, name in enumerate(names):
        concept_mae[name] = float(np.abs(pred_concepts[:, i] - target_concepts[:, i]).mean())
    concept_mae["overall"] = float(np.mean(list(concept_mae.values())))

    constraint_rates = {
        "rule1_rate": violated_1_count / max(total_samples, 1),
        "rule2_rate": violated_2_count / max(total_samples, 1),
        "rule3_rate": violated_3_count / max(total_samples, 1),
    }

    report = {
        "classification": metrics,
        "risk": {"auc": auc, "ece": ece},
        "concept_mae": concept_mae,
        "constraint_violations": constraint_rates,
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate CBM melanoma model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data", default="data/ham10000", help="Path to dataset directory")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--img-size", type=int, default=None,
                        help="Override image size from the checkpoint/config")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (Strategy 2)")
    parser.add_argument("--tta-aug", type=int, default=None,
                        help="Number of TTA views (defaults to config)")
    parser.add_argument("--export-viz", action="store_true", help="Export prediction visualizations")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    print(f"Device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_cfg = checkpoint.get("config", {})
    for section in ("model", "data", "augmentation", "constraints", "evaluation", "domain"):
        saved_section = checkpoint_cfg.get(section)
        if isinstance(saved_section, dict):
            cfg.setdefault(section, {}).update(saved_section)
    if args.img_size is not None:
        cfg["data"]["img_size"] = args.img_size

    model_cfg = cfg["model"]
    model_cfg["img_size"] = cfg["data"]["img_size"]

    model = CBMModel(model_cfg).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded from {args.checkpoint}")

    test_dataset = load_dataset(
        args.data, mode="test",
        img_size=cfg["data"]["img_size"],
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        seed=cfg["seed"],
        augmentation_cfg=cfg.get("augmentation", {}),
    )
    test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Test set: {len(test_dataset)} samples")

    eval_cfg = {
        "malignant_indices": tuple(
            i for i, c in enumerate(cfg["model"].get("classes", CLASS_NAMES))
            if c in cfg["model"].get("malignant_classes", ["mel", "bcc", "akiec"])
        ),
        "ece_n_bins": cfg["evaluation"].get("ece_n_bins", 10),
        "constraints": cfg.get("constraints", {}),
    }

    tta_cfg = cfg["evaluation"].get("tta", {})
    use_tta = args.tta or tta_cfg.get("enabled", False)
    tta_aug = args.tta_aug if args.tta_aug is not None else tta_cfg.get("num_augmentations", 4)

    report = evaluate(model, test_loader, eval_cfg, device, use_tta=use_tta, tta_aug=tta_aug)

    print(f"\n{'='*55}")
    print("EVALUATION REPORT")
    if use_tta:
        print(f"  TTA: enabled ({tta_aug} views)")
    print(f"{'='*55}")
    print(f"\nAccuracy: {report['classification']['accuracy']:.4f}")
    print(f"F1 (Macro): {report['classification']['f1_macro']:.4f}")
    print(f"F1 (Weighted): {report['classification']['f1_weighted']:.4f}")

    print(f"\nPer-class metrics:")
    for cls_name, m in report["classification"]["per_class"].items():
        print(f"  {cls_name:<10s}: P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} (n={m['support']})")

    print(f"\nConcept MAE:")
    for name, mae in report["concept_mae"].items():
        print(f"  {name:<25s}: {mae:.4f}")

    print(f"\nRisk Score:")
    print(f"  AUC: {report['risk']['auc']:.4f}")
    print(f"  ECE: {report['risk']['ece']:.4f}")

    print(f"\nConstraint Violation Rates:")
    for rule, rate in report["constraint_violations"].items():
        print(f"  {rule}: {rate:.4f}")

    output_path = args.output or "evaluation_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {output_path}")

    if args.export_viz:
        viz_dir = Path(cfg["evaluation"].get("viz_output_dir", "runs/viz_output"))
        viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nExporting visualizations to {viz_dir}...")
        for i, sample in enumerate(test_dataset):
            if i >= 5:
                break
            try:
                img_path = test_dataset.df.iloc[i]["image_path"]
            except (KeyError, AttributeError):
                continue
            image_tensor = sample["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model.predict(image_tensor)
            save_path = viz_dir / f"prediction_{i:04d}.png"
            export_prediction_visualization(img_path, pred, str(save_path))
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
