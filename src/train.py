#!/usr/bin/env python3
"""Training script for the CBM melanoma model.

L_total = L_diagnosis + λ_concept·L_concept + λ_constraint·L_constraint
          + λ_domain·L_domain (DANN, Strategy 3)
          + λ_coral·L_coral   (CORAL, Strategy 4)

With MixUp (Strategy 5) and aggressive color augmentation (Strategy 1).

Usage:
  python src/train.py --config configs/default.yaml --data data/synthetic --epochs 3
  python src/train.py --config configs/default.yaml --data data/ham10000 --epochs 60
  python src/train.py --data data/ham10000 --pad-data data/pad_ufes20 --epochs 60
"""

import argparse
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datasets import load_dataset, get_dataloader, load_pad_ufes20_dataset
from src.models.cbm_model import CBMModel
from src.losses.multitask_loss import MultiTaskLoss
from src.losses.alignment_loss import AlignmentLoss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg_device="auto"):
    if cfg_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg_device)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def mixup_data(x, y, alpha=0.2):
    """MixUp augmentation (Strategy 5).

    Creates convex combinations of pairs of examples:
        x_mix = λ·x₁ + (1 − λ)·x₂
        y_mix = λ·y₁ + (1 − λ)·y₂

    Args:
        x: (B, C, H, W) image batch.
        y: (B,) label batch.
        alpha: Beta distribution parameter.
               0.0 = no mixing, 1.0 = uniform mixing.
               Smaller values (0.1-0.4) work best for medical images.

    Returns:
        mixed_x, labels_a, labels_b, lam
        where mixed_y = lam·y_a + (1−lam)·y_b.
    """
    if alpha <= 0:
        return x, y, y, 1.0

    batch_size = x.size(0)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam = max(lam, 1.0 - lam)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute MixUp loss as a convex combination of two CE losses."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def train_one_epoch(
    model, source_loader, target_loader, criterion, coral_loss_fn,
    optimizer, device, epoch, num_epochs, cfg,
    writer=None, log_interval=10, scaler=None, use_amp=False,
):
    model.train()
    running = {
        "total": 0.0, "diagnosis": 0.0, "concept": 0.0,
        "constraint_total": 0.0, "constraint_rule1": 0.0,
        "constraint_rule2": 0.0, "constraint_rule3": 0.0,
        "domain": 0.0, "coral": 0.0,
    }
    correct = 0
    total_samples = 0

    lambda_domain = cfg["loss"].get("lambda_domain", 0.1)
    lambda_coral = cfg["loss"].get("lambda_coral", 0.05)
    domain_enabled = cfg["domain"].get("enabled", False)
    mixup_alpha = cfg["mixup"].get("alpha", 0.2)

    target_iter = iter(target_loader) if target_loader is not None else None

    pbar = tqdm(source_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=False)
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        abcd = batch["abcd_targets"].to(device)

        if mixup_alpha > 0:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=mixup_alpha)

        if target_loader is not None and target_iter is not None:
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)
            target_images = target_batch["image"].to(device)
        else:
            target_images = None

        optimizer.zero_grad(set_to_none=True)
        amp_context = torch.amp.autocast("cuda") if use_amp else nullcontext()
        with amp_context:
            outputs = model(images)

            if domain_enabled and target_images is not None:
                with torch.no_grad():
                    target_out = model(target_images)
                target_gvec = target_out["global_vec"]
                source_gvec = outputs["global_vec"]

                coral_loss_val = coral_loss_fn(source_gvec, target_gvec) if lambda_coral > 0 else torch.tensor(0.0, device=device)

                if model.domain_classifier is not None:
                    domain_logits_s = model.domain_forward(source_gvec, reverse=True, lambda_val=1.0)
                    domain_logits_t = model.domain_forward(target_gvec, reverse=True, lambda_val=1.0)

                    domain_labels_s = torch.zeros(source_gvec.size(0), dtype=torch.long, device=device)
                    domain_labels_t = torch.ones(target_gvec.size(0), dtype=torch.long, device=device)

                    domain_logits_all = torch.cat([domain_logits_s, domain_logits_t], dim=0)
                    domain_labels_all = torch.cat([domain_labels_s, domain_labels_t], dim=0)

                    domain_loss_val = F.cross_entropy(domain_logits_all, domain_labels_all)
                else:
                    domain_loss_val = torch.tensor(0.0, device=device)
            else:
                coral_loss_val = torch.tensor(0.0, device=device)
                domain_loss_val = torch.tensor(0.0, device=device)

            if mixup_alpha > 0:
                target_dict = {"label": labels_a, "abcd_targets": abcd}
                loss_dict_a = criterion(outputs, target_dict)
                target_dict_b = {"label": labels_b, "abcd_targets": abcd}
                loss_dict_b = criterion(outputs, target_dict_b)

                loss_diag = mixup_criterion(
                    criterion.class_criterion, outputs["class_logits"], labels_a, labels_b, lam
                )
                loss_concept = loss_dict_a["concept"]
                loss_constraint = loss_dict_a["constraint_total"]
                loss_dict = {"diagnosis": loss_diag, "concept": loss_concept,
                              "constraint_total": loss_constraint,
                              "constraint_rule1": loss_dict_a["constraint_rule1"],
                              "constraint_rule2": loss_dict_a["constraint_rule2"],
                              "constraint_rule3": loss_dict_a["constraint_rule3"]}
            else:
                target_dict = {"label": labels, "abcd_targets": abcd}
                loss_dict = criterion(outputs, target_dict)

            total_loss = (
                loss_dict["diagnosis"]
                + criterion.lambda_concept * loss_dict["concept"]
                + criterion.lambda_constraint * loss_dict["constraint_total"]
                + lambda_domain * domain_loss_val
                + lambda_coral * coral_loss_val
            )

            loss_dict["total"] = total_loss
            loss_dict["domain"] = domain_loss_val
            loss_dict["coral"] = coral_loss_val

        if scaler is not None and use_amp:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
            optimizer.step()

        for k in running:
            if k in loss_dict:
                running[k] += loss_dict[k].item() * images.size(0)

        _, pred = torch.max(outputs["class_logits"], 1)
        correct += (pred == labels).sum().item()
        total_samples += images.size(0)

        pbar.set_postfix(
            loss=f"{running['total']/total_samples:.3f}",
            acc=f"{100*correct/total_samples:.1f}%",
        )

        global_step = (epoch - 1) * len(source_loader) + batch_idx
        if writer and batch_idx % log_interval == 0:
            for k in ["total", "diagnosis", "concept", "constraint_total",
                       "constraint_rule1", "constraint_rule2", "constraint_rule3",
                       "domain", "coral"]:
                if k in loss_dict:
                    writer.add_scalar(f"loss/train_{k}", loss_dict[k].item(), global_step)

    return {k: v / max(total_samples, 1) for k, v in running.items()}, 100 * correct / max(total_samples, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device, use_amp=False):
    model.eval()
    running = {
        "total": 0.0, "diagnosis": 0.0, "concept": 0.0,
        "constraint_total": 0.0,
    }
    correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="[Val]", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        abcd = batch["abcd_targets"].to(device)

        amp_context = torch.amp.autocast("cuda") if use_amp else nullcontext()
        with amp_context:
            outputs = model(images)
            target_dict = {"label": labels, "abcd_targets": abcd}
            loss_dict = criterion(outputs, target_dict)

        for k in running:
            running[k] += loss_dict[k].item() * images.size(0)

        _, pred = torch.max(outputs["class_logits"], 1)
        correct += (pred == labels).sum().item()
        total_samples += images.size(0)
        all_preds.append(pred.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    from sklearn.metrics import f1_score
    val_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {k: v / max(total_samples, 1) for k, v in running.items()}, 100 * correct / max(total_samples, 1), val_macro_f1


def main():
    parser = argparse.ArgumentParser(description="Train CBM melanoma model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default="data/synthetic")
    parser.add_argument("--pad-data", default=None, help="PAD-UFES-20 path for domain adaptation (unlabeled)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr-backbone", type=float, default=None)
    parser.add_argument("--lr-head", type=float, default=None)
    parser.add_argument("--lambda-concept", type=float, default=None)
    parser.add_argument("--lambda-constraint", type=float, default=None)
    parser.add_argument("--lambda-domain", type=float, default=None)
    parser.add_argument("--lambda-coral", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--disable-domain", action="store_true", help="Disable DANN+CORAL domain adaptation")
    parser.add_argument("--disable-mixup", action="store_true", help="Disable MixUp")
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False

    device = get_device(args.device or cfg["device"])
    print(f"Device: {device}")

    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["data"]["batch_size"]
    img_size = args.img_size or cfg["data"]["img_size"]
    num_workers = args.num_workers if args.num_workers is not None else cfg["data"]["num_workers"]
    lambda_concept = args.lambda_concept or cfg["loss"]["lambda_concept"]
    lambda_constraint = args.lambda_constraint or cfg["loss"]["lambda_constraint"]

    model_cfg = cfg["model"]
    model_cfg["img_size"] = img_size
    model_cfg["domain"] = cfg.get("domain", {"enabled": False})

    if args.disable_domain:
        model_cfg["domain"]["enabled"] = False
    if args.disable_mixup:
        cfg["mixup"]["alpha"] = 0.0

    if args.pad_data:
        cfg["domain"]["pad_data_dir"] = args.pad_data
        if not args.disable_domain:
            model_cfg["domain"]["enabled"] = True

    model = CBMModel(model_cfg).to(device)
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    print(f"Model built. Total params: {sum(p.numel() for p in model.parameters()):,}")
    if model.domain_classifier is not None:
        print(f"  Domain classifier: enabled (DANN, Strategy 3)")

    train_dataset = load_dataset(
        args.data, mode="train",
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        img_size=img_size,
        seed=cfg["seed"],
    )
    val_dataset = load_dataset(
        args.data, mode="val",
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        img_size=img_size,
        seed=cfg["seed"],
    )

    train_labels = torch.tensor([train_dataset.df.iloc[i]["class_idx"]
                                  for i in range(len(train_dataset))])
    class_counts = torch.bincount(train_labels, minlength=model_cfg["num_classes"])
    num_classes = model_cfg["num_classes"]
    n_samples = len(train_labels)
    class_weights_tensor = n_samples / (num_classes * class_counts.float())
    class_weights_tensor[class_counts == 0] = 0.0
    class_weights_tensor = class_weights_tensor.to(device)

    sample_weights = class_weights_tensor[train_labels].cpu()
    train_sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_dataset), replacement=True,
    )

    train_loader = get_dataloader(train_dataset, batch_size=batch_size,
                                  sampler=train_sampler, num_workers=num_workers)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers)

    target_domain_enabled = model_cfg.get("domain", {}).get("enabled", False)
    target_loader = None
    if target_domain_enabled:
        pad_dir = cfg["domain"].get("pad_data_dir", cfg["data"]["pad_ufes20_dir"])
        domain_batch_size = cfg["domain"].get("domain_batch_size", batch_size)
        try:
            target_dataset = load_pad_ufes20_dataset(pad_dir, img_size=img_size)
            target_loader = get_dataloader(target_dataset, batch_size=min(domain_batch_size, len(target_dataset)),
                                            shuffle=True, num_workers=num_workers)
            print(f"Domain target (PAD-UFES-20, unlabeled): {len(target_dataset)} images")
        except Exception as e:
            print(f"WARNING: Domain target dataset not available: {e}")
            target_domain_enabled = False
            model_cfg["domain"]["enabled"] = False

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    classes = model_cfg.get("classes", ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"])
    malignant_classes = model_cfg.get("malignant_classes", ["mel", "bcc", "akiec"])
    malignant_indices = tuple(i for i, c in enumerate(classes) if c in malignant_classes)

    constraint_cfg = {
        "concept_high": cfg["constraints"]["concept_high"],
        "concept_low": cfg["constraints"]["concept_low"],
        "diameter_mm_threshold": cfg["constraints"]["diameter_mm_threshold"],
        "alpha1": cfg["constraints"]["alpha1"],
        "alpha2": cfg["constraints"]["alpha2"],
        "alpha3": cfg["constraints"]["alpha3"],
    }

    criterion = MultiTaskLoss(
        lambda_concept=lambda_concept,
        lambda_constraint=lambda_constraint,
        malignant_indices=malignant_indices,
        class_weights=class_weights_tensor,
        **constraint_cfg,
    )

    coral_loss_fn = AlignmentLoss(
        mode=cfg.get("domain", {}).get("alignment_mode", "coral"),
        weight=1.0,
    ).to(device)

    backbone_lr = args.lr_backbone or cfg["training"]["backbone_lr"]
    head_lr = args.lr_head or cfg["training"]["head_lr"]
    weight_decay = cfg["training"]["weight_decay"]

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "encoder" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ], weight_decay=weight_decay, foreach=False)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    use_amp = device.type == "cuda" and not args.disable_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_dir = Path(cfg["logging"]["log_dir"]) / f"run_{int(time.time())}"
    writer = SummaryWriter(log_dir)
    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_interval = cfg["logging"]["log_interval"]
    patience = cfg["training"]["early_stop_patience"]
    if args.disable_early_stop:
        patience = None

    best_val_loss = float("inf")
    best_val_macro_f1 = 0.0
    epochs_no_improve = 0
    start_epoch = 1

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and not args.freeze_encoder:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt and not args.disable_amp:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", ckpt.get("val_loss", best_val_loss)))
        print(f"Resumed from {args.checkpoint} at epoch {start_epoch - 1}")

    print(f"\n{'='*55}")
    print(f"Training CBM for {epochs} epochs")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  λ_concept={lambda_concept}, λ_constraint={lambda_constraint}")
    print(f"  Backbone LR={backbone_lr}, Head LR={head_lr}")
    print(f"  Weight decay={weight_decay}")
    print(f"  MixUp alpha: {cfg['mixup']['alpha']}")
    print(f"  DANN (domain adversarial): {target_domain_enabled}")
    print(f"  CORAL alignment: {target_domain_enabled}")
    print(f"  AMP: {use_amp}")
    print(f"  Early stopping: {'disabled' if patience is None else f'patience={patience} on val_macro_f1'}")
    print(f"  Class weights: {[f'{w:.3f}' for w in class_weights_tensor.tolist()]}")
    print(f"{'='*55}\n")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        train_losses, train_acc = train_one_epoch(
            model, train_loader, target_loader, criterion, coral_loss_fn,
            optimizer, device, epoch, epochs, cfg,
            writer=writer, log_interval=log_interval, scaler=scaler, use_amp=use_amp,
        )
        if args.skip_validation:
            val_losses = {"total": float("inf"), "diagnosis": float("inf"),
                           "concept": float("inf"), "constraint_total": float("inf")}
            val_acc = float("nan")
            val_macro_f1 = 0.0
        else:
            val_losses, val_acc, val_macro_f1 = validate(model, val_loader, criterion, device, use_amp=False)

        scheduler.step()
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_losses['total']:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_losses['total']:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Val Macro F1: {val_macro_f1:.4f} | "
              f"Time: {epoch_time:.1f}s")

        for key in ["val_total", "val_diagnosis", "val_concept", "val_constraint_total"]:
            loss_key = key.replace("val_", "")
            if loss_key in val_losses:
                writer.add_scalar(f"loss/{key}", val_losses[loss_key], epoch)
        writer.add_scalar("metrics/val_acc", val_acc, epoch)
        writer.add_scalar("metrics/val_macro_f1", val_macro_f1, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_loss = val_losses["total"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(), "config": cfg,
                "val_loss": val_losses["total"], "val_acc": val_acc,
                "val_macro_f1": val_macro_f1, "best_val_loss": best_val_loss,
                "class_weights": class_weights_tensor.cpu().tolist(),
            }, checkpoint_dir / "best.pt")
            print(f"  >> New best model saved (val_macro_f1={best_val_macro_f1:.4f})")
        else:
            epochs_no_improve += 1
            if patience is not None and epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(), "config": cfg,
            "val_loss": val_losses["total"], "val_acc": val_acc,
            "val_macro_f1": val_macro_f1, "best_val_loss": best_val_loss,
        }, checkpoint_dir / "latest.pt")

    writer.close()
    print(f"\nTraining complete. Best val macro F1: {best_val_macro_f1:.4f}")
    print(f"Logs: {log_dir}")
    print(f"Best model: {checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
