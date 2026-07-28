#!/usr/bin/env python3
"""Training script for the CBM melanoma model.

L_total = L_diagnosis + λ_concept * L_concept + λ_constraint * L_constraint

Usage:
  python src/train.py --config configs/default.yaml --data data/synthetic --epochs 3
  python src/train.py --config configs/default.yaml --data data/ham10000 --epochs 60
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datasets import load_synthetic_dataset, get_dataloader
from src.models.cbm_model import CBMModel
from src.losses.multitask_loss import MultiTaskLoss


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


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs, writer=None, log_interval=10):
    model.train()
    running = {
        "total": 0.0, "diagnosis": 0.0, "concept": 0.0,
        "constraint_total": 0.0, "constraint_rule1": 0.0,
        "constraint_rule2": 0.0, "constraint_rule3": 0.0,
    }
    correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=False)
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        abcd = batch["abcd_targets"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)

        target_dict = {"label": labels, "abcd_targets": abcd}
        loss_dict = criterion(outputs, target_dict)

        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k in running:
            running[k] += loss_dict[k].item() * images.size(0)

        _, pred = torch.max(outputs["class_logits"], 1)
        correct += (pred == labels).sum().item()
        total_samples += images.size(0)

        pbar.set_postfix(
            loss=f"{running['total']/total_samples:.3f}",
            acc=f"{100*correct/total_samples:.1f}%",
        )

        global_step = (epoch - 1) * len(dataloader) + batch_idx
        if writer and batch_idx % log_interval == 0:
            writer.add_scalar("loss/train_total", loss_dict["total"].item(), global_step)
            writer.add_scalar("loss/diagnosis", loss_dict["diagnosis"].item(), global_step)
            writer.add_scalar("loss/concept", loss_dict["concept"].item(), global_step)
            writer.add_scalar("loss/constraint_total", loss_dict["constraint_total"].item(), global_step)
            writer.add_scalar("loss/constraint_rule1", loss_dict["constraint_rule1"].item(), global_step)
            writer.add_scalar("loss/constraint_rule2", loss_dict["constraint_rule2"].item(), global_step)
            writer.add_scalar("loss/constraint_rule3", loss_dict["constraint_rule3"].item(), global_step)

    return {k: v / total_samples for k, v in running.items()}, 100 * correct / total_samples


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running = {
        "total": 0.0, "diagnosis": 0.0, "concept": 0.0,
        "constraint_total": 0.0,
    }
    correct = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="[Val]", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        abcd = batch["abcd_targets"].to(device)

        outputs = model(images)
        target_dict = {"label": labels, "abcd_targets": abcd}
        loss_dict = criterion(outputs, target_dict)

        for k in running:
            running[k] += loss_dict[k].item() * images.size(0)

        _, pred = torch.max(outputs["class_logits"], 1)
        correct += (pred == labels).sum().item()
        total_samples += images.size(0)

    return {k: v / total_samples for k, v in running.items()}, 100 * correct / total_samples


def main():
    parser = argparse.ArgumentParser(description="Train CBM melanoma model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default="data/synthetic")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr-backbone", type=float, default=None)
    parser.add_argument("--lr-head", type=float, default=None)
    parser.add_argument("--lambda-concept", type=float, default=None)
    parser.add_argument("--lambda-constraint", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = get_device(args.device or cfg["device"])
    print(f"Device: {device}")

    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["data"]["batch_size"]
    lambda_concept = args.lambda_concept or cfg["loss"]["lambda_concept"]
    lambda_constraint = args.lambda_constraint or cfg["loss"]["lambda_constraint"]

    model_cfg = cfg["model"]
    model_cfg["img_size"] = cfg["data"]["img_size"]

    model = CBMModel(model_cfg).to(device)
    print(f"Model built. Total params: {sum(p.numel() for p in model.parameters()):,}")

    train_dataset = load_synthetic_dataset(
        args.data, mode="train",
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        seed=cfg["seed"],
    )
    val_dataset = load_synthetic_dataset(
        args.data, mode="val",
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        seed=cfg["seed"],
    )
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=cfg["data"]["num_workers"])
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=cfg["data"]["num_workers"])
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
        **constraint_cfg,
    )

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
    ], weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    log_dir = Path(cfg["logging"]["log_dir"]) / f"run_{int(time.time())}"
    writer = SummaryWriter(log_dir)
    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_interval = cfg["logging"]["log_interval"]
    patience = cfg["training"]["early_stop_patience"]

    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"\n{'='*55}")
    print(f"Training CBM for {epochs} epochs")
    print(f"  λ_concept={lambda_concept}, λ_constraint={lambda_constraint}")
    print(f"  Backbone LR={backbone_lr}, Head LR={head_lr}")
    print(f"  Malignant classes: {malignant_classes}")
    print(f"{'='*55}\n")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_losses, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs,
            writer=writer, log_interval=log_interval,
        )
        val_losses, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_losses['total']:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_losses['total']:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")

        writer.add_scalar("loss/val_total", val_losses["total"], epoch)
        writer.add_scalar("loss/val_diagnosis", val_losses["diagnosis"], epoch)
        writer.add_scalar("loss/val_concept", val_losses["concept"], epoch)
        writer.add_scalar("loss/val_constraint_total", val_losses["constraint_total"], epoch)
        writer.add_scalar("metrics/val_acc", val_acc, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "val_loss": val_losses["total"],
                "val_acc": val_acc,
            }, checkpoint_dir / "best.pt")
            print(f"  >> New best model saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": cfg,
        }, checkpoint_dir / "latest.pt")

    writer.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Logs: {log_dir}")
    print(f"Best model: {checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
