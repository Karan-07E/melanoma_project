"""PyTorch Dataset classes for HAM10000, PAD-UFES-20, and synthetic data.

Provides a unified interface: SyntheticDataset, HAM10000Dataset, PADUFES20Dataset.
All return {image, label, image_id, abcd_targets, mask_path} dicts.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

from src.data.transforms import get_train_transforms, get_val_transforms


CLASS_NAMES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: n for i, n in enumerate(CLASS_NAMES)}

ABCD_KEYS = ["asymmetry", "border", "color", "normalized_lesion_area", "diameter_mm"]


class LesionDataset(Dataset):
    """Base dataset for lesion images with optional ABCD targets.

    Args:
        df: DataFrame with columns [image_path, class_idx, image_id].
            Optional columns: mask_path, asymmetry, border, color,
            normalized_lesion_area, diameter_mm.
        transform: Albumentations Compose or None.
        img_size: Image size (default 224).
    """

    def __init__(self, df: pd.DataFrame, transform=None, img_size: int = 224):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["image_path"]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (self.img_size, self.img_size):
            image = cv2.resize(image, (self.img_size, self.img_size))

        if self.transform:
            augmented = self.transform(image=image)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label = torch.tensor(int(row["class_idx"]), dtype=torch.long)

        out = {
            "image": image_tensor,
            "label": label,
            "image_id": str(row.get("image_id", f"img_{idx}")),
        }

        has_abcd = all(k in self.df.columns for k in ABCD_KEYS)
        if has_abcd:
            out["abcd_targets"] = torch.tensor([
                float(row[k]) for k in ABCD_KEYS
            ], dtype=torch.float32)
        else:
            out["abcd_targets"] = torch.zeros(len(ABCD_KEYS), dtype=torch.float32)

        return out


def load_synthetic_dataset(
    data_dir: str,
    mode: str = "train",
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42,
) -> Dataset:
    """Load the synthetic dataset with stratified split.

    Args:
        data_dir: Path to the synthetic data directory.
        mode: One of 'train', 'val', 'test'.
        train_split: Fraction for training.
        val_split: Fraction for validation.
        seed: Random seed for reproducibility.

    Returns:
        LesionDataset instance.
    """
    labels_path = Path(data_dir) / "labels.csv"
    df = pd.read_csv(labels_path)

    abcd_cache_path = Path(data_dir).parent / "abcd_cache" / "synthetic_abcd.csv"
    if abcd_cache_path.exists():
        abcd_df = pd.read_csv(abcd_cache_path)
        df = df.merge(abcd_df, on="image_id", how="left")

    np.random.seed(seed)
    classes = df["class_idx"].unique()

    train_dfs, val_dfs, test_dfs = [], [], []
    for cls in classes:
        cls_df = df[df["class_idx"] == cls].sample(frac=1, random_state=seed)
        n = len(cls_df)
        n_train = int(n * train_split)
        n_val = int(n * val_split)
        train_dfs.append(cls_df.iloc[:n_train])
        val_dfs.append(cls_df.iloc[n_train:n_train + n_val])
        test_dfs.append(cls_df.iloc[n_train + n_val:])

    if mode == "train":
        split_df = pd.concat(train_dfs).sample(frac=1, random_state=seed)
        transform = get_train_transforms()
    elif mode == "val":
        split_df = pd.concat(val_dfs).sample(frac=1, random_state=seed)
        transform = get_val_transforms()
    else:
        split_df = pd.concat(test_dfs).sample(frac=1, random_state=seed)
        transform = get_val_transforms()

    return LesionDataset(split_df, transform=transform)


def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """Create a PyTorch DataLoader from a dataset."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
