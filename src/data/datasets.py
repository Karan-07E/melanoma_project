"""PyTorch Dataset classes for HAM10000, PAD-UFES-20, and synthetic data.

Provides a unified interface. All return {image, label, image_id, abcd_targets} dicts.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit

from src.data.transforms import get_train_transforms, get_val_transforms


CLASS_NAMES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: n for i, n in enumerate(CLASS_NAMES)}

PAD_UFES20_CLASS_MAP = {
    "MEL": "mel",
    "NEV": "nv",
    "BCC": "bcc",
    "ACK": "akiec",
    "SCC": "akiec",
    "SEK": "bkl",
}

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
    img_size: int = 224,
    seed: int = 42,
    img_size: int = 224,
) -> Dataset:
    """Load the synthetic or HAM10000 dataset with stratified split.

    Works for any dataset that has a labels.csv with columns:
    [image_id, class_name, class_idx, image_path].

    Args:
        data_dir: Path to the data directory.
        mode: One of 'train', 'val', 'test'.
        train_split: Fraction for training.
        val_split: Fraction for validation.
        img_size: Image size.
        seed: Random seed for reproducibility.

    Returns:
        LesionDataset instance.
    """
    labels_path = Path(data_dir) / "labels.csv"
    df = pd.read_csv(labels_path)

    data_path = Path(data_dir)
    abcd_cache_path = data_path.parent / "abcd_cache" / f"{data_path.name}_abcd.csv"
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
        transform = get_train_transforms(img_size=img_size)
    elif mode == "val":
        split_df = pd.concat(val_dfs).sample(frac=1, random_state=seed)
        transform = get_val_transforms(img_size=img_size)
    else:
        split_df = pd.concat(test_dfs).sample(frac=1, random_state=seed)
        transform = get_val_transforms(img_size=img_size)

    return LesionDataset(split_df, transform=transform, img_size=img_size)


HAM10000_DX_MAP = {
    "mel": 0, "nv": 1, "bcc": 2, "akiec": 3,
    "bkl": 4, "df": 5, "vasc": 6,
}


def load_ham10000_dataset(
    data_dir: str,
    mode: str = "train",
    train_split: float = 0.7,
    val_split: float = 0.15,
    img_size: int = 224,
    seed: int = 42,
) -> Dataset:
    """Load the HAM10000 dataset with lesion-level stratified split.

    Splits by lesion_id to prevent data leakage — multiple images of the
    same lesion stay in the same split. Class labels come from the majority
    diagnosis for that lesion.

    Expected directory structure:
        data/ham10000/
          HAM10000_metadata.csv
          HAM10000_images_part_1/ISIC_XXXXXXX.jpg
          HAM10000_images_part_2/ISIC_XXXXXXX.jpg
          HAM10000_segmentations_lesion_tschandl/ISIC_XXXXXXX_segmentation.png

    Args:
        data_dir: Path to the HAM10000 directory.
        mode: 'train', 'val', or 'test'.
        train_split, val_split: Split fractions.
        img_size: Image size.
        seed: Random seed.

    Returns:
        LesionDataset instance.
    """
    data_path = Path(data_dir)
    metadata_path = data_path / "HAM10000_metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"HAM10000 metadata not found: {metadata_path}\n"
            "Download HAM10000_metadata.csv from Harvard Dataverse."
        )

    df = pd.read_csv(metadata_path)
    df["class_idx"] = df["dx"].map(HAM10000_DX_MAP)
    invalid = df["class_idx"].isna()
    if invalid.any():
        bad = df.loc[invalid, "dx"].unique().tolist()
        print(f"  Skipping unmapped dx codes: {bad}")
        df = df[~invalid].copy()

    images_part1 = data_path / "HAM10000_images_part_1"
    images_part2 = data_path / "HAM10000_images_part_2"
    masks_dir = data_path / "HAM10000_segmentations_lesion_tschandl"

    def _resolve_image_path(image_id):
        for folder in [images_part1, images_part2]:
            for ext in [".jpg", ".jpeg", ".png"]:
                p = folder / f"{image_id}{ext}"
                if p.exists():
                    return p
        return None

    rows = []
    skipped = 0
    for _, row in df.iterrows():
        img_id = row["image_id"]
        img_path = _resolve_image_path(img_id)
        if img_path is None:
            skipped += 1
            continue

        mask_path = masks_dir / f"{img_id}_segmentation.png"
        if not mask_path.exists():
            mask_path = None

        rows.append({
            "image_path": str(img_path.resolve()),
            "mask_path": str(mask_path.resolve()) if mask_path else "",
            "class_idx": int(row["class_idx"]),
            "class_name": row["dx"],
            "image_id": img_id,
            "lesion_id": row["lesion_id"],
        })

    if skipped > 0:
        print(f"  Skipped {skipped} images (file not found)")

    full_df = pd.DataFrame(rows)

    lesion_labels = full_df.groupby("lesion_id")["class_idx"].first().reset_index()
    lesion_labels = lesion_labels.sort_values("lesion_id").reset_index(drop=True)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_split), random_state=seed)
    train_lesion_idx, rest_idx = next(splitter.split(lesion_labels, lesion_labels["class_idx"]))

    val_test_labels = lesion_labels.iloc[rest_idx]
    val_frac_of_rest = val_split / (1 - train_split) if (1 - train_split) > 0 else 0.5
    if val_frac_of_rest >= 1.0 or val_frac_of_rest <= 0.0:
        val_frac_of_rest = 0.5

    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - val_frac_of_rest), random_state=seed)
    val_lesion_idx_rel, test_lesion_idx_rel = next(
        splitter2.split(val_test_labels, val_test_labels["class_idx"])
    )

    train_lesions = set(lesion_labels.iloc[train_lesion_idx]["lesion_id"])
    val_test_lesions = val_test_labels.copy()
    val_lesions = set(val_test_lesions.iloc[val_lesion_idx_rel]["lesion_id"])
    test_lesions = set(val_test_lesions.iloc[test_lesion_idx_rel]["lesion_id"])

    if mode == "train":
        split_df = full_df[full_df["lesion_id"].isin(train_lesions)].copy()
        transform = get_train_transforms(img_size=img_size)
    elif mode == "val":
        split_df = full_df[full_df["lesion_id"].isin(val_lesions)].copy()
        transform = get_val_transforms(img_size=img_size)
    else:
        split_df = full_df[full_df["lesion_id"].isin(test_lesions)].copy()
        transform = get_val_transforms(img_size=img_size)

    split_df = split_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    abcd_cache_path = data_path.parent / "abcd_cache" / f"{data_path.name}_abcd.csv"
    if abcd_cache_path.exists():
        abcd_df = pd.read_csv(abcd_cache_path)
        split_df = split_df.merge(abcd_df, on="image_id", how="left")

    train_lesion_count = len(train_lesions)
    val_lesion_count = len(val_lesions)
    test_lesion_count = len(test_lesions)
    total_images = len(full_df)

    print(f"  HAM10000: {total_images} images from {len(lesion_labels)} unique lesions")
    print(f"  Split by lesion_id: train={train_lesion_count} ({len(full_df[full_df['lesion_id'].isin(train_lesions)])} imgs), "
          f"val={val_lesion_count} ({len(full_df[full_df['lesion_id'].isin(val_lesions)])} imgs), "
          f"test={test_lesion_count} ({len(full_df[full_df['lesion_id'].isin(test_lesions)])} imgs)")

    return LesionDataset(split_df, transform=transform)


def load_pad_ufes20_dataset(data_dir: str, img_size: int = 224) -> Dataset:
    """Load the PAD-UFES-20 dataset for cross-domain evaluation.

    Maps PAD-UFES-20 diagnostic classes to HAM10000 classes:
        MEL→mel, NEV→nv, BCC→bcc, ACK→akiec, SCC→akiec, SEK→bkl.
    Missing HAM10000 classes (df, vasc) are excluded from metrics.

    Requires: data_dir/metadata.csv + data_dir/images/

    Args:
        data_dir: Path to the PAD-UFES-20 directory.
        img_size: Image size.

    Returns:
        LesionDataset with all samples (no split — used for evaluation only).
    """
    data_path = Path(data_dir)
    metadata_path = data_path / "metadata.csv"
    images_dir = data_path / "images"

    if not metadata_path.exists():
        raise FileNotFoundError(f"PAD-UFES-20 metadata not found: {metadata_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"PAD-UFES-20 images not found: {images_dir}")

    df = pd.read_csv(metadata_path)
    df = df.dropna(subset=["diagnostic"])

    def map_class(dx):
        mapped = PAD_UFES20_CLASS_MAP.get(str(dx).upper().strip(), None)
        if mapped is None:
            return None, None
        return CLASS_TO_IDX[mapped], mapped

    rows = []
    skipped = 0
    for _, row in df.iterrows():
        img_id = row["img_id"]
        img_path = images_dir / img_id
        if not img_path.exists():
            skipped += 1
            continue

        class_idx, class_name = map_class(row["diagnostic"])
        if class_idx is None:
            skipped += 1
            continue

        rows.append({
            "image_path": str(img_path.resolve()),
            "class_idx": class_idx,
            "class_name": class_name,
            "image_id": img_id.replace(".png", "").replace(".jpg", ""),
        })

    if skipped > 0:
        print(f"  Skipped {skipped} samples (missing image or unmapped class)")

    result_df = pd.DataFrame(rows)
    transform = get_val_transforms(img_size=img_size)
    return LesionDataset(result_df, transform=transform, img_size=img_size)


def detect_dataset_type(data_dir: str) -> str:
    """Detect the dataset type from the directory structure.

    Returns: 'synthetic', 'ham10000', 'pad_ufes20', or 'unknown'.
    """
    data_path = Path(data_dir)
    if (data_path / "labels.csv").exists():
        return "synthetic"
    if (data_path / "metadata.csv").exists() and (data_path / "images").exists():
        return "pad_ufes20"
    if (data_path / "HAM10000_metadata.csv").exists():
        return "ham10000"
    return "unknown"


def load_dataset(data_dir: str, mode: str = "test", img_size: int = 224,
                 train_split: float = 0.7, val_split: float = 0.15,
                 seed: int = 42) -> Dataset:
    """Auto-detect dataset type and load the appropriate dataset.

    Args:
        data_dir: Path to the dataset directory.
        mode: 'train', 'val', or 'test' (ignored for PAD-UFES-20).
        img_size: Image size.
        train_split, val_split: Split fractions.
        seed: Random seed.

    Returns:
        LesionDataset instance.
    """
    dtype = detect_dataset_type(data_dir)
    if dtype == "pad_ufes20":
        return load_pad_ufes20_dataset(data_dir, img_size=img_size)
    elif dtype == "ham10000":
        return load_ham10000_dataset(data_dir, mode=mode, train_split=train_split,
                                      val_split=val_split, img_size=img_size, seed=seed)
    else:
        return load_synthetic_dataset(data_dir, mode=mode, train_split=train_split,
                                       val_split=val_split, img_size=img_size, seed=seed)


def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4, sampler=None):
    """Create a PyTorch DataLoader from a dataset.

    Args:
        dataset: PyTorch Dataset.
        batch_size: Batch size.
        shuffle: Whether to shuffle (ignored if sampler is provided).
        num_workers: Number of data loading workers.
        sampler: Optional Sampler (e.g., WeightedRandomSampler).
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
