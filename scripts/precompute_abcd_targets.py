#!/usr/bin/env python3
"""Precompute ABCD pseudo clinical concepts for a dataset and cache them.

Usage:
  python scripts/precompute_abcd_targets.py --data data/ham10000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.abcd_targets import compute_all_abcd, load_image, load_mask


def _resolve_ham10000_images(metadata_df, data_path):
    """Resolve HAM10000 image and mask paths.

    Returns list of dicts with image_path, mask_path, image_id.
    """
    images_part1 = data_path / "HAM10000_images_part_1"
    images_part2 = data_path / "HAM10000_images_part_2"
    masks_dir = data_path / "HAM10000_segmentations_lesion_tschandl"

    rows = []
    for _, row in metadata_df.iterrows():
        img_id = row["image_id"]
        img_path = None
        for folder in [images_part1, images_part2]:
            for ext in [".jpg", ".jpeg", ".png"]:
                p = folder / f"{img_id}{ext}"
                if p.exists():
                    img_path = p
                    break
            if img_path:
                break

        if img_path is None:
            continue

        mask_path = masks_dir / f"{img_id}_segmentation.png"
        if not mask_path.exists():
            mask_path = None

        rows.append({
            "image_id": img_id,
            "image_path": str(img_path.resolve()),
            "mask_path": str(mask_path.resolve()) if mask_path else None,
        })

    return rows


def precompute_abcd(data_dir, img_size=224):
    data_path = Path(data_dir)

    cache_dir = data_path.parent / "abcd_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{data_path.name}_abcd.csv"

    is_ham10000 = (data_path / "HAM10000_metadata.csv").exists()

    if is_ham10000:
        print(f"Detected HAM10000 dataset at {data_path}")
        metadata_df = pd.read_csv(data_path / "HAM10000_metadata.csv")
        entries = _resolve_ham10000_images(metadata_df, data_path)
        if not entries:
            print("ERROR: No HAM10000 images found. Check image directory structure.")
            return
    else:
        print(f"ERROR: HAM10000_metadata.csv not found in {data_dir}")
        return

    results = []
    print(f"Computing ABCD pseudo clinical concepts for {len(entries)} images...")

    for entry in tqdm(entries, desc="ABCD"):
        img_path = entry["image_path"]
        mask_path = entry.get("mask_path", None)

        try:
            if not Path(img_path).exists():
                raise FileNotFoundError(f"Image not found: {img_path}")

            image = load_image(img_path, img_size)

            if mask_path and Path(mask_path).exists():
                mask = load_mask(mask_path, img_size)
            else:
                mask_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            abcd = compute_all_abcd(image, mask, img_size)

            results.append({
                "image_id": entry["image_id"],
                "asymmetry": abcd["asymmetry"],
                "border": abcd["border"],
                "color": abcd["color"],
                "normalized_lesion_area": abcd["normalized_lesion_area"],
                "diameter_mm": abcd["diameter_mm"],
            })

        except Exception as e:
            print(f"  Warning: Failed on {entry['image_id']}: {e}")
            results.append({
                "image_id": entry["image_id"],
                "asymmetry": 0.5,
                "border": 0.5,
                "color": 0.5,
                "normalized_lesion_area": 0.1,
                "diameter_mm": 3.0,
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(cache_file, index=False)

    print(f"\nABCD pseudo clinical concepts saved to: {cache_file}")
    print(f"\nSummary statistics:")
    for col in ["asymmetry", "border", "color", "normalized_lesion_area"]:
        vals = result_df[col]
        print(f"  {col:<25s}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
              f"min={vals.min():.3f}, max={vals.max():.3f}")

    q3_area = result_df["normalized_lesion_area"].quantile(0.75)
    print(f"\n  Q3 normalized_lesion_area = {q3_area:.4f}")
    print(f"  (Use this as LARGE_LESION_AREA_THRESHOLD in configs/default.yaml)")

    print(f"\nDone. {len(result_df)} images processed.")


def main():
    parser = argparse.ArgumentParser(description="Precompute ABCD pseudo clinical concepts")
    parser.add_argument("--data", default="data/ham10000",
                        help="Path to dataset directory")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Image size")
    args = parser.parse_args()
    precompute_abcd(args.data, args.img_size)


if __name__ == "__main__":
    main()
