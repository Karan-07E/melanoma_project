#!/usr/bin/env python3
"""Precompute ABCD pseudo clinical concepts for a dataset and cache them.

Usage:
  python scripts/precompute_abcd_targets.py --data data/synthetic
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


def precompute_abcd(data_dir, img_size=224):
    data_path = Path(data_dir)
    labels_csv = data_path / "labels.csv"

    if not labels_csv.exists():
        print(f"ERROR: labels.csv not found in {data_dir}")
        print("Run scripts/generate_synthetic_data.py first, or download real data.")
        return

    df = pd.read_csv(labels_csv)

    cache_dir = data_path.parent / "abcd_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"{data_path.name}_abcd.csv"

    results = []
    print(f"Computing ABCD pseudo clinical concepts for {len(df)} images...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="ABCD"):
        img_path = row["image_path"]
        mask_path = row.get("mask_path", None)

        try:
            image = load_image(img_path, img_size)

            if mask_path and Path(mask_path).exists():
                mask = load_mask(mask_path, img_size)
            else:
                mask_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            abcd = compute_all_abcd(image, mask, img_size)

            results.append({
                "image_id": row["image_id"],
                "asymmetry": abcd["asymmetry"],
                "border": abcd["border"],
                "color": abcd["color"],
                "normalized_lesion_area": abcd["normalized_lesion_area"],
                "diameter_mm": abcd["diameter_mm"],
            })

        except Exception as e:
            print(f"  Warning: Failed on {row['image_id']}: {e}")
            results.append({
                "image_id": row["image_id"],
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
    parser.add_argument("--data", default="data/synthetic",
                        help="Path to dataset directory (must contain labels.csv)")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Image size")
    args = parser.parse_args()
    precompute_abcd(args.data, args.img_size)


if __name__ == "__main__":
    main()
