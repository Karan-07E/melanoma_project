#!/usr/bin/env python3
"""Download helper for HAM10000 dataset.

HAM10000 requires manual download from the Harvard Dataverse:
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

This script checks for the dataset and provides exact download instructions
if files are missing.
"""

import argparse
import os
import sys
from pathlib import Path

REQUIRED_FILES = [
    "HAM10000_images_part_1",
    "HAM10000_images_part_2",
    "HAM10000_metadata.csv",
    "HAM10000_segmentations_lesion_tschandl",
]

DATAVERSE_URL = "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T"
ZIP_URLS = {
    "HAM10000_images_part_1.zip": "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/DBW86T/XXXXX",
    "HAM10000_images_part_2.zip": "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/DBW86T/XXXXX",
    "HAM10000_metadata.csv": "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/DBW86T/XXXXX",
    "HAM10000_segmentations_lesion_tschandl.zip": "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/DBW86T/XXXXX",
}


def check_dataset(data_dir):
    """Check which required files are present."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    status = {}
    for fname in REQUIRED_FILES:
        status[fname] = (data_path / fname).exists()

    return status


def main():
    parser = argparse.ArgumentParser(description="HAM10000 download helper")
    parser.add_argument("--data-dir", default="data/ham10000",
                        help="Target directory for HAM10000")
    args = parser.parse_args()

    status = check_dataset(args.data_dir)
    missing = [k for k, v in status.items() if not v]
    present = [k for k, v in status.items() if v]

    print("=" * 60)
    print("HAM10000 Dataset Download Helper")
    print("=" * 60)
    print(f"\nTarget directory: {os.path.abspath(args.data_dir)}\n")

    print("Status:")
    for fname, exists in status.items():
        indicator = "[OK]" if exists else "[MISSING]"
        print(f"  {indicator} {fname}")

    if not missing:
        print("\nAll required files found. Ready for training.")
        return

    print(f"\n{len(missing)} file(s) missing. Manual download required.")
    print("\nSteps to download HAM10000:")
    print(f"\n  1. Go to: {DATAVERSE_URL}")
    print("  2. Download these files:")
    for fname in missing:
        print(f"     - {fname}")
    print("\n  3. Extract/unzip all files into the target directory:")
    print(f"     {os.path.abspath(args.data_dir)}/")
    print("\n  4. Expected directory structure after extraction:")
    print(f"     {os.path.abspath(args.data_dir)}/")
    print("       HAM10000_images_part_1/")
    print("         ISIC_0024306.jpg")
    print("         ...")
    print("       HAM10000_images_part_2/")
    print("         ISIC_0024307.jpg")
    print("         ...")
    print("       HAM10000_metadata.csv")
    print("       HAM10000_segmentations_lesion_tschandl/")
    print("         ISIC_0024306_segmentation.png")
    print("         ...")
    print("\n  5. Re-run this script to verify the download.")

    print("\nAlternative: Kaggle CLI")
    print("  pip install kaggle")
    print("  kaggle datasets download -d kmader/skin-cancer-mnist-ham10000")
    print("  unzip skin-cancer-mnist-ham10000.zip -d data/ham10000/")

    print("\nNote: Some sources require a free account for download.")


if __name__ == "__main__":
    main()
