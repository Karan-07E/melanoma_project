#!/usr/bin/env python3
"""Download helper for PAD-UFES-20 dataset (Brazilian, diverse skin tones).

PAD-UFES-20 requires manual download from Mendeley Data:
  https://data.mendeley.com/datasets/zr7vgbcyr2/1

This script checks for the dataset and provides download instructions.
"""

import argparse
import os
from pathlib import Path

MENDELEY_URL = "https://data.mendeley.com/datasets/zr7vgbcyr2/1"


def check_dataset(data_dir):
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    images_dir = data_path / "images"
    metadata_file = data_path / "metadata.csv"

    exists = {"images": images_dir.exists(), "metadata": metadata_file.exists()}
    return exists


def main():
    parser = argparse.ArgumentParser(description="PAD-UFES-20 download helper")
    parser.add_argument("--data-dir", default="data/pad_ufes20",
                        help="Target directory for PAD-UFES-20")
    args = parser.parse_args()

    status = check_dataset(args.data_dir)

    print("=" * 60)
    print("PAD-UFES-20 Dataset Download Helper")
    print("=" * 60)
    print(f"\nTarget directory: {os.path.abspath(args.data_dir)}\n")

    print("Status:")
    for name, exists in status.items():
        indicator = "[OK]" if exists else "[MISSING]"
        print(f"  {indicator} {name}")

    if all(status.values()):
        print("\nAll required files found. Ready for cross-domain evaluation.")
        return

    print(f"\nSome files missing. Manual download required.")
    print(f"\n  1. Go to: {MENDELEY_URL}")
    print("  2. Download the dataset (may require a free Mendeley account)")
    print("  3. Extract to the target directory:")
    print(f"     {os.path.abspath(args.data_dir)}/")
    print("\n  4. Expected directory structure:")
    print(f"     {os.path.abspath(args.data_dir)}/")
    print("       images/")
    print("         PAT_001_001.png")
    print("         ...")
    print("       metadata.csv")
    print("\n  5. Re-run this script to verify.")


if __name__ == "__main__":
    main()
