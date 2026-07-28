#!/usr/bin/env python3
"""Download pretrained CBM model checkpoints from Hugging Face Hub.

Usage:
  python scripts/download_models.py                                    # default repo
  python scripts/download_models.py --repo your-username/melanoma-cbm  # custom repo
"""

import argparse
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub required. Install: pip install huggingface_hub")
    exit(1)

HF_REPO = "karanm777/melanoma-cbm"
FILES = ["best.pt", "latest.pt"]


def main():
    parser = argparse.ArgumentParser(description="Download CBM model checkpoints")
    parser.add_argument("--repo", default=HF_REPO, help="Hugging Face Hub repo ID")
    parser.add_argument("--output", default="models", help="Local output directory")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        print(f"Downloading {fname} from {args.repo}...")
        path = hf_hub_download(
            repo_id=args.repo,
            filename=fname,
            local_dir=str(output),
            local_dir_use_symlinks=False,
        )
        print(f"  Saved: {path}")

    print(f"\nDone. Models in: {output.resolve()}")
    print(f"Run: python app/demo_app.py --checkpoint {output}/best.pt")


if __name__ == "__main__":
    main()
