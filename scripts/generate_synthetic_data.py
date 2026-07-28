#!/usr/bin/env python3
"""Generate a synthetic smoke-test dataset of procedurally drawn skin lesion images.

Creates 300 images (224x224 RGB) with 7 class labels and binary segmentation masks.
Classes are shaped to produce distinct ABCD characteristics:
  mel  — irregular, asymmetric, multi-colored, large
  nv   — round, symmetric, uniform, medium
  bcc  — slightly irregular, pearly, medium
  akiec — rough, scaly, moderate irregularity
  bkl  — stuck-on, waxy, somewhat irregular
  df   — firm, well-defined, small
  vasc — red/purple, round, small

Output structure:
  data/synthetic/
    images/         224x224 JPEG images
    masks/          224x224 binary PNG masks
    labels.csv      image_id, class_name, class_idx
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pandas as pd

NUM_CLASSES = 7
CLASS_NAMES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
IMG_SIZE = 224
SAMPLES_PER_CLASS = 40  # 280 total, close to 300

CLASS_PARAMS = {
    "mel": {
        "color_base": (80, 30, 30),
        "color_range": 60,
        "n_blobs": 3,
        "asymmetry": 0.8,
        "border_jitter": 40,
        "size_range": (80, 160),
        "n_holes": 2,
    },
    "nv": {
        "color_base": (90, 60, 40),
        "color_range": 20,
        "n_blobs": 1,
        "asymmetry": 0.1,
        "border_jitter": 5,
        "size_range": (40, 80),
        "n_holes": 0,
    },
    "bcc": {
        "color_base": (100, 60, 50),
        "color_range": 40,
        "n_blobs": 2,
        "asymmetry": 0.4,
        "border_jitter": 20,
        "size_range": (50, 120),
        "n_holes": 1,
    },
    "akiec": {
        "color_base": (120, 50, 40),
        "color_range": 50,
        "n_blobs": 2,
        "asymmetry": 0.5,
        "border_jitter": 25,
        "size_range": (40, 100),
        "n_holes": 0,
    },
    "bkl": {
        "color_base": (80, 50, 30),
        "color_range": 45,
        "n_blobs": 1,
        "asymmetry": 0.3,
        "border_jitter": 15,
        "size_range": (50, 120),
        "n_holes": 1,
    },
    "df": {
        "color_base": (90, 40, 40),
        "color_range": 25,
        "n_blobs": 1,
        "asymmetry": 0.15,
        "border_jitter": 8,
        "size_range": (30, 70),
        "n_holes": 0,
    },
    "vasc": {
        "color_base": (30, 20, 60),
        "color_range": 40,
        "n_blobs": 2,
        "asymmetry": 0.1,
        "border_jitter": 5,
        "size_range": (20, 60),
        "n_holes": 0,
    },
}


def _skin_color():
    r = np.random.randint(120, 180)
    g = np.random.randint(70, 140)
    b = np.random.randint(40, 100)
    return (r, g, b)


def _generate_lesion_mask(params):
    size = IMG_SIZE
    cx, cy = size // 2 + np.random.randint(-20, 20), size // 2 + np.random.randint(-20, 20)
    sz = np.random.randint(*params["size_range"])
    asym = params["asymmetry"]

    mask = np.zeros((size, size), dtype=np.uint8)
    img_pil = Image.fromarray(mask)
    draw = ImageDraw.Draw(img_pil)

    n_blobs = params["n_blobs"]
    for i in range(n_blobs):
        bx = cx + np.random.randint(-sz // 4, sz // 4)
        by = cy + np.random.randint(-sz // 4, sz // 4)
        r = sz // (n_blobs * 1.2)
        r = max(r, 10)
        x0 = max(0, int(bx - r * (1 - asym + 0.1)))
        y0 = max(0, int(by - r * (1 + asym - 0.1)))
        x1 = min(size, int(bx + r * (1 + asym - 0.1)))
        y1 = min(size, int(by + r * (1 - asym + 0.1)))
        draw.ellipse([x0, y0, x1, y1], fill=255)

    mask = np.array(img_pil)

    jitter = params["border_jitter"]
    if jitter > 0:
        kernel = np.ones((max(1, jitter // 4), max(1, jitter // 4)), np.uint8)
        for _ in range(2):
            dilation = np.random.randint(0, jitter // 2)
            erosion = np.random.randint(0, jitter // 2)
            if dilation > 0:
                mask = _dilate_mask(mask, dilation)
            if erosion > 0:
                mask = _erode_mask(mask, erosion)

    for _ in range(params["n_holes"]):
        hx = np.random.randint(max(0, cx - sz // 3), min(size, cx + sz // 3))
        hy = np.random.randint(max(0, cy - sz // 3), min(size, cy + sz // 3))
        hr = np.random.randint(3, max(4, sz // 8))
        mask_pil = Image.fromarray(mask)
        draw = ImageDraw.Draw(mask_pil)
        draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=0)
        mask = np.array(mask_pil)

    return mask


def _dilate_mask(mask, size):
    kernel = np.ones((size, size), np.uint8)
    from scipy.ndimage import binary_dilation
    return (binary_dilation(mask, structure=kernel) * 255).astype(np.uint8)


def _erode_mask(mask, size):
    kernel = np.ones((size, size), np.uint8)
    from scipy.ndimage import binary_erosion
    return (binary_erosion(mask, structure=kernel) * 255).astype(np.uint8)


def _generate_image(mask, params, skin_bg):
    size = IMG_SIZE
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = skin_bg[0]
    img[:, :, 1] = skin_bg[1]
    img[:, :, 2] = skin_bg[2]

    cb = np.array(params["color_base"], dtype=np.float32)
    cr = params["color_range"]

    lesion_area = mask > 0
    n_pixels = lesion_area.sum()
    if n_pixels == 0:
        return img

    colors = np.clip(cb + np.random.randint(-cr, cr, 3), 0, 255).astype(np.uint8)
    img[lesion_area] = colors

    n_clusters = np.random.randint(1, 4)
    for _ in range(n_clusters):
        cluster_colors = np.clip(cb + np.random.randint(-cr // 2, cr // 2, 3), 0, 255).astype(np.uint8)
        ys, xs = np.where(lesion_area)
        if len(xs) == 0:
            continue
        cx_cl = xs[np.random.randint(0, len(xs))]
        cy_cl = ys[np.random.randint(0, len(ys))]
        r_cl = np.random.randint(5, max(6, min(size // 6, n_pixels // 20)))
        for dy in range(-r_cl, r_cl + 1):
            for dx in range(-r_cl, r_cl + 1):
                if dx * dx + dy * dy <= r_cl * r_cl:
                    px, py = cx_cl + dx, cy_cl + dy
                    if 0 <= px < size and 0 <= py < size and mask[py, px] > 0:
                        alpha = ((r_cl - np.sqrt(dx * dx + dy * dy)) / r_cl) * 0.6
                        img[py, px] = np.clip(
                            img[py, px] * (1 - alpha) + cluster_colors * alpha, 0, 255
                        ).astype(np.uint8)

    noise = np.random.normal(0, 5, (size, size, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    hair_mask = np.random.rand(size, size) < 0.02
    for _ in range(np.random.randint(0, 3)):
        hx1 = np.random.randint(0, size)
        hy1 = np.random.randint(0, size)
        hx2 = hx1 + np.random.randint(-80, 80)
        hy2 = hy1 + np.random.randint(-80, 80)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.line([hx1, hy1, hx2, hy2], fill=(40, 25, 15), width=np.random.randint(1, 3))
        img = np.array(img_pil)

    return img


def generate_synthetic_data(output_dir, num_per_class=SAMPLES_PER_CLASS):
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    np.random.seed(42)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        params = CLASS_PARAMS[cls_name]
        for i in range(num_per_class):
            img_id = f"{cls_name}_{i:04d}"
            skin_bg = _skin_color()

            mask = _generate_lesion_mask(params)
            img = _generate_image(mask, params, skin_bg)

            img_path = images_dir / f"{img_id}.jpg"
            mask_path = masks_dir / f"{img_id}.png"

            Image.fromarray(img).save(img_path, quality=95)
            Image.fromarray(mask).save(mask_path)

            labels.append({
                "image_id": img_id,
                "class_name": cls_name,
                "class_idx": cls_idx,
                "image_path": str(img_path.resolve()),
                "mask_path": str(mask_path.resolve()),
            })

    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(output_dir / "labels.csv", index=False)

    print(f"Synthetic dataset created: {output_dir}")
    print(f"  Images: {len(labels_df)}")
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  Per class: {num_per_class}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic lesion dataset")
    parser.add_argument("--output", default="data/synthetic", help="Output directory")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_CLASS,
                        help="Samples per class")
    args = parser.parse_args()
    generate_synthetic_data(args.output, args.samples)


if __name__ == "__main__":
    main()
