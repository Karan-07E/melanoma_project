"""Data transforms for dermoscopic images using Albumentations.

Provides train and validation transforms suitable for skin lesion analysis.
Augmentations are color-safe for morphological features but aggressively
vary color/hue to promote domain invariance across skin tones/dermatoscopes.

Strategy 1 (Domain Generalization): Heavy domain-invariant color augmentation
  - CLAHE for local contrast normalization
  - Wide HSV shifts for skin-tone robustness
  - RandomGamma for lighting variation
  - RGBShift + ChannelShuffle for dermatoscope invariance
  - Solarize for extreme exposure robustness
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def get_train_transforms(cfg=None):
    """Build training augmentation pipeline with dermoscopy-safe transforms.

    Includes domain-invariant color augmentations for cross-domain
    generalization (Strategy 1).

    Args:
        cfg: Optional config dict with augmentation parameters.
        img_size: Target image size.
    """
    if cfg is None:
        cfg = {}
    size = img_size or IMG_SIZE

    domain_cfg = cfg.get("domain_aug", {})

    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.HorizontalFlip(p=cfg.get("horizontal_flip_prob", 0.5)),
        A.VerticalFlip(p=cfg.get("vertical_flip_prob", 0.3)),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)},
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            rotate=(-cfg.get("rotation_limit", 15), cfg.get("rotation_limit", 15)),
            p=0.5,
        ),
    ]

    pipeline.extend(_build_domain_aug(domain_cfg, size))

    pipeline.extend([
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(0.02, 0.1),
            hole_width_range=(0.02, 0.1),
            fill=0,
            p=0.2,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    return A.Compose(pipeline)


def get_val_transforms():
    """Build validation/testing pipeline: only resize + normalize."""
    size = img_size or IMG_SIZE
    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    flips_and_rotations = [
        base,
        A.Compose([A.Resize(height=size, width=size), A.HorizontalFlip(p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
        A.Compose([A.Resize(height=size, width=size), A.VerticalFlip(p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
        A.Compose([A.Resize(height=size, width=size), A.RandomRotate90(p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
        A.Compose([A.Resize(height=size, width=size),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
        A.Compose([A.Resize(height=size, width=size),
                    A.RandomGamma(gamma_limit=(90, 110), p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
        A.Compose([A.Resize(height=size, width=size), A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
        A.Compose([A.Resize(height=size, width=size),
                    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
    ]

    return flips_and_rotations
