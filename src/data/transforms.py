"""Data transforms for dermoscopic images using Albumentations.

Provides train and validation transforms suitable for skin lesion analysis.
Augmentations are color-safe and do not corrupt ABCD characteristics.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def get_train_transforms(cfg=None):
    """Build training augmentation pipeline with dermoscopy-safe transforms.

    Args:
        cfg: Optional config dict with augmentation parameters.
             Falls back to sensible defaults if None.

    Returns:
        albumentations.Compose pipeline.
    """
    if cfg is None:
        cfg = {}

    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.HorizontalFlip(p=cfg.get("horizontal_flip_prob", 0.5)),
        A.VerticalFlip(p=cfg.get("vertical_flip_prob", 0.3)),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=cfg.get("rotation_limit", 15),
            border_mode=0,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=cfg.get("brightness_limit", 0.2),
            contrast_limit=cfg.get("contrast_limit", 0.2),
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=cfg.get("hue_shift_limit", 10),
            sat_shift_limit=cfg.get("sat_shift_limit", 20),
            val_shift_limit=10,
            p=0.4,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms():
    """Build validation/testing pipeline: only resize + normalize."""
    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
