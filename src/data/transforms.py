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


def _build_domain_aug(cfg, size):
    """Build domain-invariant color augmentations from config.

    These transforms vary color/lighting aggressively without
    corrupting lesion morphology — forcing the encoder to learn
    shape/texture features independent of skin tone and dermatoscope.
    """
    if not cfg.get("enabled", True):
        return []

    return [
        A.CLAHE(
            clip_limit=cfg.get("clahe_clip_limit", 2.0),
            tile_grid_size=(8, 8),
            p=cfg.get("clahe_prob", 0.3),
        ),
        A.RandomGamma(
            gamma_limit=(cfg.get("gamma_min", 60), cfg.get("gamma_max", 140)),
            p=cfg.get("random_gamma_prob", 0.3),
        ),
        A.HueSaturationValue(
            hue_shift_limit=cfg.get("hue_shift_limit", 30),
            sat_shift_limit=cfg.get("sat_shift_limit", 40),
            val_shift_limit=cfg.get("val_shift_limit", 20),
            p=0.6,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=cfg.get("brightness_limit", 0.35),
            contrast_limit=cfg.get("contrast_limit", 0.35),
            p=0.6,
        ),
        A.RGBShift(
            r_shift_limit=cfg.get("rgb_shift_limit", 20),
            g_shift_limit=cfg.get("rgb_shift_limit", 20),
            b_shift_limit=cfg.get("rgb_shift_limit", 20),
            p=cfg.get("rgb_shift_prob", 0.3),
        ),
        A.ChannelShuffle(p=cfg.get("channel_shuffle_prob", 0.2)),
        A.Solarize(
            threshold_range=(cfg.get("solarize_threshold", 0.5), 1.0),
            p=cfg.get("solarize_prob", 0.1),
        ),
    ]


def get_train_transforms(cfg=None, img_size=None):
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

    pipeline = [
        A.Resize(height=size, width=size),
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


def get_val_transforms(img_size=None):
    """Build validation/testing pipeline: only resize + normalize."""
    size = img_size or IMG_SIZE
    return A.Compose([
        A.Resize(height=size, width=size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_domain_transforms(cfg=None, img_size=None):
    """Return deterministic and strong views for unlabeled target images.

    The weak view is used for feature/domain alignment and pseudo-labels. The
    strong view adds morphology-preserving geometric and color perturbations,
    so the consistency loss does not require target labels.
    """
    weak = get_val_transforms(img_size=img_size)
    strong = get_train_transforms(cfg=cfg, img_size=img_size)
    return weak, strong


def get_tta_transforms(img_size=None):
    """Build TTA (test-time augmentation) transforms for inference averaging.

    Returns a list of transform pipelines that produce varied views
    of the same image for prediction averaging (Strategy 2).
    """
    size = img_size or IMG_SIZE
    base = A.Compose([
        A.Resize(height=size, width=size),
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
