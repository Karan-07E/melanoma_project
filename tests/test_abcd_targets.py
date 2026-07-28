"""Tests for ABCD pseudo clinical concept auto-labeling."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.abcd_targets import (
    compute_asymmetry,
    compute_border_irregularity,
    compute_color_variation,
    compute_normalized_area,
    compute_all_abcd,
)


def test_asymmetry_perfect_circle():
    mask = np.zeros((100, 100), dtype=np.uint8)
    rr, cc = np.ogrid[:100, :100]
    mask[(rr - 50) ** 2 + (cc - 50) ** 2 <= 40**2] = 255
    score = compute_asymmetry(mask)
    assert 0.0 <= score <= 1.0, f"Asymmetry out of range: {score}"
    assert score < 0.3, f"Perfect circle should be symmetric, got {score}"


def test_asymmetry_irregular_shape():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 10:50] = 255
    mask[10:90, 60:90] = 255
    score = compute_asymmetry(mask)
    assert 0.0 <= score <= 1.0
    assert score > 0.1, f"Irregular shape should show some asymmetry, got {score}"


def test_border_perfect_circle():
    mask = np.zeros((200, 200), dtype=np.uint8)
    rr, cc = np.ogrid[:200, :200]
    mask[(rr - 100) ** 2 + (cc - 100) ** 2 <= 80**2] = 255
    score = compute_border_irregularity(mask)
    assert 0.0 <= score <= 1.0
    assert score < 0.3, f"Circle should have low border irregularity, got {score}"


def test_border_irregular_shape():
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:150, 30:60] = 255
    mask[30:110, 150:170] = 255
    mask[120:170, 70:130] = 255
    score = compute_border_irregularity(mask)
    assert 0.0 <= score <= 1.0


def test_color_variation_uniform():
    image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    mask = np.ones((100, 100), dtype=np.uint8) * 255
    score = compute_color_variation(image, mask)
    assert score < 0.1, f"Uniform color should have low variation, got {score}"


def test_color_variation_diverse():
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.uint8) * 255
    score = compute_color_variation(image, mask)
    assert 0.0 <= score <= 1.0


def test_normalized_area_full():
    mask = np.ones((224, 224), dtype=np.uint8) * 255
    area_norm, dia_mm = compute_normalized_area(mask, 224)
    assert abs(area_norm - 1.0) < 0.01, f"Full mask should have area 1.0, got {area_norm}"
    assert dia_mm > 0, "Diameter should be positive"


def test_normalized_area_empty():
    mask = np.zeros((224, 224), dtype=np.uint8)
    area_norm, dia_mm = compute_normalized_area(mask, 224)
    assert area_norm == 0.0, f"Empty mask should have area 0, got {area_norm}"
    assert dia_mm == 0.0, f"Empty mask should have diameter 0, got {dia_mm}"


def test_compute_all_abcd():
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255
    result = compute_all_abcd(image, mask, 100)

    for key in ["asymmetry", "border", "color", "normalized_lesion_area", "diameter_mm"]:
        assert key in result, f"Missing key: {key}"
        assert 0.0 <= result[key] <= (100.0 if key == "diameter_mm" else 1.0), \
            f"{key} out of range: {result[key]}"
