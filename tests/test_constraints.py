"""Tests for clinical constraint loss."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.losses.constraint_loss import ConstraintLoss
from src.losses.multitask_loss import MultiTaskLoss


def test_constraint_all_low_concepts():
    criterion = ConstraintLoss(
        malignant_indices=(0, 2, 3),
        concept_high=0.6,
        concept_low=0.3,
    )

    concepts = torch.tensor([[0.1, 0.1, 0.1, 0.05]])
    diameter = torch.tensor([2.0])
    logits = torch.zeros(1, 7)
    logits[:, (0, 2, 3)] = 5.0
    risk = torch.tensor([0.9])

    result = criterion(logits, risk, concepts, diameter)

    assert result["rule3"].item() > 0, "All-low concepts + high risk should penalize (rule 3)"
    assert result["rule1"].item() == 0, "No high concepts, rule 1 should be 0"
    assert result["rule2"].item() == 0, "Small diameter, rule 2 should be 0"


def test_constraint_all_high_concepts():
    criterion = ConstraintLoss(
        malignant_indices=(0, 2, 3),
        concept_high=0.6,
    )

    concepts = torch.tensor([[0.9, 0.85, 0.95, 0.8]])
    diameter = torch.tensor([2.0])
    logits = torch.zeros(1, 7)
    logits[:, 1] = 5.0
    risk = torch.tensor([0.2])

    result = criterion(logits, risk, concepts, diameter)

    assert result["rule1"].item() > 0, "All-high concepts + low P(malignant) should penalize (rule 1)"


def test_constraint_large_diameter():
    criterion = ConstraintLoss(
        malignant_indices=(0, 2, 3),
        diameter_mm_threshold=6.0,
    )

    concepts = torch.tensor([[0.4, 0.4, 0.4, 0.5]])
    diameter = torch.tensor([10.0])
    logits = torch.zeros(1, 7)
    logits[:, 1] = 5.0
    risk = torch.tensor([0.2])

    result = criterion(logits, risk, concepts, diameter)

    assert result["rule2"].item() > 0, "Large diameter + low P(malignant) should penalize (rule 2)"


def test_constraint_no_violation():
    criterion = ConstraintLoss(
        malignant_indices=(0, 2, 3),
        alpha1=0.6,
    )

    concepts = torch.tensor([[0.9, 0.85, 0.95, 0.8]])
    diameter = torch.tensor([2.0])
    logits = torch.zeros(1, 7)
    logits[:, (0, 2, 3)] = 5.0
    risk = torch.tensor([0.9])

    result = criterion(logits, risk, concepts, diameter)
    assert result["rule1"].item() == 0, "High P(malignant) should avoid rule 1 penalty"


def test_multitask_loss():
    criterion = MultiTaskLoss(
        lambda_concept=0.5,
        lambda_constraint=0.1,
        malignant_indices=(0, 2, 3),
    )

    outputs = {
        "class_logits": torch.randn(4, 7),
        "risk_score": torch.rand(4, 1),
        "concepts": torch.rand(4, 4),
        "diameter_mm": torch.rand(4) * 10,
    }
    targets = {
        "label": torch.randint(0, 7, (4,)),
        "abcd_targets": torch.rand(4, 5),
    }

    loss_dict = criterion(outputs, targets)

    assert "total" in loss_dict
    assert "diagnosis" in loss_dict
    assert "concept" in loss_dict
    assert "constraint_total" in loss_dict
    assert loss_dict["total"].item() > 0
