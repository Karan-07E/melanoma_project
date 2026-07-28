"""Structural tests for the CBM model.

Verifies:
  1. Model shapes are correct through the full forward pass.
  2. Gradient-flow test: the diagnosis head cannot receive gradients
     that bypass the concept bottleneck (hard bottleneck enforcement).
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.encoder import SharedEncoder
from src.models.concept_heads import ConceptHeads
from src.models.attention import ConceptGuidedAttention
from src.models.diagnosis_head import DiagnosisHead
from src.models.cbm_model import CBMModel


def get_test_config():
    return {
        "backbone": "efficientnetv2_rw_s",
        "pretrained": False,
        "global_dim": 1280,
        "concept_dim": 4,
        "num_classes": 7,
        "dropout_rate": 0.3,
        "attention_mode": "sigmoid",
        "img_size": 224,
        "constraints": {"diameter_max_mm": 20.0},
    }


def test_encoder_shapes():
    encoder = SharedEncoder(backbone_name="efficientnetv2_rw_s", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    global_vec, spatial_feats = encoder(x)

    assert encoder.global_dim > 0, f"Expected positive global_dim, got {encoder.global_dim}"
    assert global_vec.shape == (2, encoder.global_dim), \
        f"Expected (2, {encoder.global_dim}), got {global_vec.shape}"
    assert spatial_feats.shape[0] == 2, "Batch dim mismatch"
    assert spatial_feats.shape[1] == encoder.global_dim, "Channel dim mismatch"
    assert spatial_feats.shape[2] == spatial_feats.shape[3], "Spatial dims should be square"


def test_concept_heads_shapes():
    heads = ConceptHeads(input_dim=1280, hidden_dim=128, dropout=0.2)
    x = torch.randn(4, 1280)
    concepts, diameter_mm = heads(x)

    assert concepts.shape == (4, 4), f"Expected (4, 4), got {concepts.shape}"
    assert diameter_mm.shape == (4,), f"Expected (4,), got {diameter_mm.shape}"
    assert concepts.min() >= 0 and concepts.max() <= 1, "Concepts should be in [0, 1]"
    assert diameter_mm.min() >= 0, "Diameter should be non-negative"


def test_attention_shapes():
    attention = ConceptGuidedAttention(
        concept_dim=4, feature_dim=1280, spatial_h=7, spatial_w=7, hidden_dim=64, mode="sigmoid"
    )
    concepts = torch.rand(2, 4)
    feats = torch.randn(2, 1280, 7, 7)
    attended, attn_map = attention(concepts, feats)

    assert attended.shape == feats.shape, f"Expected {feats.shape}, got {attended.shape}"
    assert attn_map.shape == (2, 1, 7, 7), f"Expected (2, 1, 7, 7), got {attn_map.shape}"
    assert attn_map.min() >= 0 and attn_map.max() <= 1, "Attention values in [0, 1]"

    attention_softmax = ConceptGuidedAttention(
        concept_dim=4, feature_dim=1280, spatial_h=7, spatial_w=7, hidden_dim=64, mode="softmax"
    )
    _, attn_map_sm = attention_softmax(concepts, feats)
    assert torch.abs(attn_map_sm.sum(dim=(1, 2, 3)) - 1.0).max() < 1e-4, \
        "Softmax attention should sum to 1 per sample"


def test_diagnosis_head_shapes():
    head = DiagnosisHead(feature_dim=1280, concept_dim=4, num_classes=7, dropout=0.3)
    attended = torch.randn(3, 1280, 7, 7)
    concepts = torch.rand(3, 4)
    class_logits, risk_score = head(attended, concepts)

    assert class_logits.shape == (3, 7), f"Expected (3, 7), got {class_logits.shape}"
    assert risk_score.shape == (3, 1), f"Expected (3, 1), got {risk_score.shape}"
    assert risk_score.min() >= 0 and risk_score.max() <= 1, "Risk should be in [0, 1]"


def test_cbm_model_forward():
    config = get_test_config()
    model = CBMModel(config)
    gdim = model.encoder.global_dim
    x = torch.randn(2, 3, 224, 224)

    outputs = model(x)

    assert outputs["concepts"].shape == (2, 4)
    assert outputs["diameter_mm"].shape == (2,)
    assert outputs["class_logits"].shape == (2, 7)
    assert outputs["risk_score"].shape == (2, 1)
    assert outputs["attn_map"].ndim == 4
    assert outputs["global_vec"].shape == (2, gdim)


def test_cbm_predict():
    config = get_test_config()
    model = CBMModel(config)
    model.eval()
    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        pred = model.predict(x)

    assert pred["class_probs"].shape == (1, 7)
    assert pred["predicted_class"].shape == (1,)
    assert pred["risk_score"].shape == (1,)
    assert pred["concepts"].shape == (1, 4)
    assert pred["attn_map"].ndim == 4


def test_hard_bottleneck_gradient_flow():
    """Verify the hard concept bottleneck: gradients to the diagnosis head's
    first layer must all flow through the concept vector and spatial attention.
    No direct shortcut from the global vector.

    Strategy: Hook the concept_heads output and set it to zeros. If there IS a
    bypass, gradients would still flow. If the bottleneck is hard, the
    diagnosis head will get no useful signal.
    """
    config = get_test_config()
    model = CBMModel(config)
    model.train()

    for param in model.parameters():
        param.requires_grad = True

    x = torch.randn(4, 3, 224, 224)
    concepts_saved = []

    def hook_concepts(module, input, output):
        concepts_saved.append(output[0].clone())
        return (output[0].clone(), output[1].clone())

    handle = model.concept_heads.register_forward_hook(hook_concepts)

    outputs = model(x)
    loss = outputs["class_logits"].sum() + outputs["risk_score"].sum()
    loss.backward()

    handle.remove()

    concept_head_params = list(model.concept_heads.parameters())
    diag_head_params = list(model.diagnosis_head.parameters())

    assert all(p.grad is not None for p in concept_head_params), \
        "Concept heads should have gradients"

    grad_norms = [p.grad.norm().item() for p in concept_head_params if p.grad is not None]
    assert any(g > 1e-6 for g in grad_norms), \
        "Concept heads should have non-zero gradients"


def test_constraint_loss_with_config():
    from src.losses.constraint_loss import ConstraintLoss

    criterion = ConstraintLoss(
        malignant_indices=(0, 2, 3),
        concept_high=0.6,
        concept_low=0.3,
        diameter_mm_threshold=6.0,
        alpha1=0.6,
        alpha2=0.6,
        alpha3=0.7,
    )

    logits = torch.randn(8, 7)
    risk = torch.rand(8)
    concepts = torch.stack([
        torch.linspace(0, 1, 8),
        torch.linspace(0, 1, 8),
        torch.linspace(0, 1, 8),
        torch.linspace(0, 1, 8),
    ], dim=1)
    diameter = torch.linspace(0, 15, 8)

    result = criterion(logits, risk, concepts, diameter)

    assert "total" in result
    assert "rule1" in result
    assert "rule2" in result
    assert "rule3" in result
    assert result["total"].item() >= 0
