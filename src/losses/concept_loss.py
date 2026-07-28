"""Concept loss: MSE between predicted and pseudo-target ABCD scores."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptLoss(nn.Module):
    """Mean Squared Error over the 4 ABCD concept scores.

    Args:
        pred_concepts: (B, 4) predicted [A, B, C, D_normalized].
        target_concepts: (B, 4) pseudo-target [A, B, C, D_normalized].

    Returns:
        Scalar loss value.
    """

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred_concepts, target_concepts):
        loss = F.mse_loss(pred_concepts, target_concepts)
        return self.weight * loss


def concept_mae(pred_concepts, target_concepts):
    """Compute per-concept Mean Absolute Error.

    Returns:
        Dict mapping concept name → MAE and an overall average.
    """
    abs_diff = torch.abs(pred_concepts - target_concepts)
    names = ["asymmetry", "border", "color", "normalized_area"]
    mae = {}
    for i, name in enumerate(names):
        mae[name] = abs_diff[:, i].mean().item()
    mae["overall"] = abs_diff.mean().item()
    return mae
