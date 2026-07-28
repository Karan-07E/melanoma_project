"""Multi-task loss combining diagnosis, concept, and constraint losses.

L_total = L_diagnosis + λ_concept * L_concept + λ_constraint * L_constraint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.concept_loss import ConceptLoss
from src.losses.constraint_loss import ConstraintLoss


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        lambda_concept=0.5,
        lambda_constraint=0.1,
        malignant_indices=(0, 2, 3),
        concept_high=0.6,
        concept_low=0.3,
        diameter_mm_threshold=6.0,
        alpha1=0.6,
        alpha2=0.6,
        alpha3=0.7,
        class_weights=None,
    ):
        super().__init__()
        self.lambda_concept = lambda_concept
        self.lambda_constraint = lambda_constraint

        self.class_criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.concept_criterion = ConceptLoss(weight=1.0)
        self.constraint_criterion = ConstraintLoss(
            malignant_indices=malignant_indices,
            concept_high=concept_high,
            concept_low=concept_low,
            diameter_mm_threshold=diameter_mm_threshold,
            alpha1=alpha1,
            alpha2=alpha2,
            alpha3=alpha3,
            weight=1.0,
        )

    def forward(self, outputs, targets):
        """Compute the full multi-task loss.

        Args:
            outputs: dict from CBMModel.forward() with keys:
                class_logits, risk_score, concepts, diameter_mm.
            targets: dict with keys:
                label: (B,) class indices.
                abcd_targets: (B, 5) [A, B, C, D_norm, D_mm].

        Returns:
            dict with keys:
              total: scalar overall loss.
              diagnosis: classification loss.
              concept: concept MSE loss.
              constraint_total: total constraint loss.
              constraint_rule1, constraint_rule2, constraint_rule3: per-rule losses.
        """
        label = targets["label"]
        abcd = targets["abcd_targets"]  # (B, 5): [A, B, C, D_norm, D_mm]

        class_logits = outputs["class_logits"]
        risk_score = outputs["risk_score"]
        pred_concepts = outputs["concepts"]  # (B, 4)
        pred_diameter = outputs["diameter_mm"]  # (B,)

        target_concepts_4 = abcd[:, :4]

        loss_diag = self.class_criterion(class_logits, label)
        loss_concept = self.concept_criterion(pred_concepts, target_concepts_4)

        constraint_result = self.constraint_criterion(
            class_logits, risk_score, pred_concepts, pred_diameter
        )

        total = (
            loss_diag
            + self.lambda_concept * loss_concept
            + self.lambda_constraint * constraint_result["total"]
        )

        return {
            "total": total,
            "diagnosis": loss_diag,
            "concept": loss_concept,
            "constraint_total": constraint_result["total"],
            "constraint_rule1": constraint_result["rule1"],
            "constraint_rule2": constraint_result["rule2"],
            "constraint_rule3": constraint_result["rule3"],
        }
