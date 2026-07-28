"""Clinical constraint loss: soft penalties for clinically inconsistent predictions.

Three rules (all use max(0, ...), penalize violations only):

1. High-risk ABCD → raise P_malignant
   if A>τ and B>τ and C>τ: penalty += max(0, α1 - P_malignant)

2. Large diameter   → raise P_malignant
   if D_mm >= 6: penalty += max(0, α2 - P_malignant)

3. Global consistency
   if all(A,B,C,D low) and P_malignant high: penalty += max(0, P_malignant - α3)

P_malignant = sum of softmax probs over malignant classes (default: mel, bcc, akiec).

All thresholds/weights are configurable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstraintLoss(nn.Module):
    def __init__(
        self,
        malignant_indices=(0, 2, 3),
        concept_high=0.6,
        concept_low=0.3,
        diameter_mm_threshold=6.0,
        alpha1=0.6,
        alpha2=0.6,
        alpha3=0.7,
        weight=1.0,
    ):
        super().__init__()
        self.malignant_indices = malignant_indices  # e.g. (0, 2, 3) for mel, bcc, akiec
        self.concept_high = concept_high
        self.concept_low = concept_low
        self.diameter_mm_threshold = diameter_mm_threshold
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.weight = weight

    def forward(self, class_logits, risk_score, concepts, diameter_mm):
        """Compute constraint penalties.

        Args:
            class_logits: (B, 7) raw logits.
            risk_score: (B, 1) or (B,) sigmoid malignancy risk.
            concepts: (B, 4) [A, B, C, D_normalized].
            diameter_mm: (B,) estimated diameter in mm.

        Returns:
            dict with keys:
              total: scalar total constraint loss.
              rule1, rule2, rule3: per-rule losses for logging.
              violated_1, violated_2, violated_3: bool tensors indicating violations.
        """
        if risk_score.dim() == 2:
            risk_score = risk_score.squeeze(-1)

        probs = F.softmax(class_logits, dim=-1)
        p_malignant = probs[:, self.malignant_indices].sum(dim=-1)  # (B,)

        A, B, C, D_norm = concepts[:, 0], concepts[:, 1], concepts[:, 2], concepts[:, 3]

        all_high = (A > self.concept_high) & (B > self.concept_high) & (C > self.concept_high)
        all_low = (A < self.concept_low) & (B < self.concept_low) & (C < self.concept_low)
        large_diameter = (diameter_mm >= self.diameter_mm_threshold).float()

        rule1_penalty = torch.relu(self.alpha1 - p_malignant) * all_high.float()
        rule2_penalty = torch.relu(self.alpha2 - p_malignant) * large_diameter
        rule3_penalty = torch.relu(p_malignant - self.alpha3) * all_low.float()

        rule1_loss = rule1_penalty.mean()
        rule2_loss = rule2_penalty.mean()
        rule3_loss = rule3_penalty.mean()

        total_loss = (rule1_loss + rule2_loss + rule3_loss) * self.weight

        return {
            "total": total_loss,
            "rule1": rule1_loss,
            "rule2": rule2_loss,
            "rule3": rule3_loss,
            "violated_1": all_high,
            "violated_2": large_diameter > 0,
            "violated_3": all_low,
        }
