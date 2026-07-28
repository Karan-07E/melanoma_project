"""Diagnosis Head: Multi-task output from attended features + concept vector.

Input: (pooled attended feature maps + concept vector) → Shared trunk → two branches
  - Disease classification: 7-way softmax
  - Malignancy Risk Score: 1-way sigmoid

Architecture: GAP → Concat[1280 || 4] → SharedMLP(1284→256→128) → {class, risk}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagnosisHead(nn.Module):
    def __init__(
        self,
        feature_dim=1280,
        concept_dim=4,
        num_classes=7,
        dropout=0.3,
    ):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)

        combined_dim = feature_dim + concept_dim
        self.shared = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
        )

        self.class_head = nn.Linear(128, num_classes)
        self.risk_head = nn.Linear(128, 1)

    def forward(self, attended_feats, concept_vec):
        """Forward pass through diagnosis head.

        Args:
            attended_feats: (B, C, H, W) attention-modulated feature maps.
            concept_vec: (B, 4) concept bottleneck vector.

        Returns:
            class_logits: (B, 7) raw logits for disease classification.
            risk_score: (B, 1) sigmoid-activated malignancy risk score.
        """
        pooled = self.gap(attended_feats).flatten(1)  # (B, C)
        fused = torch.cat([pooled, concept_vec], dim=1)  # (B, C + 4)
        shared = self.shared(fused)  # (B, 128)

        class_logits = self.class_head(shared)  # (B, 7)
        risk_score = torch.sigmoid(self.risk_head(shared))  # (B, 1)

        return class_logits, risk_score
