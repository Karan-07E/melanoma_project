"""Concept-Guided Spatial Attention.

Maps the 4-dim concept vector to spatial attention weights over the
encoder's H×W feature maps, then performs elementwise multiplication.

Two modes (configurable via attention_mode):
  - sigmoid:  Independent gating per spatial position.
  - softmax:  Softmax over all spatial positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptGuidedAttention(nn.Module):
    def __init__(
        self,
        concept_dim=4,
        feature_dim=1280,
        spatial_h=7,
        spatial_w=7,
        hidden_dim=64,
        mode="sigmoid",
    ):
        super().__init__()
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        self.feature_dim = feature_dim
        self.mode = mode

        self.mlp = nn.Sequential(
            nn.Linear(concept_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, spatial_h * spatial_w),
        )

    def forward(self, concept_vec, spatial_feats):
        """Apply concept-guided attention to spatial feature maps.

        Args:
            concept_vec: (B, 4) concept vector [A, B, C, D].
            spatial_feats: (B, C, H, W) shared feature maps.

        Returns:
            attended_feats: (B, C, H, W) attention-modulated feature maps.
            attn_map: (B, 1, H, W) attention weights for visualization.
        """
        B = concept_vec.size(0)
        logits = self.mlp(concept_vec)  # (B, H*W)

        if self.mode == "softmax":
            attn_flat = F.softmax(logits, dim=-1)
        else:
            attn_flat = torch.sigmoid(logits)

        attn_map = attn_flat.view(B, 1, self.spatial_h, self.spatial_w)  # (B, 1, H, W)
        attended_feats = spatial_feats * attn_map  # broadcast across C

        return attended_feats, attn_map
