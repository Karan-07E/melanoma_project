"""Clinical Concept Heads: Predict ABCD scores from global feature vector.

Four independent MLP heads, each mapping 1280-d → 1 scalar in [0, 1].
D head includes a denormalization step to recover mm for constraint usage.
"""

import torch
import torch.nn as nn


class ConceptHead(nn.Module):
    """Single concept regression head: Linear → ReLU → Dropout → Linear → Sigmoid."""

    def __init__(self, input_dim=1280, hidden_dim=128, dropout=0.2, output_sigmoid=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.output_sigmoid = output_sigmoid

    def forward(self, x):
        out = self.net(x).squeeze(-1)  # (B,)
        if self.output_sigmoid:
            out = torch.sigmoid(out)
        return out


class AsymmetryHead(ConceptHead):
    def __init__(self, input_dim=1280, hidden_dim=128, dropout=0.2):
        super().__init__(input_dim, hidden_dim, dropout, output_sigmoid=True)


class BorderHead(ConceptHead):
    def __init__(self, input_dim=1280, hidden_dim=128, dropout=0.2):
        super().__init__(input_dim, hidden_dim, dropout, output_sigmoid=True)


class ColorHead(ConceptHead):
    def __init__(self, input_dim=1280, hidden_dim=128, dropout=0.2):
        super().__init__(input_dim, hidden_dim, dropout, output_sigmoid=True)


class DiameterHead(nn.Module):
    """Diameter head: predicts normalized [0,1] score, also recovers mm via denormalization.

    Raw output: sigmoid → normalized_area_score.
    Denormalized: diameter_mm = normalized_area_score * max_diameter_mm (default 20).
    """

    def __init__(self, input_dim=1280, hidden_dim=128, dropout=0.2, max_diameter_mm=20.0):
        super().__init__()
        self.max_diameter_mm = max_diameter_mm
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        raw = self.net(x).squeeze(-1)
        norm_score = torch.sigmoid(raw)
        diameter_mm = norm_score * self.max_diameter_mm
        return norm_score, diameter_mm


class ConceptHeads(nn.Module):
    """Aggregate all four concept heads.

    Forward returns:
        concepts: (B, 4) tensor [A, B, C, D_normalized]
        diameter_mm: (B,) tensor of estimated diameter in mm
    """

    def __init__(self, input_dim=1280, hidden_dim=128, dropout=0.2, max_diameter_mm=20.0):
        super().__init__()
        self.asymmetry = AsymmetryHead(input_dim, hidden_dim, dropout)
        self.border = BorderHead(input_dim, hidden_dim, dropout)
        self.color = ColorHead(input_dim, hidden_dim, dropout)
        self.diameter = DiameterHead(input_dim, hidden_dim, dropout, max_diameter_mm)

    def forward(self, global_vec):
        a = self.asymmetry(global_vec)
        b = self.border(global_vec)
        c = self.color(global_vec)
        d_norm, d_mm = self.diameter(global_vec)

        concepts = torch.stack([a, b, c, d_norm], dim=-1)  # (B, 4)
        return concepts, d_mm
