"""Full CBM Model: Wires SharedEncoder → ConceptHeads → ConceptAttention → DiagnosisHead.

Enforces the HARD CONCEPT BOTTLENECK:
  - The diagnosis head receives ONLY (attended features + concept vector).
  - The raw 1280-d global vector does NOT flow into the diagnosis head.
  - The only path from raw features to diagnosis is through the concept bottleneck.

Confirmed by test_model_shapes.py which checks gradient flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import SharedEncoder
from src.models.concept_heads import ConceptHeads
from src.models.attention import ConceptGuidedAttention
from src.models.diagnosis_head import DiagnosisHead


class CBMModel(nn.Module):
    def __init__(self, config: dict):
        """Build the full CBM model from a config dictionary.

        Args:
            config: Dict with keys matching configs/default.yaml model section.
                    Expects: backbone, pretrained, global_dim, concept_dim,
                    num_classes, dropout_rate, attention_mode.
        """
        super().__init__()

        backbone_name = config.get("backbone", "efficientnetv2_s")
        pretrained = config.get("pretrained", True)
        concept_dim = config.get("concept_dim", 4)
        num_classes = config.get("num_classes", 7)
        dropout_rate = config.get("dropout_rate", 0.3)
        attention_mode = config.get("attention_mode", "sigmoid")
        img_size = config.get("img_size", 224)

        self.encoder = SharedEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            img_size=img_size,
        )

        global_dim = self.encoder.global_dim

        self.concept_heads = ConceptHeads(
            input_dim=global_dim,
            hidden_dim=128,
            dropout=0.2,
            max_diameter_mm=config.get("constraints", {}).get("diameter_max_mm", 20.0),
        )
        self.attention = ConceptGuidedAttention(
            concept_dim=concept_dim,
            feature_dim=global_dim,
            spatial_h=self.encoder.spatial_size[0],
            spatial_w=self.encoder.spatial_size[1],
            hidden_dim=64,
            mode=attention_mode,
        )
        self.diagnosis_head = DiagnosisHead(
            feature_dim=global_dim,
            concept_dim=concept_dim,
            num_classes=num_classes,
            dropout=dropout_rate,
        )

        self._global_dim = global_dim
        self._concept_dim = concept_dim
        self._num_classes = num_classes

    def forward(self, x):
        """Full forward pass through the CBM pipeline.

        Args:
            x: (B, 3, H, W) input image tensor.

        Returns:
            dict with keys:
              concepts: (B, 4) clinical concept scores [A, B, C, D].
              diameter_mm: (B,) estimated diameter in mm.
              class_logits: (B, 7) disease classification logits.
              risk_score: (B, 1) malignancy risk score [0, 1].
              attn_map: (B, 1, H_a, W_a) attention weights.
              global_vec: (B, 1280) raw global vector (for analysis only).
        """
        global_vec, spatial_feats = self.encoder(x)
        concepts, diameter_mm = self.concept_heads(global_vec)
        attended_feats, attn_map = self.attention(concepts, spatial_feats)
        class_logits, risk_score = self.diagnosis_head(attended_feats, concepts)

        return {
            "concepts": concepts,
            "diameter_mm": diameter_mm,
            "class_logits": class_logits,
            "risk_score": risk_score,
            "attn_map": attn_map,
            "global_vec": global_vec,
        }

    def predict(self, x):
        """Convenience method returning processed outputs for inference.

        Returns:
            dict with class_probs, predicted_class, risk_score, risk_level,
            concepts, attn_map.
        """
        outputs = self.forward(x)
        class_probs = F.softmax(outputs["class_logits"], dim=-1)
        pred_class = class_probs.argmax(dim=-1)

        risk_score = outputs["risk_score"].squeeze(-1)

        return {
            "class_probs": class_probs,
            "predicted_class": pred_class,
            "risk_score": risk_score,
            "concepts": outputs["concepts"],
            "diameter_mm": outputs["diameter_mm"],
            "attn_map": outputs["attn_map"],
        }

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def concept_dim(self):
        return self._concept_dim
