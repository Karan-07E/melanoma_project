"""Full CBM Model: Wires SharedEncoder → ConceptHeads → ConceptAttention → DiagnosisHead.

Enforces the HARD CONCEPT BOTTLENECK:
  - The diagnosis head receives ONLY (attended features + concept vector).
  - The raw global vector does NOT flow into the diagnosis head.

Optional domain adversarial head (Strategy 3):
  - DomainClassifier attached to encoder's global features.
  - Used during training only — not in inference.

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

        self.domain_classifier = None
        domain_cfg = config.get("domain", {})
        if domain_cfg.get("enabled", False):
            from src.models.domain_adversarial import DomainClassifier
            self.domain_classifier = DomainClassifier(
                feature_dim=global_dim,
                hidden_dim=domain_cfg.get("domain_classifier_hidden", 64),
                dropout=dropout_rate,
            )

    def forward(self, x, return_domain_features=False):
        """Full forward pass through the CBM pipeline.

        Args:
            x: (B, 3, H, W) input image tensor.
            return_domain_features: If True, also return raw global_vec
                                    for domain classifier use.

        Returns:
            dict with keys: concepts, diameter_mm, class_logits,
            risk_score, attn_map, global_vec.
        """
        global_vec, spatial_feats = self.encoder(x)
        concepts, diameter_mm = self.concept_heads(global_vec)
        attended_feats, attn_map = self.attention(concepts, spatial_feats)
        class_logits, risk_score = self.diagnosis_head(attended_feats, concepts)

        result = {
            "concepts": concepts,
            "diameter_mm": diameter_mm,
            "class_logits": class_logits,
            "risk_score": risk_score,
            "attn_map": attn_map,
            "global_vec": global_vec,
        }
        return result

    def domain_forward(self, global_vec, reverse=True, lambda_val=1.0):
        """Forward pass through the domain classifier (for DANN)."""
        if self.domain_classifier is None:
            return None
        return self.domain_classifier(global_vec, reverse=reverse, lambda_val=lambda_val)

    def predict(self, x):
        """Convenience method returning processed outputs for inference."""
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
