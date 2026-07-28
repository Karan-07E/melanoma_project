"""Shared Encoder: EfficientNetV2-S backbone with dual output.

Outputs:
  - global_vec: (B, feature_dim)  — pooled feature vector after GAP
  - spatial_feats: (B, feature_dim, H, W) — feature maps before pooling

Uses timm with num_classes=0 (pretrained) to get the full feature extractor,
then extracts both the spatial feature map and the GAP-pooled vector.
"""

import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class SharedEncoder(nn.Module):
    def __init__(
        self,
        backbone_name="efficientnetv2_s",
        pretrained=True,
        freeze_stages=6,
        img_size=224,
    ):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm required: pip install timm>=0.9.12")

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        dummy = torch.randn(1, 3, img_size, img_size)
        with torch.no_grad():
            feat_map = self.backbone(dummy)
        self._global_dim = feat_map.shape[1]
        self._spatial_size = (feat_map.shape[2], feat_map.shape[3])

        self._img_size = img_size

        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)

    def _freeze_stages(self, num_stages):
        stage_prefixes = [f"stages.{i}" for i in range(num_stages)]
        for name, param in self.backbone.named_parameters():
            for prefix in stage_prefixes:
                if name.startswith(prefix):
                    param.requires_grad = False
                    break

    def forward(self, x):
        spatial_feats = self.backbone(x)
        global_vec = self.global_pool(spatial_feats).flatten(1)
        return global_vec, spatial_feats

    @property
    def global_dim(self):
        return self._global_dim

    @property
    def spatial_size(self):
        return self._spatial_size
