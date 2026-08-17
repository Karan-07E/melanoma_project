"""CORAL (Correlation Alignment) loss for domain adaptation.

Minimizes the difference between second-order statistics (covariance
matrices) of source and target domain feature distributions.

L_coral = 1 / (4 * d²) * ||C_source - C_target||²_F

where C_s and C_t are d×d covariance matrices of the source and
target feature vectors respectively.

Unlike DANN (adversarial), CORAL is a simple statistical matching
with no adversarial training instability. Both can be used together.

Strategy 4 from the domain generalization roadmap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def coral_loss(source_features, target_features):
    """Compute CORAL loss between two batches of feature vectors.

    Args:
        source_features: (B_s, D) features from source domain (HAM10000).
        target_features: (B_t, D) features from target domain (PAD-UFES-20).

    Returns:
        Scalar CORAL loss value.
    """
    d = source_features.size(1)
    n_s = source_features.size(0)
    n_t = target_features.size(0)

    if n_s < 2 or n_t < 2:
        return source_features.sum() * 0.0

    source_centered = source_features - source_features.mean(dim=0, keepdim=True)
    target_centered = target_features - target_features.mean(dim=0, keepdim=True)

    cov_source = (source_centered.t() @ source_centered) / (n_s - 1)
    cov_target = (target_centered.t() @ target_centered) / (n_t - 1)

    diff = cov_source - cov_target
    loss = (diff * diff).sum() / (4.0 * d * d)

    return loss


def mean_alignment_loss(source_features, target_features):
    """Match first-order feature statistics alongside covariance statistics."""
    if source_features.size(0) == 0 or target_features.size(0) == 0:
        return source_features.sum() * 0.0
    return (source_features.mean(dim=0) - target_features.mean(dim=0)).pow(2).mean()


def mmd_loss(source_features, target_features, kernel="rbf", sigma=1.0):
    """Compute Maximum Mean Discrepancy (MMD) between two feature distributions.

    MMD measures the distance between mean embeddings of two distributions
    in a reproducing kernel Hilbert space (RKHS).

    Alternative to CORAL — both serve the same purpose of aligning
    feature distributions across domains.

    Args:
        source_features: (B_s, D) source domain features.
        target_features: (B_t, D) target domain features.
        kernel: 'rbf' or 'linear'.
        sigma: RBF kernel bandwidth.

    Returns:
        Scalar MMD loss value.
    """
    n_s = source_features.size(0)
    n_t = target_features.size(0)
    if n_s < 2 or n_t < 2:
        return source_features.sum() * 0.0

    if kernel == "linear":
        source_mean = source_features.mean(dim=0)
        target_mean = target_features.mean(dim=0)
        return ((source_mean - target_mean) ** 2).sum().sqrt()

    combined = torch.cat([source_features, target_features], dim=0)

    def _rbf_kernel(x, y):
        xx = (x * x).sum(dim=1, keepdim=True)
        yy = (y * y).sum(dim=1, keepdim=True)
        dist = xx + yy.t() - 2 * (x @ y.t())
        return torch.exp(-dist / (2.0 * sigma * sigma))

    k_ss = _rbf_kernel(source_features, source_features)
    k_tt = _rbf_kernel(target_features, target_features)
    k_st = _rbf_kernel(source_features, target_features)

    mmd = (
        k_ss.sum() / (n_s * n_s)
        + k_tt.sum() / (n_t * n_t)
        - 2.0 * k_st.sum() / (n_s * n_t)
    )

    return mmd.clamp(min=0.0).sqrt()


class AlignmentLoss(nn.Module):
    """Unified domain alignment loss (CORAL + optional MMD).

    Configurable via mode:
      - 'coral':  Correlation Alignment (default, stable)
      - 'mmd':    Maximum Mean Discrepancy
      - 'both':   Sum of CORAL + MMD
    """

    def __init__(self, mode="coral", weight=1.0, normalize=True, mean_weight=0.25):
        super().__init__()
        self.mode = mode
        self.weight = weight
        self.normalize = normalize
        self.mean_weight = mean_weight

    def forward(self, source_features, target_features):
        if self.normalize:
            source_features = F.normalize(source_features, dim=1)
            target_features = F.normalize(target_features, dim=1)

        if self.mode == "coral":
            loss = coral_loss(source_features, target_features)
        elif self.mode == "mmd":
            loss = mmd_loss(source_features, target_features, kernel="rbf")
        elif self.mode == "both":
            loss = coral_loss(source_features, target_features) + mmd_loss(source_features, target_features)
        else:
            raise ValueError(f"Unknown alignment mode: {self.mode}")

        mean_loss = mean_alignment_loss(source_features, target_features)
        return self.weight * (loss + self.mean_weight * mean_loss)
