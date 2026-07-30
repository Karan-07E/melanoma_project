"""Temperature scaling for model calibration.

Applies a learned temperature parameter T to logits before softmax:
  calibrated_probs = softmax(logits / T)

T > 1: softens probabilities (reduces overconfidence)
T < 1: sharpens probabilities
T = 1: identity

Temperature is optimized on validation set to minimize NLL/ECE.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS


class TemperatureScaler(nn.Module):
    """Learn a single temperature parameter for logit calibration.

    Usage:
        scaler = TemperatureScaler()
        scaler.fit(logits_val, labels_val, lr=0.01, max_iter=100)
        calibrated_logits = scaler.calibrate(logits_test)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def calibrate(self, logits):
        """Apply temperature scaling to logits. Returns calibrated logits."""
        self.eval()
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits).float()
        with torch.no_grad():
            return self(logits)

    def fit(self, logits, labels, lr=0.01, max_iter=200):
        """Optimize temperature on validation set.

        Args:
            logits: (N, C) raw logits from model.
            labels: (N,) ground truth class indices.
            lr: Learning rate for LBFGS optimizer.
            max_iter: Maximum LBFGS iterations.

        Returns:
            self
        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits).float()
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long()

        self.train()
        optimizer = LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            calibrated_logits = self(logits)
            loss = F.cross_entropy(calibrated_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.eval()

        print(f"Temperature scaling: T = {self.temperature.item():.4f}")
        return self
