"""Visualization utilities: attention heatmap overlay, concept bar chart, diagnosis display."""

import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F


CLASS_NAMES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
CONCEPT_NAMES = ["Asymmetry", "Border", "Color", "Normalized Area"]
RISK_LEVELS = ["LOW", "MODERATE", "HIGH", "VERY HIGH"]
RISK_THRESHOLDS = [0.0, 0.25, 0.50, 0.75]


def get_risk_level(risk_score):
    """Bucket risk score into risk level string."""
    for i, thresh in enumerate(RISK_THRESHOLDS[1:], 1):
        if risk_score < thresh:
            return RISK_LEVELS[i - 1]
    return RISK_LEVELS[-1]


def overlay_attention(image, attn_map, alpha=0.5):
    """Overlay attention heatmap on original image.

    Args:
        image: RGB numpy array (H, W, 3).
        attn_map: (H_a, W_a) numpy array, will be resized to image size.
        alpha: Blend factor.

    Returns:
        RGB numpy array (H, W, 3) with attention overlay.
    """
    if isinstance(attn_map, torch.Tensor):
        attn_map = attn_map.detach().cpu().numpy()
    if attn_map.ndim == 3:
        attn_map = attn_map[0]

    H, W = image.shape[:2]
    attn_resized = cv2.resize(attn_map, (W, H))

    attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

    heatmap = plt.cm.jet(attn_norm)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay


def create_prediction_figure(image, concepts, class_probs, risk_score, attn_map=None):
    """Create a comprehensive prediction visualization figure.

    Args:
        image: RGB numpy array (H, W, 3).
        concepts: (4,) numpy array or list [A, B, C, D].
        class_probs: (7,) numpy array or list of class probabilities.
        risk_score: Scalar float.
        attn_map: Optional (H_a, W_a) attention map.

    Returns:
        PIL Image of the combined figure.
    """
    if attn_map is not None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = [axes[0], axes[1], None]

    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontweight="bold")
    axes[0].axis("off")

    if attn_map is not None:
        attention_overlay = overlay_attention(image, attn_map)
        axes[1].imshow(attention_overlay)
        axes[1].set_title("Concept-Guided Attention", fontweight="bold")
        axes[1].axis("off")
        ax_prob = axes[2]
    else:
        ax_prob = axes[1]

    colors = ["#e74c3c" if p == max(class_probs) else "#3498db" for p in class_probs]
    bars = ax_prob.barh(CLASS_NAMES, class_probs, color=colors)
    ax_prob.set_xlabel("Probability")
    ax_prob.set_title(f"Diagnosis (Risk: {risk_score:.2f} — {get_risk_level(risk_score)})",
                      fontweight="bold")
    ax_prob.set_xlim(0, 1)

    for bar, prob in zip(bars, class_probs):
        ax_prob.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{prob:.1%}", va="center", fontsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return Image.open(buf)


def create_concept_chart(concepts):
    """Create a bar chart of ABCD concept scores.

    Args:
        concepts: (4,) array-like [A, B, C, D].

    Returns:
        PIL Image.
    """
    if isinstance(concepts, torch.Tensor):
        concepts = concepts.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    colors_bar = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]
    bars = ax.bar(CONCEPT_NAMES, concepts, color=colors_bar)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("ABCD Clinical Concepts", fontweight="bold")

    for bar, val in zip(bars, concepts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    return Image.open(buf)


def export_prediction_visualization(image_path, output, save_path):
    """Export a full prediction visualization to disk.

    Args:
        image_path: Path to the input image.
        output: Dict from CBMModel.predict().
        save_path: Path to save the visualization PNG.
    """
    image = np.array(Image.open(image_path).convert("RGB").resize((224, 224)))

    concepts = output["concepts"]
    if isinstance(concepts, torch.Tensor):
        concepts = concepts.detach().cpu().numpy()
    if concepts.ndim == 2:
        concepts = concepts[0]

    class_probs = output["class_probs"]
    if isinstance(class_probs, torch.Tensor):
        class_probs = class_probs.detach().cpu().numpy()
    if class_probs.ndim == 2:
        class_probs = class_probs[0]

    risk_score = output["risk_score"]
    if isinstance(risk_score, torch.Tensor):
        risk_score = risk_score.detach().cpu().item()
    if hasattr(risk_score, '__len__') and hasattr(risk_score, '__getitem__'):
        risk_score = float(np.array(risk_score).flatten()[0])

    attn_map = output.get("attn_map")
    if attn_map is not None and isinstance(attn_map, torch.Tensor):
        attn_map = attn_map.detach().cpu().numpy()
        if attn_map.ndim == 4:
            attn_map = attn_map[0, 0]

    fig = create_prediction_figure(image, concepts, class_probs, risk_score, attn_map)
    fig.save(save_path, dpi=150)
    print(f"Visualization saved to: {save_path}")
