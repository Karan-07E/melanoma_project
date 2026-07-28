#!/usr/bin/env python3
"""Local Gradio demo app for CBM melanoma analysis.

Upload a dermoscopic image → See ABCD concept scores, attention heatmap,
disease diagnosis, and malignancy risk assessment.

**DISCLAIMER: For research and educational use only. Not a diagnostic device.
Not for clinical decision-making.**
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import gradio as gr
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.cbm_model import CBMModel
from src.utils.viz import (
    get_risk_level,
    create_concept_chart,
    create_prediction_figure,
    CONCEPT_NAMES,
)
from src.data.datasets import CLASS_NAMES


DISCLAIMER = (
    "**DISCLAIMER**: For research and educational use only. "
    "Not a diagnostic device. Not for clinical decision-making. "
    "This is a research prototype and must not be used for actual medical diagnosis."
)

RISK_COLORS = {
    "LOW": "#27ae60",
    "MODERATE": "#f39c12",
    "HIGH": "#e67e22",
    "VERY HIGH": "#e74c3c",
}


def load_model(checkpoint_path, config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    model_cfg["img_size"] = cfg["data"]["img_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu")

    model = CBMModel(model_cfg).to(device)
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from {checkpoint_path}")
    else:
        print("WARNING: No checkpoint provided or found. Using untrained model weights.")
    model.eval()
    return model, device


def predict(image, model, device):
    if image is None:
        return None, None, "No image uploaded."

    pil_img = Image.fromarray(image).convert("RGB").resize((224, 224))
    img_array = np.array(pil_img, dtype=np.float32) / 255.0
    img_array = (img_array - np.float32([0.485, 0.456, 0.406])) / np.float32([0.229, 0.224, 0.225])
    img_tensor = torch.from_numpy(img_array.copy()).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model.predict(img_tensor)

    concepts = pred["concepts"][0].cpu().numpy()
    class_probs = pred["class_probs"][0].cpu().numpy()
    risk_score = float(pred["risk_score"][0].cpu().item())
    risk_level = get_risk_level(risk_score)
    pred_class = int(pred["predicted_class"][0].cpu().item())
    attn_map = pred["attn_map"][0].cpu().numpy()

    concept_chart = create_concept_chart(concepts)

    pred_fig = create_prediction_figure(
        np.array(pil_img), concepts, class_probs, risk_score, attn_map
    )

    diagnosis_text = f"""
### Diagnosis
**Predicted Class**: {CLASS_NAMES[pred_class]}

**Malignancy Risk Score**: {risk_score:.3f}

**Risk Level**: <span style='color:{RISK_COLORS[risk_level]};font-weight:bold;font-size:1.2em'>{risk_level}</span>

### Clinical Concepts (ABCD)
| Concept | Score |
|---------|-------|
| Asymmetry (A) | {concepts[0]:.3f} |
| Border Irregularity (B) | {concepts[1]:.3f} |
| Color Variation (C) | {concepts[2]:.3f} |
| Lesion Area (Normalized) (D) | {concepts[3]:.3f} |

### Class Probabilities
| Class | Probability |
|-------|-------------|
"""
    for i, name in enumerate(CLASS_NAMES):
        diagnosis_text += f"| {name} | {class_probs[i]:.1%} |\n"

    diagnosis_text += f"\n---\n{DISCLAIMER}"

    return pred_fig, concept_chart, diagnosis_text


def create_demo():
    model, device = load_model(None)

    with gr.Blocks(title="CBM Melanoma Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Clinically Interpretable Melanoma Diagnosis and Malignancy Risk Assessment\n"
            "### Concept Bottleneck Model (CBM) with EfficientNetV2-S"
        )
        gr.Markdown(
            f"**{DISCLAIMER}**"
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Upload Dermoscopic Image", type="numpy")
                analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

                gr.Markdown("""
                ### How It Works
                1. Upload a dermoscopic image (224x224 recommended)
                2. The model predicts 4 ABCD clinical concepts
                3. Concepts guide spatial attention on the lesion
                4. Final diagnosis + malignancy risk are computed
                5. Review the attention heatmap, concepts, and risk assessment
                """)

            with gr.Column(scale=1):
                prediction_img = gr.Image(label="Prediction & Attention Heatmap", type="numpy")
                concept_img = gr.Image(label="ABCD Concept Scores", type="numpy")

        with gr.Row():
            diagnosis_output = gr.Markdown("### Analysis will appear here")

        analyze_btn.click(
            fn=lambda img: predict(img, model, device),
            inputs=[image_input],
            outputs=[prediction_img, concept_img, diagnosis_output],
        )

        gr.Markdown(
            "### About This System\n\n"
            "This is a **research prototype** implementing a Concept Bottleneck Model (CBM) "
            "for skin lesion analysis. The model:\n\n"
            "- Predicts 4 interpretable ABCD clinical concepts (Asymmetry, Border, Color, Diameter)\n"
            "- Uses these concepts to guide spatial attention to clinically relevant regions\n"
            "- Produces a 7-class disease classification and a malignancy risk score\n"
            "- Enforces clinical consistency via soft constraint penalties during training\n\n"
            "**IMPORTANT**: This system is for research and educational purposes only. "
            "It is NOT a medical device and must NOT be used for clinical diagnosis or "
            "treatment decisions. Always consult a qualified dermatologist."
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="CBM Melanoma Analysis Demo")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--port", type=int, default=7860, help="Port for the demo app")
    args = parser.parse_args()

    global model, device
    if args.checkpoint:
        model, device = load_model(args.checkpoint)
    else:
        model, device = load_model(None)

    demo = create_demo()
    demo.launch(server_port=args.port, share=False)


if __name__ == "__main__":
    model, device = None, None
    main()
