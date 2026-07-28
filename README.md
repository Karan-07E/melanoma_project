# Clinically Interpretable Melanoma Diagnosis and Malignancy Risk Assessment in Dermoscopic Images Using EfficientNetV2-S with Concept Bottleneck Learning and Clinical Constraint Optimization

**DISCLAIMER: For research and educational use only. Not a diagnostic device. Not for clinical decision-making. This is a research prototype and must not be used for actual medical diagnosis.**

## Overview

This project implements a **Concept Bottleneck Model (CBM)** for skin lesion analysis that:

1. Takes a 224×224 dermoscopic RGB image as input
2. Predicts four interpretable **ABCD clinical concepts**: Asymmetry, Border irregularity, Color variation, Diameter (Normalized Lesion Area)
3. Forces predictions through a **strict 4-dimensional concept bottleneck** (no shortcut path from raw features to diagnosis)
4. Uses the concept vector to compute **concept-guided spatial attention** over encoder feature maps
5. Produces two final outputs: **7-class disease classification** (mel, nv, bcc, akiec, bkl, df, vasc) and a **Malignancy Risk Score** (0–1)
6. Trains with a **multi-task loss** combining diagnosis CE, concept MSE, and **soft clinical-consistency constraints**
7. Evaluates in-domain on **HAM10000** and cross-domain on **PAD-UFES-20**

## Architecture

```
Dermoscopic Image (224×224×3)
        │
        ▼
┌─────────────────────────┐
│  Shared Encoder          │  EfficientNetV2-S
│  Global Vector 1280-d    │  + Spatial Feature Maps (1280×7×7)
└──────────┬──────────────┘
           │
     ┌─────┴─────────────┐
     ▼                   ▼
┌─────────┐      ┌──────────────────────┐
│ 4 Concept│      │ Concept-Guided       │
│ Heads    │      │ Spatial Attention    │
│ A B C D  │      │ (4→64→H×W → Sigmoid) │
└────┬─────┘      └──────────┬───────────┘
     │                       │
     ▼  Concept Bottleneck   ▼
  [A,B,C,D]₄          Attended Feature Maps
     │                       │
     └───────┬───────────────┘
             ▼
┌──────────────────────────┐
│     Diagnosis Head        │
│  GAP + Concat → MLP      │
│  ┌────────┐ ┌───────────┐ │
│  │7-class │ │Malignancy │  │
│  │Softmax │ │Risk [0,1] │  │
│  └────────┘ └───────────┘ │
└──────────────────────────┘
```

## Quick Start (Local, No Downloads)

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate synthetic smoke-test data
python scripts/generate_synthetic_data.py

# 3. Precompute ABCD pseudo clinical concepts
python scripts/precompute_abcd_targets.py --data data/synthetic

# 4. Train (3 epochs on CPU, ~5-10 min)
python src/train.py --config configs/default.yaml --data data/synthetic --epochs 3

# 5. Evaluate
python src/evaluate.py --checkpoint models/best.pt --data data/synthetic

# 6. Launch demo app
python app/demo_app.py --checkpoint models/best.pt

# 7. Run tests
pytest tests/ -v
```

## Training with Real Data

### HAM10000 (in-domain)

1. Download HAM10000 from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
2. Extract to `data/ham10000/`
3. Run: `python scripts/download_ham10000.py` to verify

```bash
python scripts/precompute_abcd_targets.py --data data/ham10000
python src/train.py --data data/ham10000 --epochs 60
python src/evaluate.py --checkpoint models/best.pt --data data/ham10000
```

### PAD-UFES-20 (cross-domain evaluation)

1. Download from [Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1)
2. Extract to `data/pad_ufes20/`
3. Run: `python scripts/download_pad_ufes20.py` to verify

```bash
python src/evaluate.py --checkpoint models/best.pt --data data/pad_ufes20
```

## Project Structure

```
melanoma-cbm/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml               # All hyperparameters
├── data/
│   ├── synthetic/                  # Auto-generated smoke-test data
│   └── abcd_cache/                 # Precomputed ABCD targets
├── scripts/
│   ├── download_ham10000.py
│   ├── download_pad_ufes20.py
│   ├── generate_synthetic_data.py
│   └── precompute_abcd_targets.py
├── src/
│   ├── models/
│   │   ├── encoder.py              # EfficientNetV2-S
│   │   ├── concept_heads.py        # ABCD concept heads
│   │   ├── attention.py            # Concept-guided attention
│   │   ├── diagnosis_head.py       # Multi-task output head
│   │   └── cbm_model.py            # Full CBM model
│   ├── losses/
│   │   ├── concept_loss.py
│   │   ├── constraint_loss.py
│   │   └── multitask_loss.py
│   ├── data/
│   │   ├── abcd_targets.py         # ABCD pseudo concept labeler
│   │   ├── datasets.py             # Dataset classes
│   │   └── transforms.py           # Data transforms
│   ├── train.py
│   ├── evaluate.py
│   └── utils/
│       ├── metrics.py              # ECE, AUC, per-class metrics
│       └── viz.py                  # Visualization utilities
├── app/
│   └── demo_app.py                 # Gradio demo
├── tests/
│   ├── test_abcd_targets.py
│   ├── test_model_shapes.py
│   └── test_constraints.py
└── notebooks/
    └── walkthrough.ipynb
```

## Key Features

- **Hard Concept Bottleneck**: Diagnosis head only sees (attended features + concept vector), never raw global features
- **Multi-Task Loss**: `L_total = L_diagnosis + 0.5 × L_concept + 0.1 × L_constraint`
- **Clinical Constraints**: Soft penalties for clinically inconsistent predictions
  - High ABCD → should raise malignancy probability
  - Large lesion (D ≥ 6mm) → should raise malignancy probability
  - All concepts low but high malignancy → penalize
- **Concept-Guided Attention**: ABCD concepts steer spatial attention to clinically relevant regions
- **Pseudo Clinical Concepts**: ABCD targets auto-generated via classical CV (SSIM, isoperimetric ratio, HSV entropy, normalized area) — no manual annotation needed
- **Cross-Domain Evaluation**: Train on HAM10000 (European skin tones), test on PAD-UFES-20 (Brazilian, diverse skin tones)

## Evaluation Metrics

The evaluation script (`src/evaluate.py`) produces:
- In-domain accuracy (HAM10000)
- Cross-domain accuracy (PAD-UFES-20)
- Per-class precision, recall, F1
- Concept prediction MAE (per concept + overall)
- ECE and AUC for Malignancy Risk Score
- Constraint violation rates

## Configuration

All hyperparameters are in `configs/default.yaml`:
- Loss weights: `lambda_concept`, `lambda_constraint`
- Constraint thresholds: `concept_high`, `concept_low`, `diameter_mm_threshold`
- Training: lr, epochs, patience, batch size
- Model: backbone, dimensions, dropout, attention mode

Override via CLI: `python src/train.py --lambda-concept 0.3 --epochs 30`

## License

This project is for research and educational purposes only. See DISCLAIMER at top.

## Citation

If you use this work, please cite:

```bibtex
@software{melanoma_cbm_2024,
  title = {Clinically Interpretable Melanoma Diagnosis and Malignancy Risk Assessment
           Using EfficientNetV2-S with Concept Bottleneck Learning},
  year = {2024},
  note = {Research prototype. Not a diagnostic device.}
}
```
