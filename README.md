# Clinically Interpretable Melanoma Diagnosis and Malignancy Risk Assessment in Dermoscopic Images Using EfficientNetV2-S with Concept Bottleneck Learning and Clinical Constraint Optimization

**DISCLAIMER**: For research and educational use only. Not a diagnostic device. Not for clinical decision-making. This is a research prototype and must not be used for actual medical diagnosis.

---

## Overview

This project implements a **Concept Bottleneck Model (CBM)** for skin lesion analysis that:

1. Takes a 224×224 dermoscopic RGB image as input
2. Predicts four interpretable **ABCD clinical concepts**: Asymmetry, Border irregularity, Color variation, Diameter (Normalized Lesion Area)
3. Forces predictions through a **strict 4-dimensional concept bottleneck** (no shortcut path from raw features to diagnosis)
4. Uses the concept vector to compute **concept-guided spatial attention** over encoder feature maps
5. Produces two final outputs: **7-class disease classification** (mel, nv, bcc, akiec, bkl, df, vasc) and a **Malignancy Risk Score** (0–1)
6. Trains with a **multi-task loss** combining diagnosis CE, concept MSE, and **soft clinical-consistency constraints**
7. Evaluates in-domain on **HAM10000** and cross-domain on **PAD-UFES-20**

---

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

---

## Quick Start (Real Data)

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download HAM10000 and PAD-UFES-20 manually under data/, then verify
python scripts/download_ham10000.py
python scripts/download_pad_ufes20.py

# 3. Precompute ABCD pseudo clinical concepts for HAM10000
python scripts/precompute_abcd_targets.py --data data/ham10000

# 4. Train with unsupervised PAD-UFES-20 domain adaptation
python src/train.py --data data/ham10000 --pad-data data/pad_ufes20 --epochs 60

# 5. Evaluate on PAD-UFES-20
python src/evaluate.py --checkpoint models/best.pt --data data/pad_ufes20 --tta

# 6. Launch demo app (auto-downloads pretrained checkpoint if missing)
python app/demo_app.py

# 7. Run tests
pytest tests/ -v
```

---

## Pretrained Models

Pretrained checkpoints are hosted on Hugging Face Hub. The demo app auto-downloads them if a local checkpoint is missing.

### Manual download

```bash
python scripts/download_models.py
```

Models land in `models/` — ready for evaluation, demo, or fine-tuning.

### Upload your own

```bash
hf upload your-username/melanoma-cbm models/ .
```

Then update `HF_REPO` in `scripts/download_models.py` and `app/demo_app.py`.

---

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

### Recommended training (with all improvements + domain adaptation)

```bash
# Full domain-adaptive training (DANN + CORAL + MixUp + heavy aug + class weights)
python src/train.py \
  --data data/ham10000 \
  --pad-data data/pad_ufes20 \
  --epochs 60 \
  --img-size 320 \
  --batch-size 16

# Evaluate in-domain
python src/evaluate.py --checkpoint models/best.pt --data data/ham10000

# Evaluate cross-domain with Test-Time Augmentation
python src/evaluate.py --checkpoint models/best.pt --data data/pad_ufes20 --tta

# Optimize threshold + calibrate temperature
python scripts/optimize_threshold.py --checkpoint models/best.pt --data data/ham10000
```

### Baseline training (no domain adaptation, for comparison)

```bash
python src/train.py --data data/ham10000 --disable-domain --disable-mixup --epochs 50 --img-size 320 --batch-size 16
```

---

## Results

### Baseline (trained on HAM10000, no class weights, no sampler)

#### In-Domain: HAM10000 test set (1,509 samples)

| Metric | Value |
|--------|-------|
| Accuracy | **84.36%** |
| Macro F1 | 0.7163 |
| Weighted F1 | 0.8416 |
| Melanoma Precision | 0.578 |
| Melanoma Recall | 0.554 |
| Melanoma F1 | **0.565** |
| Malignancy AUC | 0.747 |
| ECE (calibration error) | **0.289** |

**Per-class breakdown:**

| Class | Support | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| **mel** | 168 | 0.578 | **0.554** | **0.565** |
| nv | 1,007 | 0.912 | 0.944 | 0.928 |
| bcc | 78 | 0.913 | 0.808 | 0.857 |
| akiec | 50 | 0.531 | 0.680 | 0.596 |
| bkl | 166 | 0.770 | 0.645 | 0.702 |
| df | 18 | 0.647 | 0.611 | 0.629 |
| vasc | 22 | 0.875 | 0.636 | 0.737 |

#### Cross-Domain: PAD-UFES-20 (2,298 samples, Brazilian/diverse skin tones)

| Metric | Value |
|--------|-------|
| Accuracy | **23.54%** |
| Macro F1 | 0.1500 |
| Weighted F1 | 0.2380 |
| Melanoma Recall | 0.173 |
| Melanoma F1 | 0.095 |
| Malignancy AUC | 0.542 |
| ECE | 0.327 |

**Key insight**: The model trained on European skin tones (HAM10000) drops from 84.4% to 23.5% accuracy on Brazilian/diverse skin tones (PAD-UFES-20). Melanoma recall collapses from 55% to 17%. This demonstrates a severe cross-domain performance gap driven by skin-tone distribution shift.

---

## Confusion Analysis — Melanoma Misclassifications

The most clinically critical analysis: **where do melanomas go?**

| Melanoma classified as → | Count | % | Clinical severity |
|---------------------------|-------|---|-------------------|
| **nv** (benign mole) | **59** | **35.1%** | **CRITICAL** — melanoma mistaken for harmless mole |
| bkl (benign keratosis) | 7 | 4.2% | High — benign diagnosis, missed cancer |
| akiec (pre-cancerous) | 5 | 3.0% | Moderate — still gets referral |
| df (dermatofibroma) | 3 | 1.8% | High — benign diagnosis |
| vasc (vascular) | 1 | 0.6% | High — benign diagnosis |
| **Total missed melanomas** | **75** | **44.6%** | — |
| **Correctly identified** | **93** | **55.4%** | — |

**35.1% of all melanomas are classified as benign nevi** — the most dangerous possible error (a cancerous lesion labeled harmless). This is driven by severe class imbalance: 1,007 nv vs 168 mel (6:1 ratio) with an unweighted loss function.

---

## Repository Audit — Key Findings

An engineering audit of the baseline training pipeline identified these issues:

| Component | Finding | Severity |
|-----------|---------|----------|
| **Loss function** | CrossEntropyLoss with no class weights despite 6:1 imbalance | **Critical** |
| **Sampling** | Random shuffle only — minority classes starved per batch | **Critical** |
| **Early stopping** | Tracked val_loss instead of val_macro_f1 | High — biases toward majority class |
| **Calibration** | ECE computed but no temperature scaling applied | High — 0.289 ECE is clinically dangerous |
| **Threshold** | Argmax only — no melanoma-specific threshold tuning | Medium |
| **Resolution** | 224×224 fixed (native HAM10000 is 600×450) | Medium |
| **Augmentation** | Deprecated ShiftScaleRotate, no hair artifact simulation | Low |

---

## Improvements Implemented

### 1. Class Balance — Inverse-Frequency Weights + WeightedRandomSampler

Class weights computed from the **training split only** (no data leakage):

```
weight[c] = N_samples / (num_classes × count[c])
```

`WeightedRandomSampler` with replacement ensures balanced per-epoch class exposure.

**Expected impact**: Melanoma recall +15–20%, macro F1 +8–12%, significantly fewer mel→nv misclassifications.

### 2. Early Stopping on Validation Macro F1

Previously tracked `val_loss` (dominated by majority class performance). Now tracks `val_macro_f1` which equally weights all 7 classes.

**Expected impact**: Model selected at peak minority class performance, not peak overall accuracy.

### 3. Configurable Image Resolution

```bash
python src/train.py --img-size 320   # better detail preservation
python src/train.py --img-size 224   # original (faster, less memory)
```

HAM10000 native resolution is 600×450. Training at 320×320 preserves more clinical detail.

### 4. Temperature Scaling for Calibration

New module: `src/utils/temperature_scaling.py` — learns a temperature parameter T via LBFGS on validation logits:

```
calibrated_probs = softmax(logits / T)
```

**Expected impact**: ECE drops from 0.289 → < 0.05. Risk scores become clinically meaningful.

### 5. Melanoma Threshold Optimization

New script: `scripts/optimize_threshold.py` — searches thresholds [0.05, 0.95] on the **validation set only**, finding the threshold that maximizes melanoma F1 while maintaining precision ≥ 0.3.

### 6. Augmentation Improvements

- Replaced deprecated `ShiftScaleRotate` → `Affine` (Albumentations 2.x compatible)
- Added `CoarseDropout` for hair artifact robustness (p=0.2)

### 7. PAD-UFES-20 Cross-Domain Support

Added `load_pad_ufes20_dataset()` with HAM10000 class mapping:

```
MEL → mel   NEV → nv   BCC → bcc
ACK → akiec   SCC → akiec   SEK → bkl
```

Auto-detection: `load_dataset()` detects dataset type from directory structure — no manual specification needed.

### 8. Aggressive Domain-Invariant Color Augmentation (Strategy 1)

Heavy color/lighting augmentations that preserve lesion morphology but vary skin tone and dermatoscope characteristics:
- CLAHE (local contrast normalization, p=0.3)
- RandomGamma (γ ∈ [60, 140], p=0.3)
- Heavy HSV jitter (±30 hue, ±40 saturation, ±20 value, p=0.6)
- RGBShift (±20 per channel, p=0.3)
- ChannelShuffle (p=0.2)
- Solarize (p=0.1)

Forces the encoder to learn shape/texture features invariant to skin tone and imaging equipment.

### 9. Test-Time Augmentation — TTA (Strategy 2)

Flag: `--tta` on `evaluate.py`. Averages predictions over 8 augmented views (original, h-flip, v-flip, 90° rotate, brightness shift, gamma shift, double flip, HSV shift). No retraining needed — just faster/more robust inference.

### 10. Domain Adversarial Training — DANN (Strategy 3)

New module: `src/models/domain_adversarial.py`. Gradient Reversal Layer + domain classifier predicts HAM10000 vs PAD-UFES-20 from encoder features. The encoder is forced to _hide_ domain identity, producing dataset-agnostic features.

Enabled with `--pad-data data/pad_ufes20`. Uses PAD-UFES-20 images **without labels** — evaluation protocol remains valid.

```
L_total += λ_domain × L_domain_classifier
```

### 11. CORAL + MMD Feature Alignment (Strategy 4)

New module: `src/losses/alignment_loss.py`. Minimizes the Frobenius norm between source and target feature covariance matrices (CORAL) and/or Maximum Mean Discrepancy (MMD). Simpler than DANN, no adversarial instability.

```
L_total += λ_coral × ||C_source - C_target||²_F
```

### 12. MixUp Training (Strategy 5)

Convex combinations of image pairs during training. Creates synthetic intermediate examples that bridge domain gaps. Controlled by `mixup.alpha` (0.2 default — small mixing for medical images).

```python
x_mix = λ·x₁ + (1−λ)·x₂    # λ ~ Beta(α, α)
y_mix = λ·y₁ + (1−λ)·y₂
```

---

## Domain Generalization — Implemented Strategies

Five strategies from the domain generalization roadmap have been implemented. Strategies 6-7 remain as future work.

| # | Strategy | Status | Config Key | Est. Gain |
|---|----------|--------|------------|-----------|
| 1 | **Aggressive color augmentation** (CLAHE, RandomGamma, heavy HSV, RGBShift, ChannelShuffle, Solarize) | ✅ Implemented | `augmentation.domain_aug` | +15–20% |
| 2 | **Test-Time Augmentation** (8-view prediction averaging) | ✅ Implemented | `evaluation.tta` / `--tta` | +3–5% |
| 3 | **Domain Adversarial Training (DANN)** (gradient reversal + domain classifier) | ✅ Implemented | `domain.enabled` / `--pad-data` | +10–15% |
| 4 | **CORAL + MMD feature alignment** (covariance/moment matching) | ✅ Implemented | `loss.lambda_coral` | +5–10% |
| 5 | **MixUp training** (convex image pair blending) | ✅ Implemented | `mixup.alpha` / `--disable-mixup` | +5–8% |
| 6 | Self-supervised pretraining on PAD-UFES-20 | Future | — | +10–20% |
| 7 | Fitzpatrick skin-type conditioning | Future | — | +5–8% |

**Cumulative estimate** (Strategies 1–5): 23.5% → **55–65%** cross-domain accuracy.

### How DANN + CORAL use PAD-UFES-20

Strategies 3 and 4 use PAD-UFES-20 images during HAM10000 training — but **only the images, never the labels**. The domain classifier learns to distinguish HAM10000 from PAD-UFES-20, and the encoder learns to hide that information. This is standard unsupervised domain adaptation and does not invalidate cross-domain evaluation.

### Training with domain adaptation

```bash
# Full training with all strategies (DANN + CORAL + MixUp + heavy aug)
python src/train.py \
  --data data/ham10000 \
  --pad-data data/pad_ufes20 \
  --epochs 60 \
  --img-size 320 \
  --batch-size 16

# Train without domain adaptation (baseline)
python src/train.py \
  --data data/ham10000 \
  --epochs 60 \
  --img-size 320 \
  --batch-size 16

# Disable specific strategies
python src/train.py --data data/ham10000 --pad-data data/pad_ufes20 \
  --disable-domain   # turn off DANN + CORAL
  --disable-mixup    # turn off MixUp
```

---

## Project Structure

```
CBM-Neuro-approach/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml               # All hyperparameters
├── data/
│   ├── ham10000/                   # HAM10000 dataset (manual download)
│   ├── pad_ufes20/                 # PAD-UFES-20 dataset (manual download)
│   └── abcd_cache/                 # Precomputed ABCD targets
├── scripts/
│   ├── download_ham10000.py
│   ├── download_pad_ufes20.py
│   ├── download_models.py          # Pull checkpoints from HF Hub
│   ├── optimize_threshold.py       # Melanoma threshold + temperature scaling
│   └── precompute_abcd_targets.py
├── src/
│   ├── models/
│   │   ├── encoder.py              # EfficientNetV2-S
│   │   ├── concept_heads.py        # ABCD concept heads
│   │   ├── attention.py            # Concept-guided attention
│   │   ├── diagnosis_head.py       # Multi-task output head
│   │   ├── cbm_model.py            # Full CBM model
│   │   └── domain_adversarial.py   # DANN: Gradient Reversal + Domain Classifier (Strategy 3)
│   ├── losses/
│   │   ├── concept_loss.py
│   │   ├── constraint_loss.py      # 3 clinical constraint rules
│   │   ├── multitask_loss.py       # L_total = L_diag + λ_concept·L_concept + λ_constraint·L_constraint
│   │   └── alignment_loss.py       # CORAL + MMD feature alignment (Strategy 4)
│   ├── data/
│   │   ├── abcd_targets.py         # ABCD pseudo concept labeler (SSIM, isoperimetric ratio, HSV entropy)
│   │   ├── datasets.py             # HAM10000 / PAD-UFES-20 dataset classes + auto-detection
│   │   └── transforms.py           # Dermoscopy-safe data augmentation
│   ├── train.py                    # Training with class weights, weighted sampler, macro F1 early stop
│   ├── evaluate.py                 # Full metric report + cross-domain evaluation
│   └── utils/
│       ├── metrics.py              # ECE, AUC, per-class P/R/F1, confusion matrix
│       ├── temperature_scaling.py  # Temperature calibration via LBFGS
│       └── viz.py                  # Attention heatmap + concept chart visualization
├── app/
│   └── demo_app.py                 # Gradio web app (auto-downloads model from HF Hub)
├── tests/
│   ├── test_abcd_targets.py        # ABCD labeler unit tests
│   ├── test_model_shapes.py        # Shape + gradient flow validation (hard bottleneck)
│   └── test_constraints.py         # Clinical constraint logic tests
```

---

## Key Features

- **Hard Concept Bottleneck**: Diagnosis head only sees (attended features + concept vector), never raw global features. Verified by gradient flow test.
- **Multi-Task Loss**: `L_total = L_diagnosis + λ_concept·L_concept + λ_constraint·L_constraint + λ_domain·L_domain + λ_coral·L_coral`
- **Clinical Constraints**: Three soft penalty rules
  - Rule 1: High ABCD scores → should raise malignancy probability
  - Rule 2: Large lesion (D ≥ 6mm) → should raise malignancy probability
  - Rule 3: All concepts low but high malignancy → penalize inconsistency
- **Concept-Guided Attention**: ABCD concepts steer spatial attention to clinically relevant regions
- **Pseudo Clinical Concepts**: ABCD targets auto-generated via classical CV — no manual dermatologist annotation needed
- **Class Imbalance Handling**: Inverse-frequency class weights + WeightedRandomSampler from training split only
- **Temperature Calibration**: LBFGS-optimized temperature scaling for clinically meaningful risk scores
- **Melanoma Threshold Optimization**: Validation-set-only threshold search maximizing melanoma F1
- **Domain-Invariant Augmentation** (Strategy 1): Heavy color/lighting transforms preserve morphology while varying skin tone
- **Test-Time Augmentation** (Strategy 2): `--tta` flag averages 8 augmented views at inference
- **Domain Adversarial Training** (Strategy 3): DANN with gradient reversal — encoder hides domain identity
- **CORAL Feature Alignment** (Strategy 4): Minimizes covariance gap between HAM10000 and PAD-UFES-20 features
- **MixUp** (Strategy 5): Synthetic convex combinations bridge domain gap
- **Cross-Domain Evaluation**: Train on HAM10000 (European skin tones), test on PAD-UFES-20 (Brazilian, diverse skin tones)

---

## Evaluation Metrics

The evaluation script (`src/evaluate.py`) produces:

- In-domain accuracy (HAM10000)
- Cross-domain accuracy (PAD-UFES-20)
- Per-class precision, recall, F1 (all 7 classes)
- Confusion matrix
- Concept prediction MAE (per concept + overall)
- ECE (Expected Calibration Error) for Malignancy Risk Score
- AUC (ROC Area Under Curve) for binary malignant/benign classification
- Constraint violation rates (Rule 1, Rule 2, Rule 3)

The threshold optimization script (`scripts/optimize_threshold.py`) additionally produces:

- Optimal melanoma threshold (maximizing melanoma F1 on validation set)
- Precision/recall/F1 at each threshold
- Temperature scaling factor
- Brier score before/after calibration

---

## Configuration

All hyperparameters are in `configs/default.yaml`:

| Section | Key parameters |
|---------|---------------|
| Data | `img_size`, `batch_size`, `num_workers`, `train_split`, `val_split` |
| Augmentation | `domain_aug.enabled`, `hue_shift_limit`, `clahe_prob`, `rgb_shift_prob`, `solarize_prob` |
| Model | `backbone`, `concept_dim`, `num_classes`, `malignant_classes`, `dropout_rate` |
| Training | `epochs`, `backbone_lr`, `head_lr`, `weight_decay`, `early_stop_patience` |
| Loss | `lambda_concept`, `lambda_constraint`, `lambda_domain`, `lambda_coral` |
| Domain | `enabled`, `pad_data_dir`, `domain_batch_size`, `domain_classifier_hidden` |
| MixUp | `enabled`, `alpha` |
| Constraints | `concept_high`, `concept_low`, `diameter_mm_threshold`, `alpha1`, `alpha2`, `alpha3` |
| Evaluation | `tta.enabled`, `tta.num_augmentations`, `ece_n_bins` |
| Risk | `thresholds` (LOW/MODERATE/HIGH/VERY HIGH) |

Override via CLI:

```bash
# Full domain-adaptive training
python src/train.py \
  --data data/ham10000 \
  --pad-data data/pad_ufes20 \
  --img-size 320 \
  --batch-size 16 \
  --epochs 60

# Baseline (no domain adaptation)
python src/train.py --data data/ham10000 --disable-domain --disable-mixup --epochs 60

# Evaluate with TTA
python src/evaluate.py --checkpoint models/best.pt --data data/pad_ufes20 --tta
```

---

## Citation

If you use this work, please cite:

```bibtex
@software{melanoma_cbm_2026,
  title = {Clinically Interpretable Melanoma Diagnosis and Malignancy Risk Assessment
           Using EfficientNetV2-S with Concept Bottleneck Learning and
           Clinical Constraint Optimization},
  year = {2026},
  note = {Research prototype. Not a diagnostic device.}
}
```
