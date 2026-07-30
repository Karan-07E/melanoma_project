# Model Used in the Melanoma Detection Project

## Short Answer

For this project, we used a Concept Bottleneck Model (CBM) built on a pretrained EfficientNetV2-RW-S image encoder. The model was trained on the HAM10000 dermoscopic image dataset to classify seven skin lesion categories and estimate malignancy risk. This model is suitable for our project because it combines strong image-feature extraction with clinically interpretable ABCD concepts: asymmetry, border irregularity, color variation, and lesion size.

## Project Goal

The goal of this project is to classify dermoscopic skin lesion images and support melanoma risk assessment in a way that is not only accurate, but also explainable. In medical image analysis, explainability matters because a model should not behave like a black box. Our model therefore predicts both the final disease class and intermediate clinical concepts that are commonly used in dermatology.

This project is for research and educational use only. It is not a diagnostic medical device.

## Dataset Used

The main dataset used for training and evaluation was HAM10000. It contains dermoscopic images from seven diagnostic classes:

- mel: melanoma
- nv: melanocytic nevus
- bcc: basal cell carcinoma
- akiec: actinic keratosis / intraepithelial carcinoma
- bkl: benign keratosis-like lesions
- df: dermatofibroma
- vasc: vascular lesions

The project uses a 70 percent training split, 15 percent validation split, and 15 percent test split. The input image size is 224 x 224 RGB.

## Model Architecture

The model used is:

**EfficientNetV2-RW-S + Concept Bottleneck Model + Concept-Guided Attention**

The pipeline is:

1. A dermoscopic image is passed into a pretrained EfficientNetV2-RW-S encoder.
2. The encoder extracts deep visual features from the image.
3. Four concept heads predict ABCD clinical concepts:
   - Asymmetry
   - Border irregularity
   - Color variation
   - Normalized lesion area / diameter-related score
4. The concept vector guides a spatial attention module.
5. The diagnosis head predicts:
   - Seven-class lesion diagnosis
   - Malignancy risk score from 0 to 1

The model configuration used:

| Component | Value |
|---|---|
| Backbone | EfficientNetV2-RW-S |
| Pretrained | Yes |
| Input size | 224 x 224 |
| Feature dimension | 1280 |
| Concept dimension | 4 |
| Number of classes | 7 |
| Attention mode | Sigmoid |
| Dropout | 0.3 |
| Training epochs configured | 60 |
| Backbone learning rate | 0.00001 |
| Head learning rate | 0.001 |
| Weight decay | 0.0001 |

## Why This Model Was Chosen

### 1. EfficientNetV2-RW-S is strong for image classification

EfficientNetV2 is designed to extract high-quality image features efficiently. Dermoscopic lesion classification depends on subtle visual patterns such as color distribution, shape, texture, and borders. A pretrained EfficientNetV2-RW-S backbone gives the project a strong visual feature extractor without needing to train a large CNN from scratch.

### 2. Transfer learning is suitable for HAM10000

HAM10000 is useful, but it is still relatively small compared with very large computer vision datasets. Using a pretrained encoder helps the model start with general image understanding and then fine-tune those features for skin lesion classification. This usually improves stability and performance when medical data is limited.

### 3. The Concept Bottleneck improves explainability

A normal CNN only gives a final prediction, which can be difficult to justify. Our CBM also predicts clinical concepts before the final diagnosis. This is important because the model can explain predictions using ABCD-style features:

- Is the lesion asymmetric?
- Are the borders irregular?
- Is there color variation?
- Is the lesion large or visually extensive?

This makes the model more appropriate for a medical research project than a purely black-box classifier.

### 4. Concept-guided attention helps focus on relevant lesion regions

The model uses the predicted concepts to guide spatial attention over image feature maps. This helps connect clinical reasoning with visual localization. Instead of only learning abstract image features, the model uses ABCD concept information to influence where it focuses in the lesion image.

### 5. Multi-task learning improves usefulness

The model predicts multiple outputs:

- Disease class
- Malignancy risk score
- ABCD concept scores

This is better for the project than a single-output classifier because melanoma screening is not only about class names. Risk estimation and interpretable clinical factors are also important.

### 6. Clinical constraint loss encourages medically consistent predictions

The training objective includes classification loss, concept loss, and clinical constraint loss. The constraint loss encourages consistency between high-risk clinical concepts and malignancy risk. For example, if ABCD concepts are high, the model should generally assign higher malignancy risk.

## Training Summary

The model was trained using HAM10000 with precomputed ABCD concept targets generated from lesion images and segmentation masks. The configured training length was 60 epochs.

During training, the best validation checkpoint was found at epoch 20. The earlier training run stopped at epoch 35 because early stopping was enabled with patience 15. After disabling early stopping, training was continued until epoch 60.

Because of CUDA/cuDNN instability near the end of training, the final epochs were completed using stability workarounds. The validated best checkpoint remains the most reliable checkpoint for reporting validation performance.

## Evaluation Results

Two checkpoints are available:

| Checkpoint | Description |
|---|---|
| `models/best.pt` | Best validated checkpoint, selected by validation loss |
| `models/latest.pt` | Latest checkpoint after continuing to epoch 60 |

### Best Validated Checkpoint: `models/best.pt`

| Metric | Result |
|---|---:|
| Validation-best epoch | 20 |
| Validation loss | 0.4712 |
| Validation accuracy | 85.66% |
| Test accuracy | 84.36% |
| Macro F1 | 71.63% |
| Weighted F1 | 84.16% |
| Risk AUC | 74.67% |
| Concept MAE | 0.0771 |

### Latest Epoch-60 Checkpoint: `models/latest.pt`

| Metric | Result |
|---|---:|
| Final epoch | 60 |
| Test accuracy | 86.41% |
| Macro F1 | 77.06% |
| Weighted F1 | 86.31% |
| Risk AUC | 69.39% |
| Concept MAE | 0.0844 |

The epoch-60 checkpoint achieved higher classification accuracy and F1 score, while the best validated checkpoint had better risk AUC and concept prediction error. For formal reporting, `models/best.pt` should be presented as the main checkpoint because it was selected by validation performance. The epoch-60 checkpoint can be mentioned as an additional completed training run.

## Why It Is Best for This Project

This model is the best fit for our project because the project is not only about image classification; it is about clinically interpretable melanoma risk assessment. A simple CNN could classify images, but it would not provide a clear clinical explanation. Our CBM uses EfficientNetV2-RW-S for strong visual learning and adds ABCD concepts for interpretability.

The model is therefore suitable because it balances:

- Accuracy: strong classification performance on HAM10000
- Explainability: predicts ABCD clinical concepts
- Medical relevance: uses concepts familiar in skin lesion assessment
- Risk assessment: outputs malignancy risk, not only class labels
- Practicality: pretrained EfficientNetV2-RW-S is efficient enough for local training and inference

## Suggested Answer if Someone Asks

We used a Concept Bottleneck Model with a pretrained EfficientNetV2-RW-S backbone. EfficientNetV2-RW-S extracts strong visual features from dermoscopic images, and the concept bottleneck makes the model interpretable by predicting ABCD clinical concepts before the final diagnosis. This is better for our melanoma project than a normal black-box CNN because it gives both classification results and clinically meaningful explanations, such as asymmetry, border irregularity, color variation, and lesion size. The model was trained on HAM10000 for seven-class skin lesion classification and malignancy risk prediction.

## Limitations

This model is a research prototype and should not be used for real clinical diagnosis. The ABCD labels are pseudo-labels generated from image processing and segmentation masks, not direct dermatologist annotations. The model should be validated on more diverse external datasets before any real-world medical use.

