# Histopathologic Cancer Detection - Kaggle Mini Project

## Problem Description

This project addresses the Kaggle competition "Histopathologic Cancer Detection".  
The goal is to build a binary image classification model to detect metastatic cancer in small histopathologic image patches.

Each image patch is a 96x96 pixel RGB image cropped from larger digital pathology scans.  
The model predicts whether a patch contains cancerous tissue (label = 1) or not (label = 0).

---

## Dataset

- **Training Images:** ~220,000 patches
- **Image Size:** 96x96 pixels, RGB format
- **Labels:** 
  - `0`: Non-cancerous
  - `1`: Cancerous
- **Files:**
  - `train_labels.csv`: Contains `id` and `label`
  - `sample_submission.csv`: Template for Kaggle submissions

---

## Exploratory Data Analysis (EDA)

- Visualized label distribution: slight imbalance (~60% non-cancerous, ~40% cancerous).
- Displayed random examples of cancerous vs non-cancerous patches.
- Checked for missing IDs, duplicates, corrupted images — **no major cleaning needed**.

---

## Modeling Approach

### 1. Baseline CNN
- Simple 4-layer convolutional network.
- Trained from scratch as a baseline.

### 2. Transfer Learning - ResNet50
- Used pretrained **ResNet50** from ImageNet.
- Frozen convolutional base; added custom dense classification head.
- Fine-tuned learning rate, dropout, and optimizer.

---

## 🎛Hyperparameter Tuning

- **Learning Rate:** 1e-3, 1e-4, 1e-5
- **Dropout Rate:** 0.3–0.5
- **Optimizers:** Adam, RMSProp
- **Batch Sizes:** 16, 32, 64

Final best settings:
- Learning Rate: 1e-4
- Dropout: 0.5
- Optimizer: Adam
- Batch Size: 32

---

## Results

| Model         | Validation Accuracy | Validation AUC | Notes                      |
|:--------------|:---------------------|:--------------|:---------------------------|
| Baseline CNN  | ~78%                  | ~0.83         | Easy to overfit             |
| ResNet50      | ~86%                  | ~0.90         | Best generalization         |

---

## Conclusion

- Transfer learning significantly improved performance over baseline CNN.
- Data augmentation helped generalize better.
- Early stopping and learning rate scheduling stabilized training.
- Future improvements:
  - Fine-tuning ResNet top layers
  - Experimenting with EfficientNet
  - Trying CutMix or MixUp augmentations
  - Using Focal Loss for better handling class imbalance

---

## How to Run

1. Install requirements:
   ```bash
   pip install tensorflow pandas numpy matplotlib seaborn
