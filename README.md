#  Cystic Hygroma Detection using Deep Learning

A deep learning-based binary classification system for detecting **Cystic Hygroma** from ultrasound images, using a hybrid ensemble of **ResNet-50** (transfer learned on BUSI dataset) and **DenseNet-169 with CBAM attention**, trained with **4-fold stratified cross-validation**.

---

##  Project Overview

Cystic hygroma is a rare congenital lymphatic malformation typically detected via ultrasound. This project automates its classification (cystic hygroma vs. normal) using a two-stage deep learning pipeline:

1. **Preprocessing** — ROI extraction + CLAHE enhancement on grayscale ultrasound images
2. **Transfer Learning** — ResNet-50 pre-trained on the BUSI (Breast Ultrasound Images) dataset
3. **Hybrid Ensemble** — Fine-tuned ResNet-50 + DenseNet-169 with CBAM attention module
4. **Evaluation** — Accuracy, Sensitivity, Specificity, Precision, F1 Score, AUC-ROC across 4 folds

---

##  Repository Structure

```
cystic-hygroma-detection/
│
├── cystichygroma.ipynb      # Main notebook (preprocessing → training → evaluation)
└── README.md
```

>  **Note:** Trained model weights, preprocessed images, and result outputs (confusion matrices, ROC curves, loss/accuracy plots) are stored in **Google Drive** due to file size constraints.

---

##  Model Architecture

### Stage 1 — BUSI Pre-training
- **ResNet-50** pre-trained on ImageNet, fine-tuned on the [BUSI Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) (benign / malignant / normal, 3-class)
- Saved weights used to initialize the cystic hygroma classifier

### Stage 2 — Hybrid Ensemble (Binary Classification)
| Component | Details |
|-----------|---------|
| ResNet-50 | Transfer weights from BUSI, FC replaced for 2-class output |
| DenseNet-169 | Pretrained backbone + CBAM attention on final feature maps |
| Fusion | Averaged softmax outputs from both models |
| Loss | Cross-Entropy |
| Optimizer | Adam (lr = 1e-4) |
| Training | 4-fold Stratified K-Fold, up to 1000 epochs, early stopping (patience=25) |

### CBAM Attention Module
Applies both **channel attention** and **spatial attention** on DenseNet-169 feature maps (1664 channels) to focus on diagnostically relevant regions.

---

##  Preprocessing Pipeline

Each ultrasound image goes through:
1. **Grayscale conversion**
2. **ROI extraction** via contour detection (removes black borders)
3. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
4. **Resize** to 256×256

---

##  Evaluation Metrics (per fold)

- Accuracy
- Sensitivity (Recall)
- Specificity
- Precision
- F1 Score
- AUC-ROC

Final results reported as **Mean ± Std** across all 4 folds.

**Outputs saved to Google Drive:**
- `results/cm_fold{n}.png` — Confusion matrices
- `results/loss_curve_fold{n}.png` — Train vs. Validation loss
- `results/accuracy_curve_fold{n}.png` — Train vs. Validation accuracy
- `results/roc_curve.png` — Combined ROC curve for all folds
- `results/metrics.txt` — Full metrics log

---

##  Requirements

```
torch
torchvision
opencv-python
numpy
matplotlib
seaborn
scikit-learn
tqdm
```

> This project was developed and run on **Google Colab** with GPU acceleration. Data is loaded from **Google Drive**.

---

##  How to Run

1. Open `cystichygroma.ipynb` in **Google Colab**
2. Mount your Google Drive and set up the folder structure:
   ```
   MyDrive/paperproject/
   ├── data/
   │   ├── cystic_hygroma/
   │   └── normal/
   ├── preprocessed/
   ├── busi/
   ├── models/
   └── results/
   ```
3. Upload your `kaggle.json` when prompted to download the BUSI dataset
4. Run all cells in order

---

##  Datasets

| Dataset | Source | Usage |
|--------|--------|-------|
| Cystic Hygroma Ultrasound | Custom / Private | Primary classification target |
| BUSI (Breast Ultrasound Images) | [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) | Pre-training ResNet-50 |

---

##  Author

**srilakshmir513-hue**  
Medical Image Analysis · Deep Learning · PyTorch
