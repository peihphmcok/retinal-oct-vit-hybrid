# Hybrid Vision Transformer and Gradient Boosting for Retinal OCT Classification

Official research repository for the Master in Machine Learning and Data Mining (MLDM) — Semester 1 Research Methodology, Université Jean Monnet (UJM).

---

## Authors
- **Jayden Tan**
- **Yuhsaku Nakai**
- **Phuong Dang Cao**
- **Tuan Hiep Pham**

---

## Abstract & Motivation

Optical Coherence Tomography (OCT) is the non-invasive clinical imaging gold standard for diagnosing sight-threatening macular pathologies, including Choroidal Neovascularization (CNV / Wet AMD), Diabetic Macular Edema (DME), and Drusen (Dry AMD). Existing automated diagnostic paradigms suffer from a dichotomy:
- **Full End-to-End Deep Learning (CNNs & Vision Transformers)**: Yields high diagnostic accuracy (>96–98%) but requires substantial GPU hardware, long multi-epoch training regimes, and functions largely as a clinical black box with high risk of overfitting on imbalanced cohorts.
- **Traditional Machine Learning (e.g., HOG/LBP + SVM)**: Highly compute-efficient, but lacks the representational power required to resolve subtle morphological biomarkers (<80–90% accuracy).

To bridge this gap, this project develops a **lightweight, clinically interpretable hybrid framework** that couples the representational strength of self-supervised Vision Transformers with the efficiency, tabular interpretability, and robust regularization of Gradient Boosted Decision Trees (XGBoost).

---

## Methodology & Architecture

The framework operates via a modular, leak-free pipeline:

```text
Raw Retinal OCT Scan (84,484 Images)
   │
   ▼
[Step 1] Contrast-Limited Adaptive Histogram Equalization (CLAHE)
   │     Enhances local layer contrast and reduces speckle noise
   ▼
[Step 2] Frozen Vision Transformer (ViT-Base/16)
   │     Single forward pass extraction -> 768-dimensional latent vector per scan
   ▼
[Step 3] Leak-Free 5-Fold Stratified Cross-Validation Partitioning
   │     Strict partition isolation to prevent feature or distribution leakage
   ▼
[Step 4] Orthogonal PCA Dimensionality Compression (k = 150)
   │     Captures 88.2% of latent variance while reducing dimensionality by 80.5%
   ▼
[Step 5] Latent-Space SMOTE Class Balancing
   │     Synthesizes minority class representations strictly within training folds
   ▼
[Step 6] Regularized Gradient Tree Boosting (XGBoost)
   │     Multi-class classification with validation early-stopping
   ▼
[Step 7] Dual Explainability & Attribution
         Global: TreeSHAP feature importance mapped to retinal biomarkers
         Local:  14x14 ViT Attention Rollout overlays on morphological lesions
```

---

## Dataset

Evaluated on the standardized **Mendeley OCT2017 Benchmark** (Kermany et al., 2018):

| Class | Clinical Pathology | Raw Training Set | Benchmark Test Set |
| :--- | :--- | :---: | :---: |
| **0: CNV** | Choroidal Neovascularization (Wet AMD) | 37,205 (44.57%) | 250 (25.00%) |
| **1: DME** | Diabetic Macular Edema (Intraretinal Fluid) | 11,348 (13.59%) | 250 (25.00%) |
| **2: DRUSEN** | Sub-RPE Drusen Deposits (Dry AMD) | 8,616 (10.32%) | 250 (25.00%) |
| **3: NORMAL** | Healthy Macular Contour | 26,315 (31.52%) | 250 (25.00%) |
| **Total** | **All 4 Phenotypes** | **83,484 (100%)** | **1,000 (100%)** |

*Note: The official 1,000-image test set remains strictly isolated and untouched during all preprocessing, PCA fitting, and SMOTE balancing to guarantee unbiased clinical evaluation.*

---

## Key Experimental Results

### 1. Diagnostic Performance (M5 Proposed Model)
Evaluated across 5-fold cross-validation and the untouched 1,000-image test benchmark:

| Metric | 5-Fold CV (Mean ± SD) | Benchmark Test Performance | 95% Confidence Interval |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 88.53% ± 0.31% | **97.00%** | 95.90% – 98.00% |
| **Macro F1-Score** | - | **97.00%** | - |
| **Macro Sensitivity (Recall)** | - | **97.00%** | - |
| **Macro Specificity** | - | **99.00%** | - |
| **Macro OVR ROC-AUC** | - | **99.88%** | 99.79% – 99.98% |

#### Per-Class Diagnostic Metrics
| Class | Precision (%) | Recall / Sensitivity (%) | Specificity (%) | F1-Score (%) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CNV** | 94.64 | 98.80 | 98.13 | **96.67** | 0.9991 |
| **DME** | 98.75 | 94.80 | 99.60 | **96.73** | 0.9975 |
| **Drusen** | 97.57 | 96.40 | 99.20 | **96.98** | 0.9992 |
| **Normal** | 97.22 | 98.00 | 99.07 | **97.61** | 0.9993 |

### 2. Ablation Study (M0 to M5)
Validating each architectural component on the test benchmark:

| Model ID | Experimental Configuration | Test Accuracy (%) | Macro F1 (%) | Fit Time (s) |
| :--- | :--- | :---: | :---: | :---: |
| **M0** | No CLAHE + SMOTE + XGBoost | 97.00 | 97.00 | 39.60 |
| **M1** | No SMOTE + XGBoost | 95.40 | 95.41 | 29.90 |
| **M2** | SMOTE + RBF-SVM | 95.40 | 95.41 | 68.97 |
| **M3** | SMOTE + Random Forest | 95.10 | 95.11 | 64.62 |
| **M4** | SMOTE + LightGBM | 96.70 | 96.70 | 24.72 |
| **M5 (Ours)** | **CLAHE + ViT + PCA (150) + SMOTE + Tuned XGBoost** | **97.00** | **97.00** | 41.39 |

> **Key takeaway**: SMOTE provides a crucial **+1.60%** boost in balanced diagnostic accuracy (M1: 95.40% → M5: 97.00%), resolving minority misclassifications in DME and Drusen.

### 3. Computational Efficiency & Latency
| Architecture Paradigm | Trainable DL Parameters | Training Hardware | Classifier Training Time | Per-Scan Inference Latency |
| :--- | :---: | :---: | :---: | :---: |
| InceptionV3 (Kermany et al., 2018) | ~23.8M | High-VRAM GPU | Multi-epoch (hours) | ~18.5 ms |
| Pure ViT-Base (Akca et al., 2024) | ~86.6M | High-VRAM GPU | Multi-epoch (hours) | ~24.0 ms |
| 18-CNN Ensemble + SVM (Barua et al., 2021) | 0 (180 passes) | Multi-GPU + CPU | Very High | >150 ms |
| **Proposed Hybrid ViT-XGBoost (M5)** | **0 (1 frozen pass)** | **Standard CPU** | **41.39 s** | **0.019 ms** |

---

## Repository Structure

```text
retinal-oct-vit-hybrid/
├── configs/
│   └── config.yaml                     # Global configurations, hyperparameters, paths
├── src/
│   ├── data/
│   │   └── clahe_preprocessor.py       # CLAHE contrast enhancement & image loader
│   ├── models/
│   │   ├── vit_extractor.py            # Pretrained ViT-Base/16 feature extractor
│   │   ├── dim_reduction.py            # Orthogonal PCA compression module
│   │   ├── balancer.py                 # Latent-space SMOTE balancer
│   │   └── classifiers.py              # XGBoost, LightGBM, SVM, RF wrappers
│   ├── utils/
│   │   ├── seed.py                     # Deterministic random seed controller
│   │   └── metrics.py                  # Multi-class metrics and bootstrap CI computation
│   └── pipeline/
│       ├── step1_extract_features.py   # GPU feature extraction (CLAHE & Raw)
│       ├── step2_train_evaluate.py     # PCA sweep, 5-Fold CV & benchmark training
│       └── step3_ablation_and_figures.py # Full M0-M5 ablation & 6 IEEE figures/tables
├── outputs/
│   ├── tables/                         # Generated IEEE-formatted CSV tables (Table I - VIII)
│   └── figures/                        # Generated 300 DPI publication figures (Figure 1 - 6)
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Git exclusions (data, large arrays, caches)
└── README.md                           # Documentation
```

---

## Getting Started

### 1. Environment Setup

Python 3.10+ is recommended.

```bash
# Clone the repository
git clone https://github.com/<your-username>/retinal-oct-vit-hybrid.git
cd retinal-oct-vit-hybrid

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# .\.venv\Scripts\Activate.ps1  # On Windows PowerShell

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

*For GPU acceleration in PyTorch (NVIDIA CUDA 12.1):*
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Pipeline Execution

The experiment is structured into three clean, modular steps:

#### **Step 1: Feature Extraction** *(Run once on GPU or Google Colab)*
Extracts 768-D ViT embeddings for all 84,484 scans with and without CLAHE:
```bash
python -m src.pipeline.step1_extract_features --mode both
```
*Outputs: Saved in `outputs/features/`.*

#### **Step 2: PCA Sweep & 5-Fold Cross-Validation** *(Runs on CPU, ~2 min)*
Performs PCA trade-off analysis ($k \in \{64, 150, 256, 768\}$) and runs 5-Fold Stratified CV:
```bash
python -m src.pipeline.step2_train_evaluate
```
*Outputs: `outputs/tables/table1_dataset_distribution.csv`, `outputs/tables/table2_pca_tradeoff.csv`, and benchmark summary.*

#### **Step 3: Ablation Study & Publication Figures** *(Runs on CPU, ~2-3 min)*
Executes full M0–M5 ablation, clinical misclassification analysis, TreeSHAP global attribution, and exports all 6 IEEE-compliant figures and tables:
```bash
python -m src.pipeline.step3_ablation_and_figures
```
*Outputs: All 8 CSV tables in `outputs/tables/` and 6 figures (300 DPI PNG) in `outputs/figures/`.*

---

## References

1. **D. S. Kermany, M. Goldbaum, W. Cai, et al.**, "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning," *Cell*, vol. 172, no. 5, pp. 1122–1131, 2018.
2. **A. Dosovitskiy, L. Beyer, A. Kolesnikov, et al.**, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.
3. **T. Chen and C. Guestrin**, "XGBoost: A Scalable Tree Boosting System," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.*, 2016, pp. 785–794.
4. **S. M. Lundberg, G. Erion, H. Chen, et al.**, "From Local Explanations to Global Understanding with Explainable AI for Trees," *Nature Machine Intelligence*, vol. 2, no. 1, pp. 56–67, 2020.
5. **N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer**, "SMOTE: Synthetic Minority Over-sampling Technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002.
6. **P. D. Barua, O. Faust, N. A. Salim, et al.**, "Multilevel Deep Feature Generation Framework for Automated Detection of Retinal Abnormalities Using OCT Images," *Entropy*, vol. 23, no. 12, p. 1651, 2021.
7. **M. Opoku, E. A. Frimpong, C. Weyori, et al.**, "CLAHE-CapsNet: Efficient Retina OCT Classification Using Capsule Networks with Contrast Limited Adaptive Histogram Equalization," *Informatics in Medicine Unlocked*, vol. 42, p. 101372, 2023.
8. **E. Akca, S. Bayraktar, and K. Ozturk**, "Classification of Retinal Diseases from OCT Images Using Fine-Tuned Vision Transformers," *Diagnostics*, vol. 14, no. 3, p. 289, 2024.
9. **K. A. Umer, M. U. G. Khan, M. F. Tariq, et al.**, "A Multi-Scale Feature Fusion Framework for Multi-Class Retinal OCT Classification," *IEEE Access*, vol. 11, pp. 48312–48325, 2023.
10. **S. G. Gidijala, P. K. Mallik, et al.**, "Deep Transfer Learning for Automated Retinal Disease Detection in Optical Coherence Tomography," *Journal of Medical Imaging and Health Informatics*, 2026.
