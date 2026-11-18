# Hybrid Vision Transformer and Gradient Boosting for Retinal OCT Classification

Official research repository for the MLDM Master Program (Semester 1 - Research Methodology), Jean Monnet University (UJM).

---

## Authors
- **Jayden Tan**
- **Yuhsaku Nakai**
- **Phuong Dang Cao**
- **Tuan Hiep Pham**

---

## Abstract & Research Motivation

Optical Coherence Tomography (OCT) is the non-invasive clinical gold standard for diagnosing sight-threatening macular diseases. Automatic classification on large-scale cohorts typically faces a critical trade-off:
- **Traditional Machine Learning (e.g., HOG/LBP + SVM)** requires minimal compute but suffers from low representational capacity (<80% accuracy).
- **End-to-End Deep Learning & Vision Transformers** achieve high accuracy (>96-98%) but demand days of multi-epoch GPU backpropagation, risk overfitting, and lack tabular interpretability.

To resolve this limitation, we present a lightweight hybrid framework:
1. **Single-Pass Feature Extraction**: Raw OCT B-scans are contrast-enhanced via CLAHE and passed once through a frozen Vision Transformer (`vit_base_patch16_224`) to extract compact 768-dimensional latent representations.
2. **Leak-Free Vector Pipeline**: Enforces strict 5-Fold Stratified Cross-Validation. Dimensionality reduction (PCA to 150-D) and class balancing (Latent-Space SMOTE) are fitted exclusively on internal training partitions to guarantee zero data leakage.
3. **Tuned Gradient Boosting**: Fast CPU training using Regularized XGBoost with early stopping, evaluated on the untouched 1,000-image official test benchmark.

---

## Dataset Architecture

We benchmark on the standardized Mendeley OCT2017 dataset (Kermany et al., 2018; DOI: 10.17632/rscbjbr9sj.1):

| Class Label | Clinical Pathology | Raw Training Set | Official Benchmark Test Set |
| :--- | :--- | :---: | :---: |
| **0: CNV** | Choroidal Neovascularization (Wet AMD) | 37,205 (44.57%) | 250 (25.00%) |
| **1: DME** | Diabetic Macular Edema (Intraretinal Fluid) | 11,348 (13.59%) | 250 (25.00%) |
| **2: DRUSEN** | Sub-RPE Drusen Deposits (Dry AMD) | 8,616 (10.32%) | 250 (25.00%) |
| **3: NORMAL** | Healthy Macular Contour | 26,315 (31.52%) | 250 (25.00%) |
| **TOTAL** | **All 4 Phenotypes** | **83,484 (100%)** | **1,000 (100%)** |

*Note: The official 1,000-image test set remains 100% untouched during all preprocessing, PCA fitting, and SMOTE balancing to guarantee rigorous clinical evaluation.*

---

## Methodological Pipeline

```text
Raw OCT Scan (84,484 images)
   │
   ▼
[Step 1] CLAHE Contrast Enhancement & Resize (224x224, 3-channel normalized)
   │
   ▼
[Step 2] Frozen ViT-Base Backbone (Single forward pass -> 768-D vectors)
   │
   ▼
[Step 3] Leak-Free 5-Fold Stratified Cross-Validation Partitioning
   │
   ▼
[Step 4] Orthogonal PCA Compression (150 Principal Components, fitted on train fold only)
   │
   ▼
[Step 5] Latent-Space SMOTE Balancing (Applied strictly on 150-D train fold vectors)
   │
   ▼
[Step 6] Tuned XGBoost Multi-Class Classification (Early stopping on validation fold)
   │
   ▼
Official Test Benchmark Evaluation (1,000 untouched images)
```

---

## Experimental Benchmark Results

### 1. 5-Fold Stratified Cross-Validation (Our Framework)

Evaluated across all 5 folds on the official 1,000 balanced benchmark images:

| Metric | Mean ± Std | Best Fold |
| :--- | :---: | :---: |
| **Validation Accuracy** | 88.50% ± 0.41% | 89.09% |
| **Test Accuracy** | **97.16% ± 0.27%** | **97.50%** |
| **Macro Sensitivity (Recall)** | **97.16% ± 0.27%** | 97.50% |
| **Macro Specificity** | **99.05% ± 0.09%** | 99.17% |
| **Macro F1-Score** | **97.16% ± 0.27%** | 97.50% |

### 2. SOTA Benchmark Comparison on Mendeley OCT2017

| Study & Year | Methodological Category | Pipeline Architecture | Test Accuracy | Macro Specificity | Compute Profile |
| :--- | :--- | :--- | :---: | :---: | :---: |
| Wali & Sipani (2024) | Traditional ML | HOG + SVM | 78.80% | - | Low (CPU) |
| Gidijala et al. (2026) | Hybrid CNN + ML | EfficientNet + SVM | 90.76% | 90.80% | Low (1-pass) |
| Hassan et al. (2026) | Hybrid CNN + ML | ResNet-50 + Random Forest | 92.89% | 93.10% | Low (1-pass) |
| Fang et al. (2019) | End-to-End DL | Iterative Fusion CNN (IFCNN) | 93.25% | 97.60% | High (GPU) |
| Kermany et al. (2018) | Foundational DL | InceptionV3 Fine-Tuning | 96.60% | 97.40% | High (GPU) |
| Barua et al. (2021) | Heavy Ensemble | 18 CNNs (10k features) + SVM | 97.40% | 99.13% | Very High (180 passes) |
| **Proposed Framework** | **Lightweight ViT Hybrid** | **Frozen ViT + PCA + SMOTE + XGBoost** | **97.16%** | **99.05%** | **Fast CPU (~3-5 min)** |

---

## Repository Structure

```text
Research-Methodology/
├── configs/
│   └── config.yaml             # Experiment hyperparameters & dataset paths
├── src/
│   ├── data/
│   │   └── clahe_preprocessor.py # CLAHE enhancement & tensor normalization
│   ├── models/
│   │   ├── vit_extractor.py    # Pretrained ViT feature extractor
│   │   ├── dim_reduction.py    # PCA projection module
│   │   ├── balancer.py         # Latent-space SMOTE balancer
│   │   └── classifiers.py      # XGBoost and baseline models
│   ├── utils/
│   │   ├── seed.py             # Reproducibility seed setup
│   │   └── metrics.py          # Multiclass evaluation metrics
│   └── pipeline/
│       ├── step1_extract_features.py # Feature extraction pipeline
│       └── step2_train_evaluate.py   # 5-fold training & evaluation pipeline
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## Environment Setup

Requires **Python 3.10+**.

### Windows

```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Activate
.\.venv\Scripts\Activate.ps1
# (or in CMD: .\.venv\Scripts\activate.bat)

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **NVIDIA GPU Acceleration**:
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### macOS (Apple Silicon & Intel)

```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip setuptools wheel
pip install torch torchvision
pip install -r requirements.txt
```

---

## Execution Guide

### Step 1: Feature Extraction (Run once on GPU / Google Colab)
Passes all 84,484 scans through the frozen ViT backbone to save compact feature matrices:
```bash
python -m src.pipeline.step1_extract_features
```
*Outputs: `outputs/features/X_train_raw.npy` (~256 MB) and `X_test_raw.npy` (~3 MB).*

### Step 2: 5-Fold Cross-Validation & Benchmark Evaluation (CPU, ~3-5 min)
Executes leak-free 5-Fold Stratified Cross-Validation, PCA compression, SMOTE balancing, and XGBoost training:
```bash
python -m src.pipeline.step2_train_evaluate
```
*Outputs: Console summary, `outputs/benchmark_results.txt`, and `outputs/benchmark_results.json`.*

---

## Citation & References

- Kermany, D. S., et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. **Cell**, 172(5), 1122–1131.
- Barua, P. D., et al. (2021). *Multilevel Deep Feature Generation Framework for Automated Detection of Retinal Abnormalities Using OCT Images*. **Entropy**, 23(12), 1651.
- Opoku, M., et al. (2023). *CLAHE-CapsNet: Efficient Retina OCT Classification Using Capsule Networks with Contrast Limited Adaptive Histogram Equalization*. **PMC**, PMC10688733.
