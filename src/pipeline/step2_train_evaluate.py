import os
import sys
import time
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.dim_reduction import PCAReducer
from src.models.balancer import LatentSMOTEBalancer
from src.models.classifiers import get_xgboost_classifier
from src.utils.metrics import compute_classification_metrics
from src.utils.seed import set_seed

CLASS_DESCRIPTIONS = {
    0: ("0: CNV", "Choroidal Neovascularization (Wet AMD)"),
    1: ("1: DME", "Diabetic Macular Edema (Intraretinal Fluid)"),
    2: ("2: DRUSEN", "Sub-RPE Drusen Deposits (Dry AMD)"),
    3: ("3: NORMAL", "Healthy Macular Contour & Foveal Depression"),
}


def export_table1_dataset_distribution(y_train_raw: np.ndarray, y_tr: np.ndarray, y_val: np.ndarray, y_tr_bal: np.ndarray, y_test_raw: np.ndarray):
    """Export Table I: Mendeley OCT2017 dataset distribution and partitioning."""
    os.makedirs("outputs/tables", exist_ok=True)
    rows = []
    n_train_total = len(y_train_raw)
    n_bal_total = len(y_tr_bal)
    n_test_total = len(y_test_raw)

    for cls_idx in range(4):
        label, desc = CLASS_DESCRIPTIONS[cls_idx]
        raw_c = int(np.sum(y_train_raw == cls_idx))
        tr_c = int(np.sum(y_tr == cls_idx))
        val_c = int(np.sum(y_val == cls_idx))
        bal_c = int(np.sum(y_tr_bal == cls_idx))
        test_c = int(np.sum(y_test_raw == cls_idx))
        rows.append({
            "Class Label": label,
            "Clinical Pathology Description": desc,
            "Raw Training Set": f"{raw_c:,} ({raw_c / n_train_total * 100:.2f}%)",
            "Internal Train (85%)": f"{tr_c:,}",
            "Internal Val (15%)": f"{val_c:,}",
            "Internal Train After SMOTE": f"{bal_c:,} ({bal_c / n_bal_total * 100:.2f}%)",
            "Official Test Set": f"{test_c:,} ({test_c / n_test_total * 100:.2f}%)",
        })

    rows.append({
        "Class Label": "TOTAL",
        "Clinical Pathology Description": "All 4 Retinal Phenotypes",
        "Raw Training Set": f"{n_train_total:,} (100.00%)",
        "Internal Train (85%)": f"{len(y_tr):,}",
        "Internal Val (15%)": f"{len(y_val):,}",
        "Internal Train After SMOTE": f"{n_bal_total:,} (Balanced)",
        "Official Test Set": f"{n_test_total:,} (100.00%)",
    })

    df_t1 = pd.DataFrame(rows)
    out_path = "outputs/tables/table1_dataset_distribution.csv"
    df_t1.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return df_t1


def export_table2_pca_tradeoff(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, config: dict):
    """Export Table II: PCA dimensionality reduction and variance trade-off."""
    seed = config["training"]["random_seed"]
    k_neighbors = config["training"]["smote"]["k_neighbors"]
    total_dim = X_tr.shape[1]

    print("Running PCA sweep (k = 64, 150, 256, 768)...")
    sweep_configs = [
        ("PCA (k = 64)", 64, True),
        ("PCA (k = 150 - Selected)", 150, True),
        ("PCA (k = 256)", 256, True),
        ("Raw ViT Embeddings (k = 768, No PCA)", 768, False),
    ]

    rows = []
    for label, k, use_pca in sweep_configs:
        t0 = time.time()
        reducer = PCAReducer(n_components=k, use_pca=use_pca)
        X_tr_red, exp_var = reducer.fit_transform(X_tr)
        X_val_red = reducer.transform(X_val)
        X_test_red = reducer.transform(X_test)

        balancer = LatentSMOTEBalancer(k_neighbors=k_neighbors, random_state=seed)
        X_tr_bal, y_tr_bal = balancer.fit_resample(X_tr_red, y_tr)

        model = get_xgboost_classifier(config)
        model.fit(X_tr_bal, y_tr_bal, eval_set=[(X_val_red, y_val)], verbose=False)
        fit_time = round(time.time() - t0, 2)

        val_acc = float(np.mean(model.predict(X_val_red) == y_val) * 100)
        test_acc = float(np.mean(model.predict(X_test_red) == y_test) * 100)
        dim_reduction = (1.0 - (k / total_dim)) * 100.0

        rows.append({
            "Feature Space Dimension": label,
            "Cumulative Variance Retained (%)": round(exp_var * 100, 2),
            "Dimensionality Reduction (%)": round(dim_reduction, 2),
            "Val Acc (%)": round(val_acc, 2),
            "Test Acc on 1,000 Imgs (%)": round(test_acc, 2),
            "XGBoost CPU Fit Time (s)": fit_time,
        })
        print(f"  {label}: var={exp_var * 100:.2f}%, val={val_acc:.2f}%, test={test_acc:.2f}%, time={fit_time:.2f}s")

    df_t2 = pd.DataFrame(rows)
    out_path = "outputs/tables/table2_pca_tradeoff.csv"
    df_t2.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return df_t2


def train_and_benchmark(config_path: str = "configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config["training"]["random_seed"]
    set_seed(seed)

    paths = config["feature_extraction"]["export_paths"]
    train_feat_path = paths["train_features"]
    train_lbl_path = paths["train_labels"]
    test_feat_path = paths["test_features"]
    test_lbl_path = paths["test_labels"]

    if not os.path.exists(train_feat_path) or not os.path.exists(test_feat_path):
        print(f"Missing feature files: {train_feat_path} or {test_feat_path}")
        return

    X_train_raw = np.load(train_feat_path)
    y_train_raw = np.load(train_lbl_path)
    X_test_raw = np.load(test_feat_path)
    y_test_raw = np.load(test_lbl_path)
    print(f"Loaded features: train={X_train_raw.shape}, test={X_test_raw.shape}")

    n_splits = config["training"].get("n_splits", 5)
    n_components = config["training"]["pca"]["n_components"]
    k_neighbors = config["training"]["smote"]["k_neighbors"]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    tr_idx, val_idx = next(sss.split(X_train_raw, y_train_raw))
    X_tr85, y_tr85 = X_train_raw[tr_idx], y_train_raw[tr_idx]
    X_val15, y_val15 = X_train_raw[val_idx], y_train_raw[val_idx]

    pca_init = PCAReducer(n_components=n_components)
    X_tr85_pca, _ = pca_init.fit_transform(X_tr85)
    smote_init = LatentSMOTEBalancer(k_neighbors=k_neighbors, random_state=seed)
    _, y_tr85_bal = smote_init.fit_resample(X_tr85_pca, y_tr85)

    export_table1_dataset_distribution(y_train_raw, y_tr85, y_val15, y_tr85_bal, y_test_raw)
    export_table2_pca_tradeoff(X_tr85, y_tr85, X_val15, y_val15, X_test_raw, y_test_raw, config)

    print(f"Running {n_splits}-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_val_metrics, fold_test_metrics, fold_details = [], [], []
    best_overall_model = None
    best_val_acc = -1.0

    for fold_idx, (f_tr_idx, f_val_idx) in enumerate(skf.split(X_train_raw, y_train_raw), 1):
        t0 = time.time()
        X_tr, y_tr = X_train_raw[f_tr_idx], y_train_raw[f_tr_idx]
        X_val, y_val = X_train_raw[f_val_idx], y_train_raw[f_val_idx]

        pca_reducer = PCAReducer(n_components=n_components)
        X_tr_pca, exp_var = pca_reducer.fit_transform(X_tr)
        X_val_pca = pca_reducer.transform(X_val)
        X_test_pca = pca_reducer.transform(X_test_raw)

        balancer = LatentSMOTEBalancer(k_neighbors=k_neighbors, random_state=seed)
        X_tr_bal, y_tr_bal = balancer.fit_resample(X_tr_pca, y_tr)

        model = get_xgboost_classifier(config)
        model.fit(X_tr_bal, y_tr_bal, eval_set=[(X_val_pca, y_val)], verbose=False)
        elapsed = time.time() - t0

        val_metric = compute_classification_metrics(y_val, model.predict(X_val_pca))
        test_pred = model.predict(X_test_pca)
        test_prob = model.predict_proba(X_test_pca)
        test_metric = compute_classification_metrics(y_test_raw, test_pred, test_prob)

        fold_val_metrics.append(val_metric)
        fold_test_metrics.append(test_metric)

        info = {
            "fold": fold_idx,
            "pca_variance": round(exp_var * 100, 2),
            "val_accuracy": round(val_metric["accuracy"] * 100, 2),
            "test_accuracy": round(test_metric["accuracy"] * 100, 2),
            "macro_sensitivity": round(test_metric["macro_sensitivity"] * 100, 2),
            "macro_specificity": round(test_metric["macro_specificity"] * 100, 2),
            "macro_f1": round(test_metric["macro_f1"] * 100, 2),
            "fit_time_s": round(elapsed, 2),
        }
        fold_details.append(info)
        print(
            f"  Fold {fold_idx}/{n_splits}: val={info['val_accuracy']:.2f}%, "
            f"test={info['test_accuracy']:.2f}%, f1={info['macro_f1']:.2f}%, time={elapsed:.1f}s"
        )

        if val_metric["accuracy"] > best_val_acc:
            best_val_acc = val_metric["accuracy"]
            best_overall_model = model

    val_accs = [m["accuracy"] * 100 for m in fold_val_metrics]
    test_accs = [m["accuracy"] * 100 for m in fold_test_metrics]
    test_sens = [m["macro_sensitivity"] * 100 for m in fold_test_metrics]
    test_spec = [m["macro_specificity"] * 100 for m in fold_test_metrics]
    test_f1 = [m["macro_f1"] * 100 for m in fold_test_metrics]

    summary = {
        "val_accuracy_mean": round(float(np.mean(val_accs)), 2),
        "val_accuracy_std": round(float(np.std(val_accs)), 2),
        "test_accuracy_mean": round(float(np.mean(test_accs)), 2),
        "test_accuracy_std": round(float(np.std(test_accs)), 2),
        "macro_sensitivity_mean": round(float(np.mean(test_sens)), 2),
        "macro_sensitivity_std": round(float(np.std(test_sens)), 2),
        "macro_specificity_mean": round(float(np.mean(test_spec)), 2),
        "macro_specificity_std": round(float(np.std(test_spec)), 2),
        "macro_f1_mean": round(float(np.mean(test_f1)), 2),
        "macro_f1_std": round(float(np.std(test_f1)), 2),
    }

    summary_lines = [
        "5-Fold Stratified Cross-Validation Benchmark Results",
        "",
        "Fold Details:",
    ]
    for f in fold_details:
        summary_lines.append(
            f"Fold {f['fold']} | Val: {f['val_accuracy']:.2f}% | Test: {f['test_accuracy']:.2f}% | "
            f"Sens: {f['macro_sensitivity']:.2f}% | Spec: {f['macro_specificity']:.2f}% | F1: {f['macro_f1']:.2f}%"
        )
    summary_lines.extend([
        "",
        "Summary (Mean +/- Std):",
        f"Validation Accuracy: {summary['val_accuracy_mean']:.2f}% +/- {summary['val_accuracy_std']:.2f}%",
        f"Test Accuracy:       {summary['test_accuracy_mean']:.2f}% +/- {summary['test_accuracy_std']:.2f}%",
        f"Macro Sensitivity:   {summary['macro_sensitivity_mean']:.2f}% +/- {summary['macro_sensitivity_std']:.2f}%",
        f"Macro Specificity:   {summary['macro_specificity_mean']:.2f}% +/- {summary['macro_specificity_std']:.2f}%",
        f"Macro F1-Score:      {summary['macro_f1_mean']:.2f}% +/- {summary['macro_f1_std']:.2f}%",
    ])
    summary_text = "\n".join(summary_lines) + "\n"

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/benchmark_results.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    with open("outputs/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({"folds": fold_details, "summary": summary}, f, indent=2)

    print(
        f"5-Fold CV Mean +/- SD | Val: {summary['val_accuracy_mean']:.2f} +/- {summary['val_accuracy_std']:.2f}%, "
        f"Test: {summary['test_accuracy_mean']:.2f} +/- {summary['test_accuracy_std']:.2f}%"
    )
    print("Saved outputs/benchmark_results.txt and outputs/benchmark_results.json")
    return best_overall_model, fold_test_metrics


if __name__ == "__main__":
    train_and_benchmark()
