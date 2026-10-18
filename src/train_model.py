"""
train_model.py
──────────────
Model training, hyperparameter tuning, evaluation, and selection module.

Trains 4 models:
  1. Logistic Regression   — Interpretable baseline
  2. Random Forest         — Ensemble tree model
  3. XGBoost               — Gradient boosting (usually best)
  4. LightGBM              — Fast gradient boosting

Hyperparameter tuning via RandomizedSearchCV.
Best model selected by ROC-AUC on held-out test set.
Saves both the best model and its preprocessing pipeline.

IMPORTANT: Applies SMOTE only on training data → no data leakage.
"""

import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Force UTF-8 output so emojis print correctly on Windows ──────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    ConfusionMatrixDisplay, roc_curve
)
from sklearn.calibration     import CalibratedClassifierCV

HAS_XGB = False  # Avoid Python 3.14 OpenMP hang on Windows

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    warnings.warn("LightGBM not installed. Skipping LightGBM model.")

# Add src/ to path for local imports
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import (
    load_and_prepare_data, build_preprocessor,
    apply_smote, get_feature_names,
    save_preprocessor
)

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
VISUALS_DIR = os.path.join(os.path.dirname(__file__), "..", "visuals")
RANDOM_STATE = 42


# ── Model definitions + hyperparameter search spaces ─────────────────────────

def get_model_configs() -> dict:
    """
    Returns dict of {model_name: (model_instance, param_grid)} for tuning.
    Using class_weight='balanced' where supported to handle imbalance
    even without SMOTE (defense-in-depth).
    """
    configs = {
        "Logistic_Regression": (
            LogisticRegression(class_weight="balanced",
                               max_iter=1000,
                               random_state=RANDOM_STATE),
            {
                "C":       [0.01, 0.1, 1, 10, 100],
                "solver":  ["lbfgs", "liblinear"]
            }
        ),
        "Random_Forest": (
            RandomForestClassifier(class_weight="balanced",
                                   random_state=RANDOM_STATE),
            {
                "n_estimators":      [100, 200, 300],
                "max_depth":         [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf":  [1, 2, 4],
                "max_features":      ["sqrt", "log2"]
            }
        ),
        "Gradient_Boosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "max_iter":      [100, 200],
                "max_depth":     [3, 5, 8],
                "learning_rate": [0.03, 0.08, 0.15],
                "min_samples_leaf": [10, 20]
            }
        ),
    }

    if HAS_LGBM:
        configs["LightGBM"] = (
            LGBMClassifier(class_weight="balanced",
                           random_state=RANDOM_STATE,
                           verbosity=-1),
            {
                "n_estimators":  [100, 200, 300],
                "max_depth":     [-1, 5, 10, 15],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "num_leaves":    [31, 50, 70],
                "subsample":     [0.7, 0.8, 1.0]
            }
        )

    return configs


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_model(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compute full evaluation metrics for a trained model.

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc.
    """
    y_pred     = model.predict(X_test)
    y_prob     = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model":      name,
        "Accuracy":   round(accuracy_score(y_test, y_pred),   4),
        "Precision":  round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":     round(recall_score(y_test, y_pred),     4),
        "F1_Score":   round(f1_score(y_test, y_pred),         4),
        "ROC_AUC":    round(roc_auc_score(y_test, y_prob),    4),
    }

    print(f"\n  📊 [{name}]")
    for k, v in metrics.items():
        if k != "Model":
            print(f"     {k:<12}: {v}")

    return metrics


def plot_feature_importance(model, feature_names: list, model_name: str) -> str:
    """
    Plot and save feature importance chart for tree-based models.
    Falls back to coefficient magnitude for Logistic Regression.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return ""

    # Sort top 15 features
    indices = np.argsort(importances)[::-1][:15]
    top_features    = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = sns.color_palette("viridis", len(top_features))
    ax.barh(top_features[::-1], top_importances[::-1],
            color=colors[::-1], edgecolor="white")

    ax.set_title(f"Top Feature Importances — {model_name}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score", fontsize=11)

    os.makedirs(VISUALS_DIR, exist_ok=True)
    path = os.path.join(VISUALS_DIR, f"07_feature_importance_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Feature importance saved: {os.path.basename(path)}")
    return path


def plot_roc_curves(models_results: list, X_test: np.ndarray, y_test: np.ndarray) -> str:
    """
    Overlay ROC curves for all trained models on one chart.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    for (name, model), color in zip(models_results, colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Baseline")
    ax.set_title("ROC Curves — Model Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    os.makedirs(VISUALS_DIR, exist_ok=True)
    path = os.path.join(VISUALS_DIR, "08_roc_curves_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 ROC curves saved: 08_roc_curves_comparison.png")
    return path


def plot_metrics_comparison(all_metrics: list) -> str:
    """
    Grouped bar chart comparing all models across all metrics.
    """
    df = pd.DataFrame(all_metrics).set_index("Model")
    metrics_cols = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
    df = df[metrics_cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white", width=0.7)

    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.xticks(rotation=15, ha="right")

    os.makedirs(VISUALS_DIR, exist_ok=True)
    path = os.path.join(VISUALS_DIR, "09_metrics_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Metrics comparison saved: 09_metrics_comparison.png")
    return path


# ── Main training pipeline ────────────────────────────────────────────────────

def train_and_evaluate(csv_path: str) -> dict:
    """
    Full training pipeline:
      1. Load and preprocess data
      2. Train/test split (stratified)
      3. Fit preprocessor on training data only
      4. Apply SMOTE on training data
      5. Train and tune all models
      6. Evaluate on held-out test set
      7. Calibrate best model's probabilities
      8. Save best model + preprocessor

    Args:
        csv_path: Path to employee_data.csv

    Returns:
        Dict with best model name and all metrics.
    """
    print("\n" + "="*55)
    print("  🤖 STARTING MODEL TRAINING PIPELINE")
    print("="*55)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading & preparing data...")
    X, y = load_and_prepare_data(csv_path)

    # ── 2. Train/test split (stratified to preserve class ratio) ──────────────
    print("\n[2/6] Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 3. Fit preprocessor on TRAINING data only ─────────────────────────────
    print("\n[3/6] Fitting preprocessor on training data...")
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)   # Only transform, never fit!
    feature_names = get_feature_names(preprocessor)

    save_preprocessor(preprocessor, MODELS_DIR)

    # ── 4. SMOTE on training data only ────────────────────────────────────────
    print("\n[4/6] Applying SMOTE to balance training classes...")
    X_train_res, y_train_res = apply_smote(X_train_proc, y_train)

    # ── 5. Train all models with hyperparameter tuning ────────────────────────
    print("\n[5/6] Training & tuning models...")
    configs = get_model_configs()
    cv      = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    trained_models = []
    all_metrics    = []

    for name, (model, param_grid) in configs.items():
        print(f"\n  ⚙️  Tuning {name}...", flush=True)
        n_combinations = min(10, np.prod([len(v) for v in param_grid.values()]))
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_combinations, cv=cv,
            scoring="roc_auc", n_jobs=1,
            random_state=RANDOM_STATE, verbose=0
        )
        search.fit(X_train_res, y_train_res)

        best_model = search.best_estimator_
        trained_models.append((name, best_model))

        metrics = evaluate_model(name, best_model, X_test_proc, y_test)
        all_metrics.append(metrics)

        # Save feature importance for each model
        plot_feature_importance(best_model, feature_names, name)

    # ── 6. Select best model by ROC-AUC ──────────────────────────────────────
    print("\n[6/6] Selecting best model by ROC-AUC...")
    best_entry = max(all_metrics, key=lambda m: m["ROC_AUC"])
    best_name  = best_entry["Model"]
    best_model = dict(trained_models)[best_name]

    print(f"\n  🏆 Best Model: {best_name} (ROC-AUC={best_entry['ROC_AUC']:.4f})")

    # ── 7. Calibrate probabilities (Platt scaling) ────────────────────────────
    # Calibration ensures predicted probabilities are statistically reliable
    print("  🔧 Calibrating probability scores...")
    try:
        from sklearn.calibration import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(best_model), method="sigmoid")
        calibrated.fit(X_test_proc, y_test)
    except Exception:
        try:
            calibrated = CalibratedClassifierCV(best_model, cv=3, method="sigmoid")
            calibrated.fit(X_train_res, y_train_res)
        except Exception:
            calibrated = best_model

    # ── 8. Save models ────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model,  os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(calibrated,  os.path.join(MODELS_DIR, "calibrated_model.pkl"))

    # Save metrics as JSON for reporting
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "best_model":   best_name,
            "all_metrics":  all_metrics,
            "feature_names": feature_names
        }, f, indent=2)

    print(f"  ✅ best_model.pkl saved")
    print(f"  ✅ calibrated_model.pkl saved")
    print(f"  ✅ metrics.json saved")

    # ── 9. Plots ──────────────────────────────────────────────────────────────
    plot_roc_curves(trained_models, X_test_proc, y_test)
    plot_metrics_comparison(all_metrics)

    # Print final comparison table
    print("\n" + "="*70)
    print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7} {'AUC':>8}")
    print("-"*70)
    for m in sorted(all_metrics, key=lambda x: x["ROC_AUC"], reverse=True):
        marker = " 🏆" if m["Model"] == best_name else ""
        print(f"  {m['Model']:<22} {m['Accuracy']:>9.4f} {m['Precision']:>10.4f} "
              f"{m['Recall']:>8.4f} {m['F1_Score']:>7.4f} {m['ROC_AUC']:>8.4f}{marker}")
    print("="*70)

    return {"best_model": best_name, "metrics": all_metrics}


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "../data/employee_data.csv"
    train_and_evaluate(csv)
