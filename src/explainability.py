"""
explainability.py
─────────────────
🔮 UNIQUE FEATURE: SHAP-based Per-Employee Prediction Explanation

Uses SHAP (SHapley Additive exPlanations) to explain WHY the model
predicted attrition for a specific employee — not just global importance.

For each prediction, generates:
  - A SHAP waterfall plot showing which features pushed the prediction up/down
  - A plain-English explanation of the top factors
  - A force plot (HTML, for Streamlit)

This is the #1 differentiator from typical student projects that only show
global feature importance charts.
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("SHAP not installed. Run: pip install shap")

VISUALS_DIR = os.path.join(os.path.dirname(__file__), "..", "visuals")


def compute_shap_values(model, X_transformed: np.ndarray, feature_names: list):
    """
    Compute SHAP values for a dataset using the appropriate explainer.

    Tree-based models (RF, XGBoost, LightGBM) use TreeExplainer (fast).
    Logistic Regression uses LinearExplainer.

    Args:
        model: Trained sklearn/XGB/LGBM model
        X_transformed: Preprocessed feature matrix (numpy array)
        feature_names: List of feature name strings

    Returns:
        (explainer, shap_values) or (None, None) if SHAP unavailable
    """
    if not HAS_SHAP:
        print("  ⚠️  SHAP not available. Install with: pip install shap")
        return None, None

    try:
        # Unwrap calibrated or frozen models to get base estimator
        base_model = model
        while hasattr(base_model, "estimator"):
            base_model = base_model.estimator

        model_type = type(base_model).__name__.lower()
        if any(t in model_type for t in ["forest", "xgb", "lgbm", "gradient", "boost"]):
            # TreeExplainer is fast and exact for tree models
            explainer   = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X_transformed)

            # For binary classification, tree models return list of 2 arrays
            if isinstance(shap_values, list):
                shap_values = shap_values[1]   # Class 1 (Attrition=Yes)
        else:
            # LinearExplainer for logistic regression
            explainer   = shap.LinearExplainer(base_model, X_transformed)
            shap_values = explainer.shap_values(X_transformed)

        return explainer, shap_values

    except Exception as e:
        print(f"  ⚠️  SHAP computation failed: {e}")
        return None, None


def plot_shap_waterfall(shap_values_single: np.ndarray,
                         feature_names: list,
                         feature_values: np.ndarray,
                         base_value: float,
                         employee_id: str = "Employee",
                         save: bool = True) -> str:
    """
    Generate a SHAP waterfall plot for a SINGLE employee's prediction.

    This shows exactly which features increased/decreased the attrition risk
    for this specific person — the "why" behind the prediction.

    Args:
        shap_values_single: 1D array of SHAP values for one employee.
        feature_names: List of feature name strings.
        feature_values: 1D array of actual feature values for this employee.
        base_value: Model's expected output (baseline SHAP value).
        employee_id: Label for the plot title.
        save: Whether to save the plot to visuals/

    Returns:
        Path to saved plot, or "" if not saved.
    """
    if not HAS_SHAP:
        return ""

    # Sort features by absolute SHAP value (most impactful first)
    indices   = np.argsort(np.abs(shap_values_single))[::-1][:12]  # Top 12
    top_names  = [feature_names[i] for i in indices]
    top_vals   = shap_values_single[indices]
    top_feats  = feature_values[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in top_vals]

    # Horizontal bar chart (waterfall style)
    bars = ax.barh(range(len(top_names)), top_vals[::-1],
                   color=colors[::-1], edgecolor="white", height=0.6)

    # Feature value annotations
    for i, (val, feat_val, name) in enumerate(
            zip(top_vals[::-1], top_feats[::-1], top_names[::-1])):
        label = f"{name} = {feat_val:.2g}"
        x_pos = val + (0.005 if val >= 0 else -0.005)
        ha    = "left" if val >= 0 else "right"
        ax.text(x_pos, i, label, va="center", ha=ha, fontsize=8.5)

    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value (← reduces risk   |   increases risk →)", fontsize=10)
    ax.set_title(f"SHAP Waterfall — Why is {employee_id} at risk?\n"
                 f"(Red = increases attrition risk, Green = reduces it)",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    if save:
        os.makedirs(VISUALS_DIR, exist_ok=True)
        path = os.path.join(VISUALS_DIR, f"shap_waterfall_{employee_id.replace(' ','_')}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    plt.show()
    return ""


def explain_in_plain_english(shap_values_single: np.ndarray,
                               feature_names: list,
                               feature_values: np.ndarray,
                               top_n: int = 3) -> str:
    """
    Convert SHAP values into a plain-English explanation of the top risk factors.

    Args:
        shap_values_single: 1D array of SHAP values for one prediction.
        feature_names: Feature name strings.
        feature_values: Actual feature values for this employee.
        top_n: How many top factors to explain.

    Returns:
        Human-readable string explaining the prediction.
    """
    if not HAS_SHAP:
        return "SHAP explanations not available."

    # Sort by absolute impact
    indices  = np.argsort(np.abs(shap_values_single))[::-1][:top_n]
    lines    = ["\n  🔍 WHY THIS PREDICTION? (Top contributing factors):"]
    lines.append("  " + "─" * 50)

    for rank, i in enumerate(indices, 1):
        name      = feature_names[i]
        shap_val  = shap_values_single[i]
        feat_val  = feature_values[i]
        direction = "⬆️ increases" if shap_val > 0 else "⬇️ reduces"
        strength  = abs(shap_val)

        # Strength description
        if strength > 0.15:
            strength_word = "strongly"
        elif strength > 0.07:
            strength_word = "moderately"
        else:
            strength_word = "slightly"

        lines.append(f"\n  {rank}. {name} = {feat_val:.2g}")
        lines.append(f"     This {direction} attrition risk ({strength_word})")
        lines.append(f"     SHAP contribution: {shap_val:+.4f}")

    return "\n".join(lines)


def generate_global_shap_summary(model, X_transformed: np.ndarray,
                                  feature_names: list) -> str:
    """
    Generate and save a SHAP summary (beeswarm) plot showing global feature
    importance across the entire dataset.

    More informative than simple bar-chart importance as it shows
    directionality (high value of feature X → high/low attrition risk).

    Args:
        model: Trained model.
        X_transformed: Full preprocessed dataset.
        feature_names: Feature name strings.

    Returns:
        Path to saved plot.
    """
    if not HAS_SHAP:
        return ""

    _, shap_values = compute_shap_values(model, X_transformed, feature_names)
    if shap_values is None:
        return ""

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_transformed,
                      feature_names=feature_names,
                      show=False, plot_size=None)
    plt.title("SHAP Global Feature Importance (Beeswarm)",
              fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    os.makedirs(VISUALS_DIR, exist_ok=True)
    path = os.path.join(VISUALS_DIR, "10_shap_summary_global.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 SHAP summary saved: 10_shap_summary_global.png")
    return path
