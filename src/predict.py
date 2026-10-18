"""
predict.py
──────────
Console-based prediction interface with full validation, error handling,
SHAP explanation, cost analysis, and retention recommendations.

Usage:
    python predict.py

Or for a non-interactive single prediction:
    python predict.py --demo
"""

import os
import sys
import json
import argparse
import joblib

# ── Force UTF-8 output so emojis print correctly on Windows ──────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Path setup ────────────────────────────────────────────────────────────────
SRC_DIR  = os.path.dirname(__file__)
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from preprocessing   import preprocess_single_input, load_preprocessor
from cost_analyzer   import calculate_replacement_cost, calculate_retention_roi, print_cost_report
from recommendations import generate_recommendations, print_recommendations
from explainability  import (compute_shap_values, plot_shap_waterfall,
                              explain_in_plain_english)
from utils           import (validate_input, sanitize_input, get_risk_tier,
                              log_prediction, print_banner, print_prediction_result,
                              generate_pdf_report)

MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Valid options for user prompts ────────────────────────────────────────────
DEPARTMENTS = ["Sales", "Engineering", "HR", "Marketing", "Finance"]


# ══════════════════════════════════════════════════════════════════════════════
#  Model Loader (with error handling)
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_preprocessor():
    """
    Load the trained model and preprocessor from disk.
    Tries calibrated model first, falls back to best model.

    Returns:
        (model, preprocessor, feature_names)

    Raises:
        SystemExit if required files not found.
    """
    # Load preprocessor
    try:
        preprocessor = load_preprocessor(MODELS_DIR)
    except FileNotFoundError:
        print("❌ ERROR: Preprocessor not found. Run train_model.py first.")
        print(f"   Expected at: {os.path.join(MODELS_DIR, 'preprocessor.pkl')}")
        sys.exit(1)

    # Load calibrated model (preferred) or best model
    calibrated_path = os.path.join(MODELS_DIR, "calibrated_model.pkl")
    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")

    if os.path.exists(calibrated_path):
        model = joblib.load(calibrated_path)
        print("  ✅ Loaded: calibrated_model.pkl (reliable probability scores)")
    elif os.path.exists(best_model_path):
        model = joblib.load(best_model_path)
        print("  ✅ Loaded: best_model.pkl")
    else:
        print("❌ ERROR: No trained model found. Run train_model.py first.")
        print(f"   Expected at: {best_model_path}")
        sys.exit(1)

    # Load feature names from metrics JSON
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    feature_names = []
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            feature_names = json.load(f).get("feature_names", [])

    return model, preprocessor, feature_names


# ══════════════════════════════════════════════════════════════════════════════
#  Input Collection
# ══════════════════════════════════════════════════════════════════════════════

def prompt_input(label: str, input_type: type = float,
                  choices: list = None, default=None):
    """
    Prompt user for a single input with validation and retry logic.

    Args:
        label:      Display label for the input.
        input_type: Expected Python type (float, str, etc.)
        choices:    Optional list of valid string choices.
        default:    Default value shown in brackets.

    Returns:
        Validated value of input_type.
    """
    while True:
        prompt_str = f"  {label}"
        if choices:
            prompt_str += f" [{'/'.join(choices)}]"
        if default is not None:
            prompt_str += f" (default: {default})"
        prompt_str += ": "

        raw = input(prompt_str).strip()

        if raw == "" and default is not None:
            return default

        if choices:
            if raw.title() in [c.title() for c in choices]:
                return raw.title()
            else:
                print(f"    ⚠️  Invalid choice. Please enter one of: {', '.join(choices)}")
                continue

        try:
            return input_type(raw)
        except (ValueError, TypeError):
            print(f"    ⚠️  Expected a {input_type.__name__}. Please try again.")


def collect_employee_input() -> dict:
    """
    Interactively collect all employee features from the user.
    Validates each field and re-prompts on invalid input.

    Returns:
        Dict of employee feature values.
    """
    print("\n  Please enter the employee's information:\n")

    raw = {}
    raw["Age"]               = prompt_input("Age (18–65)", float)
    raw["Department"]        = prompt_input("Department", str, choices=DEPARTMENTS)
    raw["MonthlyIncome"]     = prompt_input("Monthly Income in ₹ (e.g. 35000)", float)
    raw["YearsAtCompany"]    = prompt_input("Years at Company (0–45)", float)
    raw["JobSatisfaction"]   = prompt_input("Job Satisfaction (1=Low, 5=High)", float)
    raw["WorkLifeBalance"]   = prompt_input("Work-Life Balance (1=Low, 5=High)", float)
    raw["PerformanceRating"] = prompt_input("Performance Rating (1–5)", float)
    raw["OverTime"]          = prompt_input("Works OverTime?", str, choices=["Yes", "No"])
    raw["DistanceFromHome"]  = prompt_input("Distance from Home (km)", float)

    # Validate entire input
    is_valid, errors = validate_input(raw)
    if not is_valid:
        print("\n  ❌ Input errors found:")
        for err in errors:
            print(f"    • {err}")
        print("  Please re-enter the information.\n")
        return collect_employee_input()   # Retry recursively

    return sanitize_input(raw)


# ══════════════════════════════════════════════════════════════════════════════
#  Prediction Engine
# ══════════════════════════════════════════════════════════════════════════════

def predict_employee(employee: dict,
                      model,
                      preprocessor,
                      feature_names: list,
                      employee_name: str = "Employee") -> dict:
    """
    Run a full prediction for one employee including:
      - Model prediction + probability
      - Risk tier classification
      - SHAP explanation (if available)
      - Attrition cost calculation
      - Retention recommendations
      - DB logging
      - PDF report generation

    Args:
        employee: Dict of feature values.
        model: Loaded model.
        preprocessor: Loaded preprocessor.
        feature_names: List of feature names.
        employee_name: For report labeling.

    Returns:
        Dict of all result values.
    """
    # ── Preprocess ────────────────────────────────────────────────────────────
    X = preprocess_single_input(employee, preprocessor)

    # ── Predict ───────────────────────────────────────────────────────────────
    label    = model.predict(X)[0]
    prob     = model.predict_proba(X)[0][1]
    label_str = "Yes" if label == 1 else "No"

    # ── Risk tier ─────────────────────────────────────────────────────────────
    risk_tier, risk_advice = get_risk_tier(prob)

    # ── Print result ──────────────────────────────────────────────────────────
    print_prediction_result(label_str, prob, risk_tier, risk_advice)

    # ── SHAP explanation ──────────────────────────────────────────────────────
    shap_explanation = ""
    if feature_names:
        try:
            _, shap_values = compute_shap_values(model, X, feature_names)
            if shap_values is not None:
                sv_single  = shap_values[0]
                feat_vals  = X[0]
                shap_explanation = explain_in_plain_english(sv_single, feature_names, feat_vals)
                print(shap_explanation)
                plot_shap_waterfall(sv_single, feature_names, feat_vals,
                                    base_value=0.0, employee_id=employee_name)
        except Exception as e:
            print(f"  ⚠️  SHAP explanation skipped: {e}")

    # ── Cost analysis ─────────────────────────────────────────────────────────
    cost_report = calculate_replacement_cost(
        monthly_income   = employee["MonthlyIncome"],
        department       = employee["Department"],
        years_at_company = employee["YearsAtCompany"]
    )
    roi = calculate_retention_roi(
        total_replacement_cost = cost_report.total_replacement_cost,
        retention_program_cost = employee["MonthlyIncome"] * 3,  # 3-month salary as retention budget
        success_rate           = 0.70
    )
    print_cost_report(cost_report, roi)

    # ── Recommendations ───────────────────────────────────────────────────────
    recs = generate_recommendations(employee, prob)
    print_recommendations(recs, employee_name)

    # ── Log to database ───────────────────────────────────────────────────────
    try:
        row_id = log_prediction(
            employee         = employee,
            predicted_label  = label_str,
            probability      = float(prob),
            risk_tier        = risk_tier,
            replacement_cost = cost_report.total_replacement_cost
        )
        print(f"\n  🗄️  Prediction logged to database (ID: {row_id})")
    except Exception as e:
        print(f"  ⚠️  DB logging failed: {e}")

    # ── PDF report ────────────────────────────────────────────────────────────
    try:
        from recommendations import format_recommendations_text
        recs_text  = format_recommendations_text(recs, employee_name)
        report_path = generate_pdf_report(
            employee            = employee,
            predicted_label     = label_str,
            probability         = float(prob),
            risk_tier           = risk_tier,
            replacement_cost    = cost_report.total_replacement_cost,
            recommendations_text = recs_text
        )
        print(f"  📄 PDF Report saved → {report_path}")
    except Exception as e:
        print(f"  ⚠️  PDF report skipped: {e}")

    return {
        "label":            label_str,
        "probability":      float(prob),
        "risk_tier":        risk_tier,
        "replacement_cost": cost_report.total_replacement_cost,
        "recommendations":  recs
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Demo Mode (non-interactive)
# ══════════════════════════════════════════════════════════════════════════════

DEMO_EMPLOYEES = [
    {
        "name": "Riya Sharma (High Risk)",
        "data": {
            "Age": 26, "Department": "Sales", "MonthlyIncome": 21000,
            "YearsAtCompany": 1, "JobSatisfaction": 2, "WorkLifeBalance": 2,
            "PerformanceRating": 4, "OverTime": "Yes", "DistanceFromHome": 40
        }
    },
    {
        "name": "Arjun Mehta (Low Risk)",
        "data": {
            "Age": 38, "Department": "Engineering", "MonthlyIncome": 72000,
            "YearsAtCompany": 9, "JobSatisfaction": 5, "WorkLifeBalance": 4,
            "PerformanceRating": 4, "OverTime": "No", "DistanceFromHome": 8
        }
    }
]


# ══════════════════════════════════════════════════════════════════════════════
#  Main CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Employee Attrition Prediction — Console Interface"
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run with pre-filled demo employees")
    args = parser.parse_args()

    print_banner("🏢 EMPLOYEE ATTRITION PREDICTION SYSTEM")
    print("  Powered by ML + SHAP + Cost Analysis + Recommendations")

    # Load model
    print("\n  Loading model...")
    model, preprocessor, feature_names = load_model_and_preprocessor()

    if args.demo:
        # ── Demo mode ─────────────────────────────────────────────────────────
        for emp in DEMO_EMPLOYEES:
            print_banner(f"📋 DEMO: {emp['name']}")
            result = predict_employee(
                employee      = sanitize_input(emp["data"]),
                model         = model,
                preprocessor  = preprocessor,
                feature_names = feature_names,
                employee_name = emp["name"]
            )
    else:
        # ── Interactive mode ───────────────────────────────────────────────────
        while True:
            print_banner("📝 NEW PREDICTION")
            employee = collect_employee_input()
            employee_name = input("\n  Employee name/ID (optional, press Enter to skip): ").strip()
            employee_name = employee_name or "Employee"

            predict_employee(
                employee      = employee,
                model         = model,
                preprocessor  = preprocessor,
                feature_names = feature_names,
                employee_name = employee_name
            )

            again = input("\n  🔄 Predict for another employee? (yes/no): ").strip().lower()
            if again not in ("yes", "y"):
                print("\n  👋 Thank you for using the Attrition Prediction System!")
                break


if __name__ == "__main__":
    main()
