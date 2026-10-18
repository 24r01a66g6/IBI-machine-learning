"""
utils.py
────────
Shared utilities:
  - SQLite prediction history logger (audit trail)
  - Risk tier classifier
  - Input validator
  - PDF report generator
  - Console formatting helpers
"""

import os
import sqlite3
import json
import datetime
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(__file__))
DB_PATH       = os.path.join(BASE_DIR, "models", "prediction_history.db")
REPORTS_DIR   = os.path.join(BASE_DIR, "reports")

# ── Risk Tier Thresholds ──────────────────────────────────────────────────────
RISK_TIERS = [
    (0.80, "🔴 CRITICAL", "Immediate intervention required"),
    (0.60, "🟠 HIGH",     "Schedule retention meeting this week"),
    (0.40, "🟡 MEDIUM",   "Monitor closely; plan 1-on-1"),
    (0.20, "🟢 LOW",      "Standard engagement activities"),
    (0.00, "✅ MINIMAL",  "No immediate action needed"),
]


# ══════════════════════════════════════════════════════════════════════════════
#  INPUT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

VALID_DEPARTMENTS = {"Sales", "Engineering", "HR", "Marketing", "Finance"}

VALIDATION_RULES = {
    "Age":               (18, 65,    "Age must be between 18 and 65"),
    "MonthlyIncome":     (5000, 500000, "Monthly income must be between ₹5,000 and ₹5,00,000"),
    "YearsAtCompany":    (0, 45,     "Years at company must be between 0 and 45"),
    "JobSatisfaction":   (1, 5,      "Job satisfaction must be between 1 and 5"),
    "WorkLifeBalance":   (1, 5,      "Work-life balance must be between 1 and 5"),
    "PerformanceRating": (1, 5,      "Performance rating must be between 1 and 5"),
    "DistanceFromHome":  (0, 200,    "Distance from home must be between 0 and 200 km"),
}


def validate_input(raw_input: dict) -> tuple[bool, list[str]]:
    """
    Validate user input against business rules.

    Args:
        raw_input: Dict of feature name → raw value.

    Returns:
        (is_valid: bool, errors: list of error message strings)
    """
    errors = []

    # Validate numerical ranges
    for field, (min_val, max_val, message) in VALIDATION_RULES.items():
        val = raw_input.get(field)
        if val is None:
            errors.append(f"Missing required field: {field}")
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            errors.append(f"{field} must be a number. Got: {val!r}")
            continue
        if not (min_val <= val <= max_val):
            errors.append(message + f" (got {val})")

    # Validate Department
    dept = raw_input.get("Department", "")
    if dept not in VALID_DEPARTMENTS:
        errors.append(
            f"Department must be one of: {', '.join(sorted(VALID_DEPARTMENTS))}. "
            f"Got: {dept!r}"
        )

    # Validate OverTime
    ot = str(raw_input.get("OverTime", "")).strip().title()
    if ot not in ("Yes", "No"):
        errors.append(f"OverTime must be 'Yes' or 'No'. Got: {raw_input.get('OverTime')!r}")

    return len(errors) == 0, errors


def sanitize_input(raw_input: dict) -> dict:
    """
    Convert raw string inputs to appropriate Python types.
    Ensures OverTime is 'Yes'/'No' and numerics are float/int.
    """
    cleaned = {}
    num_fields = ["Age", "MonthlyIncome", "YearsAtCompany",
                  "JobSatisfaction", "WorkLifeBalance",
                  "PerformanceRating", "DistanceFromHome"]
    for f in num_fields:
        cleaned[f] = float(raw_input[f]) if raw_input.get(f) is not None else None

    cleaned["Department"] = str(raw_input.get("Department", "")).strip()
    cleaned["OverTime"]   = str(raw_input.get("OverTime", "No")).strip().title()
    return cleaned


# ══════════════════════════════════════════════════════════════════════════════
#  RISK TIER
# ══════════════════════════════════════════════════════════════════════════════

def get_risk_tier(probability: float) -> tuple[str, str]:
    """
    Classify an attrition probability into a human-readable risk tier.

    Args:
        probability: Attrition probability (0–1)

    Returns:
        (tier_label, tier_advice) tuple
    """
    for threshold, label, advice in RISK_TIERS:
        if probability >= threshold:
            return label, advice
    return "✅ MINIMAL", "No immediate action needed"


# ══════════════════════════════════════════════════════════════════════════════
#  SQLITE PREDICTION HISTORY LOGGER
# ══════════════════════════════════════════════════════════════════════════════

def init_db() -> None:
    """Initialize the SQLite database and create predictions table if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            age             INTEGER,
            department      TEXT,
            monthly_income  REAL,
            years_at_company REAL,
            job_satisfaction REAL,
            work_life_balance REAL,
            performance_rating REAL,
            overtime        TEXT,
            distance_from_home REAL,
            predicted_label TEXT,
            attrition_probability REAL,
            risk_tier       TEXT,
            replacement_cost REAL
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(employee: dict,
                   predicted_label: str,
                   probability: float,
                   risk_tier: str,
                   replacement_cost: Optional[float] = None) -> int:
    """
    Log a prediction result to the SQLite audit database.

    Args:
        employee: Dict of employee feature values.
        predicted_label: "Yes" or "No"
        probability: Attrition probability
        risk_tier: Risk tier label
        replacement_cost: Estimated cost if employee leaves

    Returns:
        Row ID of the inserted record.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        INSERT INTO prediction_history (
            timestamp, age, department, monthly_income, years_at_company,
            job_satisfaction, work_life_balance, performance_rating,
            overtime, distance_from_home, predicted_label,
            attrition_probability, risk_tier, replacement_cost
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.datetime.now().isoformat(),
        employee.get("Age"),
        employee.get("Department"),
        employee.get("MonthlyIncome"),
        employee.get("YearsAtCompany"),
        employee.get("JobSatisfaction"),
        employee.get("WorkLifeBalance"),
        employee.get("PerformanceRating"),
        employee.get("OverTime"),
        employee.get("DistanceFromHome"),
        predicted_label,
        probability,
        risk_tier,
        replacement_cost
    ))
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_prediction_history(limit: int = 20) -> list:
    """Retrieve recent prediction history from the database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT * FROM prediction_history ORDER BY id DESC LIMIT ?", (limit,)
    )
    rows = cursor.fetchall()
    cols = [desc[0] for desc in cursor.description]
    conn.close()
    return [dict(zip(cols, row)) for row in rows]


# ══════════════════════════════════════════════════════════════════════════════
#  PDF REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(employee: dict,
                         predicted_label: str,
                         probability: float,
                         risk_tier: str,
                         replacement_cost: float,
                         recommendations_text: str,
                         output_path: Optional[str] = None) -> str:
    """
    Generate a professional PDF report for HR managers.

    Uses fpdf2 if available, falls back to plain text report.

    Args:
        employee: Employee feature dict
        predicted_label: "Yes" or "No"
        probability: Attrition probability
        risk_tier: Risk tier label
        replacement_cost: Estimated replacement cost
        recommendations_text: Formatted recommendations string
        output_path: Where to save the PDF

    Returns:
        Path to the saved report file.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        from fpdf import FPDF

        class AttritionReport(FPDF):
            def header(self):
                self.set_fill_color(44, 62, 80)   # Dark blue header
                self.rect(0, 0, 210, 20, 'F')
                self.set_text_color(255, 255, 255)
                self.set_font("Helvetica", "B", 14)
                self.cell(0, 12, "Employee Attrition Risk Report", align="C", ln=True)
                self.set_text_color(0, 0, 0)
                self.ln(5)

            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(128, 128, 128)
                self.cell(0, 10, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} "
                                 f"| Confidential HR Document | Page {self.page_no()}", align="C")

        pdf = AttritionReport()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # ── Summary box ───────────────────────────────────────────────────────
        color = (231, 76, 60) if predicted_label == "Yes" else (46, 204, 113)
        pdf.set_fill_color(*color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 18)
        label_text = f"ATTRITION RISK: {'HIGH' if predicted_label == 'Yes' else 'LOW'}"
        pdf.cell(0, 15, label_text, align="C", fill=True, ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

        def clean_str(s: str) -> str:
            return "".join(c for c in str(s) if ord(c) < 128)

        clean_tier = clean_str(risk_tier).strip()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"Probability: {probability*100:.1f}%   |   Risk Tier: {clean_tier}", align="C", ln=True)
        pdf.ln(5)

        def section_title(title: str):
            pdf.set_fill_color(52, 73, 94)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(180, 8, f"  {title}", fill=True, ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)

        def row(label: str, value: str):
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(65, 7, f"  {label}:", border="B")
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(115, 7, f"  {clean_str(value)}", border="B", ln=True)

        # ── Employee details ──────────────────────────────────────────────────
        section_title("EMPLOYEE PROFILE")
        row("Department",        employee.get("Department", "N/A"))
        row("Age",               str(employee.get("Age", "N/A")))
        row("Monthly Income",    f"Rs.{float(employee.get('MonthlyIncome',0)):,.0f}")
        row("Years at Company",  str(employee.get("YearsAtCompany", "N/A")))
        row("OverTime",          str(employee.get("OverTime", "N/A")))
        row("Job Satisfaction",  f"{employee.get('JobSatisfaction', 'N/A')} / 5")
        row("Work-Life Balance", f"{employee.get('WorkLifeBalance', 'N/A')} / 5")
        row("Performance",       f"{employee.get('PerformanceRating', 'N/A')} / 5")
        row("Distance from Home",f"{employee.get('DistanceFromHome', 'N/A')} km")
        pdf.ln(5)

        # ── Cost analysis ─────────────────────────────────────────────────────
        section_title("FINANCIAL IMPACT")
        row("Estimated Replacement Cost", f"Rs.{replacement_cost:,.0f}")
        pdf.ln(5)

        # ── Recommendations ───────────────────────────────────────────────────
        section_title("RETENTION RECOMMENDATIONS")
        pdf.set_font("Helvetica", "", 9)
        clean_text = clean_str(recommendations_text)
        for line in clean_text.split("\n"):
            line = line.strip()
            if line:
                pdf.multi_cell(180, 5, line)

        output_path = output_path or os.path.join(
            REPORTS_DIR, f"attrition_report_{timestamp}.pdf"
        )
        pdf.output(output_path)
        return output_path

    except ImportError:
        # Fallback: plain text report
        output_path = output_path or os.path.join(
            REPORTS_DIR, f"attrition_report_{timestamp}.txt"
        )
        with open(output_path, "w") as f:
            f.write("EMPLOYEE ATTRITION RISK REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.datetime.now()}\n\n")
            f.write(f"Prediction: {predicted_label}\n")
            f.write(f"Probability: {probability*100:.1f}%\n")
            f.write(f"Risk Tier: {risk_tier}\n")
            f.write(f"Replacement Cost: Rs.{replacement_cost:,.0f}\n\n")
            f.write("Employee Profile:\n")
            for k, v in employee.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\n{recommendations_text}\n")
        return output_path


# ══════════════════════════════════════════════════════════════════════════════
#  CONSOLE FORMATTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def print_banner(title: str) -> None:
    """Print a formatted banner to console."""
    width = 60
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def print_prediction_result(label: str,
                              probability: float,
                              risk_tier: str,
                              risk_advice: str) -> None:
    """Print the main prediction result box."""
    icon = "🚨" if label == "Yes" else "✅"
    print("\n" + "┌" + "─"*50 + "┐")
    print(f"│  {icon} PREDICTION: Attrition = {label:<25}  │")
    print(f"│  📊 Probability : {probability*100:.1f}%{'':<30}│")
    print(f"│  🎯 Risk Tier   : {risk_tier:<33}│")
    print(f"│  💡 Advice      : {risk_advice[:33]:<33}│")
    print("└" + "─"*50 + "┘")
