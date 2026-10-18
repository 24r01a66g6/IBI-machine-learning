"""
generate_data.py
────────────────
Generates a realistic synthetic HR dataset with 1500 employee records.
Attrition is correlated with overtime, low satisfaction, and low income
to make the dataset realistic for ML modeling.
"""

import numpy as np
import pandas as pd
import os
import sys

# ── Force UTF-8 output so emojis print correctly on Windows ──────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
N = 1500  # Total employees to generate

# ── Helper: Weighted Attrition Logic ─────────────────────────────────────────
def compute_attrition_probability(row: dict) -> float:
    """
    Compute probability of attrition based on domain knowledge.
    Higher overtime + lower satisfaction + lower income → higher risk.
    """
    prob = 0.12  # Base attrition rate (~12%)

    # Overtime increases attrition significantly
    if row["OverTime"] == "Yes":
        prob += 0.20

    # Low job satisfaction
    if row["JobSatisfaction"] <= 2:
        prob += 0.15
    elif row["JobSatisfaction"] >= 4:
        prob -= 0.08

    # Low work-life balance
    if row["WorkLifeBalance"] <= 2:
        prob += 0.12
    elif row["WorkLifeBalance"] >= 4:
        prob -= 0.05

    # Low income
    if row["MonthlyIncome"] < 25000:
        prob += 0.10
    elif row["MonthlyIncome"] > 70000:
        prob -= 0.08

    # Short tenure (first 2 years are high-risk)
    if row["YearsAtCompany"] <= 2:
        prob += 0.10
    elif row["YearsAtCompany"] >= 8:
        prob -= 0.06

    # Younger employees are more mobile
    if row["Age"] < 30:
        prob += 0.08

    # Long commute
    if row["DistanceFromHome"] > 30:
        prob += 0.08

    # Department-based adjustments
    dept_risk = {
        "Sales": 0.08,
        "HR": 0.04,
        "Engineering": -0.03,
        "Marketing": 0.03,
        "Finance": -0.02
    }
    prob += dept_risk.get(row["Department"], 0)

    return float(np.clip(prob, 0.02, 0.95))


def generate_dataset(n: int = N) -> pd.DataFrame:
    """
    Generate a synthetic HR employee dataset with realistic feature correlations.

    Returns:
        pd.DataFrame: Dataset with 'n' employee records.
    """
    departments = ["Sales", "Engineering", "HR", "Marketing", "Finance"]
    dept_probs  = [0.30, 0.30, 0.15, 0.15, 0.10]

    # ── Base features ─────────────────────────────────────────────────────────
    dept = np.random.choice(departments, size=n, p=dept_probs)

    # Income varies by department to be realistic
    income_map = {
        "Sales":       (30000, 20000),
        "Engineering": (55000, 25000),
        "HR":          (28000, 10000),
        "Marketing":   (35000, 15000),
        "Finance":     (45000, 20000),
    }
    income = np.array([
        max(15000, int(np.random.normal(income_map[d][0], income_map[d][1])))
        for d in dept
    ])

    age               = np.random.randint(22, 58, size=n)
    years_at_company  = np.clip(
        np.random.normal(loc=6, scale=4, size=n).astype(int), 0, 35
    )
    job_satisfaction  = np.random.randint(1, 6, size=n)      # 1–5
    work_life_balance = np.random.randint(1, 6, size=n)      # 1–5
    performance_rating = np.random.randint(1, 6, size=n)     # 1–5
    overtime          = np.random.choice(["Yes", "No"], size=n, p=[0.35, 0.65])
    distance_from_home = np.random.randint(1, 60, size=n)

    # Introduce ~5% missing values in non-critical fields (realistic noise)
    def add_missing(arr, rate=0.03):
        arr = arr.astype(object)
        mask = np.random.random(len(arr)) < rate
        arr[mask] = np.nan
        return arr

    job_satisfaction_noisy  = add_missing(job_satisfaction.astype(float))
    work_life_balance_noisy = add_missing(work_life_balance.astype(float))
    distance_noisy          = add_missing(distance_from_home.astype(float))

    # ── Build raw DataFrame ───────────────────────────────────────────────────
    df = pd.DataFrame({
        "EmployeeID":       range(1001, 1001 + n),
        "Age":              age,
        "Department":       dept,
        "MonthlyIncome":    income,
        "YearsAtCompany":   years_at_company,
        "JobSatisfaction":  job_satisfaction_noisy,
        "WorkLifeBalance":  work_life_balance_noisy,
        "PerformanceRating": performance_rating,
        "OverTime":         overtime,
        "DistanceFromHome": distance_noisy,
    })

    # ── Compute attrition label per row ───────────────────────────────────────
    # Temporarily fill NaNs for probability calculation only
    df_temp = df.copy()
    df_temp["JobSatisfaction"]  = df_temp["JobSatisfaction"].fillna(3)
    df_temp["WorkLifeBalance"]  = df_temp["WorkLifeBalance"].fillna(3)
    df_temp["DistanceFromHome"] = df_temp["DistanceFromHome"].fillna(15)

    attrition_probs = df_temp.apply(
        lambda r: compute_attrition_probability(r.to_dict()), axis=1
    )
    attrition = np.array([
        "Yes" if np.random.random() < p else "No"
        for p in attrition_probs
    ])
    df["Attrition"] = attrition

    return df


if __name__ == "__main__":
    output_dir  = os.path.dirname(__file__)
    output_path = os.path.join(output_dir, "employee_data.csv")

    df = generate_dataset()

    # ── Print summary ─────────────────────────────────────────────────────────
    print("=" * 50)
    print(f"  Dataset Generated: {len(df)} employees")
    print(f"  Attrition Rate   : {(df['Attrition']=='Yes').mean()*100:.1f}%")
    print(f"  Missing Values   : {df.isnull().sum().sum()}")
    print("=" * 50)
    print(df.head())

    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to: {output_path}")
