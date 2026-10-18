"""
cost_analyzer.py
────────────────
💰 UNIQUE FEATURE: Attrition Cost Calculator + ROI Analyzer

Estimates the financial cost of losing an employee and calculates the
return on investment (ROI) of retention programs.

Cost model based on SHRM (Society for Human Resource Management) research:
  - Recruitment cost: ~20–30% of annual salary
  - Onboarding/training: ~15–20% of annual salary
  - Productivity loss during ramp-up: ~25% of annual salary for 6 months
  - Manager time cost: ~10% of annual salary

Total replacement cost ≈ 70–200% of annual salary (we use 100% for senior roles)
"""

from dataclasses import dataclass
from typing import Optional


# ── Replacement cost multipliers by income bracket ───────────────────────────
COST_MULTIPLIERS = {
    "entry":    0.50,   # <₹30K/month: 50% of annual salary
    "mid":      0.75,   # ₹30K–₹60K: 75%
    "senior":   1.00,   # ₹60K–₹90K: 100%
    "executive": 1.50,  # >₹90K: 150%
}

DEPARTMENT_RISK_MULTIPLIER = {
    "Engineering": 1.2,   # Hard to replace specialized skills
    "Sales":       1.0,
    "Finance":     1.1,
    "Marketing":   0.9,
    "HR":          0.8,
}


@dataclass
class AttritionCostReport:
    """Structured report of attrition cost for one employee."""
    monthly_income:         float
    annual_income:          float
    recruitment_cost:       float
    onboarding_cost:        float
    productivity_loss:      float
    manager_time_cost:      float
    total_replacement_cost: float
    income_bracket:         str
    department:             str
    cost_category:          str   # Low / Moderate / High / Critical


def _get_income_bracket(monthly_income: float) -> str:
    """Classify income into bracket for multiplier selection."""
    if monthly_income < 30_000:
        return "entry"
    elif monthly_income < 60_000:
        return "mid"
    elif monthly_income < 90_000:
        return "senior"
    else:
        return "executive"


def calculate_replacement_cost(monthly_income: float,
                                department: str,
                                years_at_company: float) -> AttritionCostReport:
    """
    Calculate the total financial cost of replacing one employee.

    The cost model:
      - Recruitment: Job posting, recruiter fees, interviews (20% of annual)
      - Onboarding/Training: Orientation, tools, mentoring (15% of annual)
      - Productivity loss: 6-month ramp-up at reduced capacity (25% × 0.5yr)
      - Manager time: Supervision overhead during transition (10% of annual)

    Seniority (years_at_company) acts as a knowledge-loss multiplier —
    longer tenured employees carry more institutional knowledge.

    Args:
        monthly_income:  Employee's monthly salary in ₹
        department:      Employee's department string
        years_at_company: How long they've been at the company

    Returns:
        AttritionCostReport with itemized costs.
    """
    annual = monthly_income * 12
    bracket = _get_income_bracket(monthly_income)
    base_multiplier = COST_MULTIPLIERS[bracket]
    dept_multiplier = DEPARTMENT_RISK_MULTIPLIER.get(department, 1.0)

    # Knowledge retention multiplier: longer tenure = more knowledge lost
    tenure_multiplier = 1.0 + min(years_at_company * 0.02, 0.40)  # max +40%

    total_multiplier = base_multiplier * dept_multiplier * tenure_multiplier

    # Itemized costs
    recruitment_cost  = annual * 0.20 * dept_multiplier
    onboarding_cost   = annual * 0.15
    productivity_loss = annual * 0.25 * 0.5  # 6-month productivity gap
    manager_time_cost = annual * 0.10

    total = recruitment_cost + onboarding_cost + productivity_loss + manager_time_cost
    total *= tenure_multiplier  # Apply knowledge-loss premium

    # Cost category labels
    if total < 200_000:
        category = "Low"
    elif total < 500_000:
        category = "Moderate"
    elif total < 1_000_000:
        category = "High"
    else:
        category = "Critical"

    return AttritionCostReport(
        monthly_income         = monthly_income,
        annual_income          = annual,
        recruitment_cost       = round(recruitment_cost,  2),
        onboarding_cost        = round(onboarding_cost,   2),
        productivity_loss      = round(productivity_loss, 2),
        manager_time_cost      = round(manager_time_cost, 2),
        total_replacement_cost = round(total,             2),
        income_bracket         = bracket,
        department             = department,
        cost_category          = category,
    )


def calculate_retention_roi(total_replacement_cost: float,
                             retention_program_cost:  float,
                             success_rate:            float = 0.70) -> dict:
    """
    Calculate ROI of a retention intervention.

    Args:
        total_replacement_cost: Cost of losing this employee (from above).
        retention_program_cost: Cost of retention action (e.g., salary raise).
        success_rate: Probability the retention program succeeds (default 70%).

    Returns:
        Dict with expected savings, ROI percentage, and break-even point.
    """
    expected_savings = total_replacement_cost * success_rate
    net_benefit      = expected_savings - retention_program_cost
    roi_percent      = (net_benefit / retention_program_cost) * 100 if retention_program_cost > 0 else 0

    return {
        "retention_cost":        round(retention_program_cost, 2),
        "expected_savings":      round(expected_savings, 2),
        "net_benefit":           round(net_benefit, 2),
        "roi_percent":           round(roi_percent, 1),
        "worthwhile":            net_benefit > 0,
        "break_even_success_rate": round(retention_program_cost / total_replacement_cost * 100, 1)
    }


def print_cost_report(report: AttritionCostReport,
                      roi: Optional[dict] = None) -> None:
    """Pretty-print the attrition cost report to console."""
    sep = "─" * 50
    print(f"\n  {sep}")
    print(f"  💰 ATTRITION COST ANALYSIS")
    print(f"  {sep}")
    print(f"  Department      : {report.department}")
    print(f"  Monthly Income  : ₹{report.monthly_income:>10,.0f}")
    print(f"  Annual Income   : ₹{report.annual_income:>10,.0f}")
    print(f"  Bracket         : {report.income_bracket.title()}")
    print(f"  {sep}")
    print(f"  📋 Cost Breakdown:")
    print(f"  Recruitment     : ₹{report.recruitment_cost:>10,.0f}")
    print(f"  Onboarding      : ₹{report.onboarding_cost:>10,.0f}")
    print(f"  Productivity Gap: ₹{report.productivity_loss:>10,.0f}")
    print(f"  Manager Time    : ₹{report.manager_time_cost:>10,.0f}")
    print(f"  {sep}")
    print(f"  ⚠️  TOTAL COST   : ₹{report.total_replacement_cost:>10,.0f}  [{report.cost_category}]")
    print(f"  {sep}")

    if roi:
        print(f"\n  📈 RETENTION ROI ANALYSIS:")
        print(f"  Retention Cost  : ₹{roi['retention_cost']:>10,.0f}")
        print(f"  Expected Savings: ₹{roi['expected_savings']:>10,.0f}")
        print(f"  Net Benefit     : ₹{roi['net_benefit']:>10,.0f}")
        print(f"  ROI             : {roi['roi_percent']:>9.1f}%")
        print(f"  Break-even at   : {roi['break_even_success_rate']}% retention success")
        verdict = "✅ WORTH IT" if roi["worthwhile"] else "❌ NOT WORTH IT"
        print(f"  Verdict         : {verdict}")
        print(f"  {sep}")


if __name__ == "__main__":
    # Demo
    report = calculate_replacement_cost(
        monthly_income=55000, department="Engineering", years_at_company=4
    )
    roi = calculate_retention_roi(
        total_replacement_cost=report.total_replacement_cost,
        retention_program_cost=150000,
        success_rate=0.70
    )
    print_cost_report(report, roi)
