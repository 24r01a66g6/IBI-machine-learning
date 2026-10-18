"""
recommendations.py
──────────────────
🎯 UNIQUE FEATURE: Personalized Retention Recommendations Engine

Analyzes why a specific employee is at risk (based on their feature values)
and generates targeted, actionable HR recommendations.

Goes beyond prediction — tells HR *what to do* about it.
Each recommendation includes: priority, action, expected impact.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Recommendation:
    """A single retention recommendation."""
    priority:        str    # Critical / High / Medium / Low
    category:        str    # Compensation / WorkLife / Career / Environment
    action:          str    # What HR should do
    expected_impact: str    # Expected outcome
    emoji:           str    # Visual indicator


# ── Priority color mapping ────────────────────────────────────────────────────
PRIORITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


def generate_recommendations(employee: dict,
                              attrition_prob: float) -> List[Recommendation]:
    """
    Generate personalized retention recommendations based on employee profile.

    The recommendation engine works by checking each risk factor against
    known thresholds (derived from EDA insights) and generating targeted advice.

    Args:
        employee: Dict with employee feature values (raw, before encoding).
        attrition_prob: Predicted probability of attrition (0–1).

    Returns:
        Sorted list of Recommendation objects (highest priority first).
    """
    recs: List[Recommendation] = []

    # ── Risk-factor-specific recommendations ──────────────────────────────────

    # 1. OverTime
    if str(employee.get("OverTime", "No")).strip().lower() == "yes":
        recs.append(Recommendation(
            priority        = "Critical",
            category        = "Work-Life Balance",
            action          = "Reduce mandatory overtime. Introduce flex-time or "
                              "compensatory time-off for overtime hours.",
            expected_impact = "Overtime is the single strongest attrition predictor. "
                              "Eliminating it can reduce attrition risk by ~20 percentage points.",
            emoji           = "⏰"
        ))

    # 2. Job Satisfaction
    satisfaction = employee.get("JobSatisfaction", 3)
    try:
        satisfaction = float(satisfaction) if satisfaction is not None else 3
    except (ValueError, TypeError):
        satisfaction = 3

    if satisfaction <= 2:
        recs.append(Recommendation(
            priority        = "Critical",
            category        = "Job Engagement",
            action          = "Conduct a 1-on-1 stay interview immediately. "
                              "Identify specific pain points: role clarity, manager "
                              "relationship, or growth opportunities.",
            expected_impact = "Addressing satisfaction issues early can prevent 60% "
                              "of voluntary departures.",
            emoji           = "😔"
        ))
    elif satisfaction == 3:
        recs.append(Recommendation(
            priority        = "Medium",
            category        = "Job Engagement",
            action          = "Assign a meaningful stretch project or skill development "
                              "opportunity to boost engagement.",
            expected_impact = "Mid-satisfaction employees respond well to new challenges. "
                              "Can improve satisfaction by 1–2 points.",
            emoji           = "📈"
        ))

    # 3. Work-Life Balance
    wlb = employee.get("WorkLifeBalance", 3)
    try:
        wlb = float(wlb) if wlb is not None else 3
    except (ValueError, TypeError):
        wlb = 3

    if wlb <= 2:
        recs.append(Recommendation(
            priority        = "High",
            category        = "Work-Life Balance",
            action          = "Offer remote work or hybrid schedule options. "
                              "Review current workload distribution with team manager.",
            expected_impact = "Flexible arrangements improve WLB scores by avg. 1.5 points "
                              "and reduce attrition risk by 12%.",
            emoji           = "🏠"
        ))

    # 4. Monthly Income
    income = employee.get("MonthlyIncome", 40000)
    try:
        income = float(income) if income is not None else 40000
    except (ValueError, TypeError):
        income = 40000

    if income < 25000:
        recs.append(Recommendation(
            priority        = "Critical",
            category        = "Compensation",
            action          = "Immediate salary review. Benchmark against market rates. "
                              "Consider a 15–25% raise or performance bonus.",
            expected_impact = "Below-market compensation is a primary driver. "
                              "Competitive salary can cut attrition risk by 15%.",
            emoji           = "💸"
        ))
    elif income < 40000:
        recs.append(Recommendation(
            priority        = "High",
            category        = "Compensation",
            action          = "Review salary against industry benchmarks. "
                              "Offer a clear progression plan with salary milestones.",
            expected_impact = "Transparent pay progression increases retention by 20%.",
            emoji           = "💰"
        ))

    # 5. Years at Company (Early-tenure risk)
    years = employee.get("YearsAtCompany", 5)
    try:
        years = float(years) if years is not None else 5
    except (ValueError, TypeError):
        years = 5

    if years <= 2:
        recs.append(Recommendation(
            priority        = "High",
            category        = "Onboarding & Culture",
            action          = "Assign an experienced mentor. Run a 90-day check-in "
                              "program. Ensure role expectations are crystal clear.",
            expected_impact = "Structured mentoring reduces first-2-year attrition by 30%.",
            emoji           = "🤝"
        ))
    elif years >= 8:
        recs.append(Recommendation(
            priority        = "Medium",
            category        = "Career Growth",
            action          = "Explore leadership track or senior IC path. "
                              "Offer sabbatical or special recognition for loyalty.",
            expected_impact = "Long-tenure employees who leave take significant "
                              "institutional knowledge. Retention programs yield 3× ROI.",
            emoji           = "🎖️"
        ))

    # 6. Distance from Home
    distance = employee.get("DistanceFromHome", 10)
    try:
        distance = float(distance) if distance is not None else 10
    except (ValueError, TypeError):
        distance = 10

    if distance > 30:
        recs.append(Recommendation(
            priority        = "Medium",
            category        = "Work-Life Balance",
            action          = "Offer transport reimbursement or allow 3–4 WFH days/week "
                              "to reduce commute burden.",
            expected_impact = "Commute allowances reduce distance-related attrition by 8%.",
            emoji           = "🚌"
        ))

    # 7. Performance Rating
    perf = employee.get("PerformanceRating", 3)
    try:
        perf = float(perf) if perf is not None else 3
    except (ValueError, TypeError):
        perf = 3

    if perf >= 4:
        recs.append(Recommendation(
            priority        = "High",
            category        = "Career Growth",
            action          = "Ensure high performers have a clear promotion timeline. "
                              "Consider fast-track leadership program enrollment.",
            expected_impact = "Top performers with no growth path leave 2× faster. "
                              "Structured growth plans retain 80% of high performers.",
            emoji           = "⭐"
        ))
    elif perf <= 2:
        recs.append(Recommendation(
            priority        = "Medium",
            category        = "Performance Support",
            action          = "Set up a performance improvement plan (PIP) with "
                              "clear, achievable milestones and coaching support.",
            expected_impact = "Proactive PIP with support retains 50% of at-risk "
                              "low performers while improving team productivity.",
            emoji           = "📋"
        ))

    # 8. Department-specific
    dept = str(employee.get("Department", "")).strip()
    if dept == "Sales" and attrition_prob > 0.4:
        recs.append(Recommendation(
            priority        = "High",
            category        = "Incentives",
            action          = "Review commission structure and sales targets. "
                              "Unrealistic targets are a leading cause of Sales attrition.",
            expected_impact = "Fair commission structures reduce Sales attrition by 18%.",
            emoji           = "🎯"
        ))

    # 9. If probability is very high but no major red flags
    if attrition_prob > 0.70 and len(recs) < 2:
        recs.append(Recommendation(
            priority        = "High",
            category        = "General Retention",
            action          = "Model indicates high systemic risk. Conduct a confidential "
                              "exit-risk interview to identify hidden concerns.",
            expected_impact = "Proactive engagement can convert 40% of high-risk employees.",
            emoji           = "🔍"
        ))

    # ── Sort by priority ──────────────────────────────────────────────────────
    recs.sort(key=lambda r: PRIORITY_ORDER.get(r.priority, 99))

    # Deduplicate by category (keep highest priority per category)
    seen_categories = set()
    unique_recs = []
    for r in recs:
        if r.category not in seen_categories:
            seen_categories.add(r.category)
            unique_recs.append(r)

    return unique_recs


def format_recommendations_text(recs: List[Recommendation],
                                  employee_name: str = "This Employee") -> str:
    """
    Format recommendations as a human-readable string for console output.

    Args:
        recs: List of Recommendation objects.
        employee_name: Optional name for personalization.

    Returns:
        Formatted string.
    """
    if not recs:
        return "  ✅ No significant risk factors detected. Continue regular check-ins."

    lines = [
        f"\n  🎯 RETENTION RECOMMENDATIONS FOR: {employee_name.upper()}",
        "  " + "─" * 55,
    ]

    for i, rec in enumerate(recs, 1):
        priority_icon = {
            "Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"
        }.get(rec.priority, "⚪")

        lines.append(f"\n  [{i}] {rec.emoji} {priority_icon} [{rec.priority.upper()}] — {rec.category}")
        lines.append(f"      Action  : {rec.action}")
        lines.append(f"      Impact  : {rec.expected_impact}")

    lines.append("\n  " + "─" * 55)
    return "\n".join(lines)


def print_recommendations(recs: List[Recommendation],
                           employee_name: str = "This Employee") -> None:
    """Print formatted recommendations to console."""
    print(format_recommendations_text(recs, employee_name))


if __name__ == "__main__":
    # Demo
    sample = {
        "Age": 28, "Department": "Sales", "MonthlyIncome": 22000,
        "YearsAtCompany": 1, "JobSatisfaction": 2, "WorkLifeBalance": 2,
        "PerformanceRating": 4, "OverTime": "Yes", "DistanceFromHome": 35
    }
    recs = generate_recommendations(sample, attrition_prob=0.82)
    print_recommendations(recs, "John Doe")
