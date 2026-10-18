"""
eda.py
──────
Exploratory Data Analysis module.
Generates 6 publication-quality charts saved to the visuals/ directory:

  1. Attrition rate by Department (bar chart)
  2. Attrition rate by OverTime status (bar chart)
  3. MonthlyIncome vs Attrition (box plot)
  4. Correlation heatmap of numerical features
  5. Job Satisfaction distribution by Attrition (violin plot)
  6. Age distribution by Attrition (KDE plot)

All plots auto-save to visuals/ and optionally display.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Force UTF-8 output so emojis print correctly on Windows ──────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Style settings ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
VISUALS_DIR = os.path.join(os.path.dirname(__file__), "..", "visuals")


def _save(fig: plt.Figure, filename: str) -> str:
    """Save a matplotlib figure and return the saved path."""
    os.makedirs(VISUALS_DIR, exist_ok=True)
    path = os.path.join(VISUALS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved: {filename}")
    return path


def plot_attrition_by_department(df: pd.DataFrame) -> str:
    """
    Bar chart: Attrition rate (%) per Department.
    Shows which departments have the highest turnover risk.
    """
    dept_attr = (
        df.groupby("Department")["Attrition"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index()
        .rename(columns={"Attrition": "AttritionRate"})
        .sort_values("AttritionRate", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        dept_attr["Department"], dept_attr["AttritionRate"],
        color=sns.color_palette("Reds_r", len(dept_attr)),
        edgecolor="white", linewidth=0.8
    )

    # Annotate bars with percentage
    for bar, val in zip(bars, dept_attr["AttritionRate"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_title("Employee Attrition Rate by Department", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Department", fontsize=11)
    ax.set_ylabel("Attrition Rate (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(0, dept_attr["AttritionRate"].max() + 10)

    return _save(fig, "01_attrition_by_department.png")


def plot_attrition_by_overtime(df: pd.DataFrame) -> str:
    """
    Grouped bar chart: Attrition count split by OverTime status.
    Highlights how overtime strongly correlates with attrition.
    """
    ot_counts = df.groupby(["OverTime", "Attrition"]).size().reset_index(name="Count")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=ot_counts, x="OverTime", y="Count",
                hue="Attrition", palette={"Yes": "#e74c3c", "No": "#2ecc71"},
                ax=ax, edgecolor="white")

    ax.set_title("Attrition Count by OverTime Status", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("OverTime", fontsize=11)
    ax.set_ylabel("Number of Employees", fontsize=11)
    ax.legend(title="Attrition", title_fontsize=10)

    return _save(fig, "02_attrition_by_overtime.png")


def plot_income_vs_attrition(df: pd.DataFrame) -> str:
    """
    Box plot: MonthlyIncome distribution for employees who left vs stayed.
    Lower income → higher attrition risk.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    df_plot = df.copy()
    df_plot["MonthlyIncome_K"] = df_plot["MonthlyIncome"] / 1000  # Convert to ₹K

    sns.boxplot(data=df_plot, x="Attrition", y="MonthlyIncome_K",
                palette={"Yes": "#e74c3c", "No": "#3498db"},
                width=0.45, flierprops=dict(marker='o', alpha=0.4), ax=ax)

    ax.set_title("Monthly Income vs Attrition Status", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Attrition", fontsize=11)
    ax.set_ylabel("Monthly Income (₹ Thousands)", fontsize=11)

    # Median annotation
    for i, label in enumerate(["No", "Yes"]):
        median = df_plot[df_plot["Attrition"] == label]["MonthlyIncome_K"].median()
        ax.text(i, median + 1, f"₹{median:.0f}K", ha="center", fontsize=9,
                color="white", fontweight="bold",
                bbox=dict(facecolor="#2c3e50", alpha=0.6, boxstyle="round,pad=0.3"))

    return _save(fig, "03_income_vs_attrition.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """
    Heatmap of Pearson correlation among numerical features + binary Attrition.
    """
    df_num = df[["Age", "MonthlyIncome", "YearsAtCompany",
                 "JobSatisfaction", "WorkLifeBalance",
                 "PerformanceRating", "DistanceFromHome"]].copy()
    df_num["Attrition_Bin"] = (df["Attrition"] == "Yes").astype(int)

    corr = df_num.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Upper triangle mask
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.8}, ax=ax)

    ax.set_title("Feature Correlation Heatmap (incl. Attrition)",
                 fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    return _save(fig, "04_correlation_heatmap.png")


def plot_job_satisfaction_violin(df: pd.DataFrame) -> str:
    """
    Violin plot: Job Satisfaction score distribution split by Attrition.
    Shows that employees with low satisfaction (1–2) leave more.
    """
    df_plot = df.dropna(subset=["JobSatisfaction"]).copy()
    df_plot["JobSatisfaction"] = df_plot["JobSatisfaction"].astype(int)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(data=df_plot, x="Attrition", y="JobSatisfaction",
                   palette={"Yes": "#e74c3c", "No": "#2ecc71"},
                   inner="quartile", ax=ax)

    ax.set_title("Job Satisfaction Distribution by Attrition",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Attrition", fontsize=11)
    ax.set_ylabel("Job Satisfaction Score (1–5)", fontsize=11)
    ax.set_yticks([1, 2, 3, 4, 5])

    return _save(fig, "05_job_satisfaction_violin.png")


def plot_age_distribution(df: pd.DataFrame) -> str:
    """
    KDE (density) plot: Age distribution for employees who left vs stayed.
    Younger employees tend to leave more often.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, color in [("No", "#2ecc71"), ("Yes", "#e74c3c")]:
        subset = df[df["Attrition"] == label]["Age"].dropna()
        sns.kdeplot(subset, fill=True, alpha=0.4, color=color,
                    label=f"Attrition: {label}", ax=ax)

    ax.set_title("Age Distribution by Attrition Status",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Age (years)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)

    return _save(fig, "06_age_distribution_kde.png")


def print_key_statistics(df: pd.DataFrame) -> None:
    """Print key EDA statistics to console."""
    print("\n" + "="*55)
    print("  📋 KEY EDA STATISTICS")
    print("="*55)
    print(f"  Total Employees      : {len(df)}")
    print(f"  Attrition Count      : {(df['Attrition']=='Yes').sum()}")
    print(f"  Overall Attrition %  : {(df['Attrition']=='Yes').mean()*100:.1f}%")
    print(f"  Avg Monthly Income   : ₹{df['MonthlyIncome'].mean():,.0f}")
    print(f"  Avg Age              : {df['Age'].mean():.1f} years")
    print(f"  OT Attrition Rate    : "
          f"{(df[df['OverTime']=='Yes']['Attrition']=='Yes').mean()*100:.1f}%")
    print(f"  Non-OT Attrition Rate: "
          f"{(df[df['OverTime']=='No']['Attrition']=='Yes').mean()*100:.1f}%")

    print("\n  📊 Attrition Rate by Department:")
    dept_rates = (df.groupby("Department")["Attrition"]
                  .apply(lambda x: f"{(x=='Yes').mean()*100:.1f}%"))
    for dept, rate in dept_rates.items():
        print(f"    {dept:<15} → {rate}")
    print("="*55)


def run_eda(csv_path: str) -> None:
    """
    Main EDA runner. Loads data, prints statistics, and saves all charts.

    Args:
        csv_path: Path to employee_data.csv
    """
    print("\n🔍 Running Exploratory Data Analysis...")
    df = pd.read_csv(csv_path)

    print_key_statistics(df)

    print("\n  Generating visualizations...")
    plot_attrition_by_department(df)
    plot_attrition_by_overtime(df)
    plot_income_vs_attrition(df)
    plot_correlation_heatmap(df)
    plot_job_satisfaction_violin(df)
    plot_age_distribution(df)

    print(f"\n✅ All charts saved to: visuals/")


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "../data/employee_data.csv"
    run_eda(csv)
