"""
streamlit_app.py
────────────────
🖥️ Full Interactive Streamlit Web Dashboard for Employee Attrition Prediction

Features:
  1. Individual Employee Prediction with SHAP waterfall
  2. Batch CSV Prediction with downloadable results
  3. What-If Scenario Simulator (unique!)
  4. Department Risk Heatmap (Plotly)
  5. Prediction History viewer
  6. Cost Analysis & Recommendations
  7. Model Performance Dashboard

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import io
import json
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Path setup ────────────────────────────────────────────────────────────────
APP_DIR  = os.path.dirname(__file__)
BASE_DIR = os.path.dirname(APP_DIR)
SRC_DIR  = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from preprocessing   import preprocess_single_input, load_preprocessor
from cost_analyzer   import calculate_replacement_cost, calculate_retention_roi
from recommendations import generate_recommendations, format_recommendations_text
from utils           import (validate_input, sanitize_input, get_risk_tier,
                              log_prediction, get_prediction_history,
                              generate_pdf_report)
from explainability  import (compute_shap_values, explain_in_plain_english,
                              HAS_SHAP)

import joblib

MODELS_DIR = os.path.join(BASE_DIR, "models")
DEPARTMENTS = ["Sales", "Engineering", "HR", "Marketing", "Finance"]

# ══════════════════════════════════════════════════════════════════════════════
#  Page Configuration
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title  = "Employee Attrition Predictor",
    page_icon   = "👥",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        color: white; text-align: center;
    }
    .risk-critical { background: #e74c3c; color: white; padding: 1rem;
                     border-radius: 8px; text-align: center; font-size: 1.3rem; }
    .risk-high     { background: #e67e22; color: white; padding: 1rem;
                     border-radius: 8px; text-align: center; font-size: 1.3rem; }
    .risk-medium   { background: #f1c40f; color: #2c3e50; padding: 1rem;
                     border-radius: 8px; text-align: center; font-size: 1.3rem; }
    .risk-low      { background: #2ecc71; color: white; padding: 1rem;
                     border-radius: 8px; text-align: center; font-size: 1.3rem; }
    .metric-card   { background: #f8f9fa; border-left: 4px solid #3498db;
                     padding: 1rem; border-radius: 6px; margin: 0.5rem 0; }
    .rec-card      { background: #fff3cd; border-left: 4px solid #ffc107;
                     padding: 0.8rem; border-radius: 6px; margin: 0.3rem 0; }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6; border-radius: 8px; padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Model Loading (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_artifacts():
    """Load and cache the model + preprocessor (runs only once per session)."""
    try:
        preprocessor = load_preprocessor(MODELS_DIR)

        calibrated_path = os.path.join(MODELS_DIR, "calibrated_model.pkl")
        best_path       = os.path.join(MODELS_DIR, "best_model.pkl")

        if os.path.exists(calibrated_path):
            model = joblib.load(calibrated_path)
        elif os.path.exists(best_path):
            model = joblib.load(best_path)
        else:
            return None, None, [], None

        feature_names = []
        metrics_data  = {}
        metrics_path  = os.path.join(MODELS_DIR, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                data          = json.load(f)
                feature_names = data.get("feature_names", [])
                metrics_data  = data

        return model, preprocessor, feature_names, metrics_data
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, [], {}


# ══════════════════════════════════════════════════════════════════════════════
#  Helper: Run prediction for one employee dict
# ══════════════════════════════════════════════════════════════════════════════

def run_prediction(employee: dict, model, preprocessor):
    """Run model prediction and return label, probability."""
    X         = preprocess_single_input(employee, preprocessor)
    label     = model.predict(X)[0]
    prob      = float(model.predict_proba(X)[0][1])
    label_str = "Yes" if label == 1 else "No"
    return label_str, prob, X


def risk_color(tier: str) -> str:
    if "CRITICAL" in tier: return "risk-critical"
    if "HIGH"     in tier: return "risk-high"
    if "MEDIUM"   in tier: return "risk-medium"
    return "risk-low"


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/color/96/employee-card.png", width=80)
    st.title("Navigation")
    page = st.radio("Go to", [
        "🔮 Individual Prediction",
        "📦 Batch Prediction",
        "🔄 What-If Simulator",
        "🌡️ Department Heatmap",
        "📊 Model Performance",
        "🗄️ Prediction History"
    ])
    st.markdown("---")
    st.markdown("**Model Status**")
    model, preprocessor, feature_names, metrics_data = load_artifacts()
    if model is not None:
        st.success("✅ Model loaded")
        best = metrics_data.get("best_model", "Unknown")
        st.info(f"Best model: **{best}**")
    else:
        st.error("❌ No model found. Run train_model.py first.")


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>👥 Employee Attrition Prediction System</h1>
    <p>AI-powered HR analytics — Predict · Explain · Recommend · Calculate ROI</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model not loaded. Please run the training pipeline first.")
    st.code("python src/train_model.py")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1: Individual Prediction
# ══════════════════════════════════════════════════════════════════════════════

if page == "🔮 Individual Prediction":
    st.header("🔮 Individual Employee Attrition Prediction")
    st.markdown("Enter employee details to predict attrition risk with full explanation.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Basic Info")
        age        = st.slider("Age", 18, 65, 30)
        dept       = st.selectbox("Department", DEPARTMENTS)
        income     = st.number_input("Monthly Income (₹)", 5000, 300000, 35000, step=1000)
        years      = st.slider("Years at Company", 0, 40, 3)

    with col2:
        st.subheader("😊 Satisfaction Scores")
        job_sat  = st.slider("Job Satisfaction (1–5)", 1, 5, 3)
        wlb      = st.slider("Work-Life Balance (1–5)", 1, 5, 3)
        perf     = st.slider("Performance Rating (1–5)", 1, 5, 3)

    with col3:
        st.subheader("🏢 Work Conditions")
        overtime = st.radio("Works Overtime?", ["Yes", "No"], index=1)
        distance = st.slider("Distance from Home (km)", 0, 100, 10)
        emp_name = st.text_input("Employee Name (optional)", "")

    employee = {
        "Age": age, "Department": dept, "MonthlyIncome": income,
        "YearsAtCompany": years, "JobSatisfaction": job_sat,
        "WorkLifeBalance": wlb, "PerformanceRating": perf,
        "OverTime": overtime, "DistanceFromHome": distance
    }

    if st.button("🚀 Predict Attrition Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing employee profile..."):
            label_str, prob, X = run_prediction(employee, model, preprocessor)
            risk_tier, risk_advice = get_risk_tier(prob)

        st.markdown("---")

        # ── Result metrics ─────────────────────────────────────────────────────
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("🎯 Prediction",   label_str)
        r2.metric("📊 Probability",  f"{prob*100:.1f}%")
        r3.metric("🔥 Risk Tier",    risk_tier.split()[1] if len(risk_tier.split()) > 1 else risk_tier)
        r4.metric("💡 Advice",       risk_advice[:25] + "...")

        # Color risk box
        css_class = risk_color(risk_tier)
        st.markdown(f'<div class="{css_class}">⚠️ {risk_tier} — {risk_advice}</div>',
                    unsafe_allow_html=True)
        st.markdown("")

        # ── Gauge chart ────────────────────────────────────────────────────────
        gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = prob * 100,
            title = {"text": "Attrition Probability (%)"},
            gauge = {
                "axis":  {"range": [0, 100]},
                "bar":   {"color": "#e74c3c" if prob > 0.5 else "#2ecc71"},
                "steps": [
                    {"range": [0, 20],  "color": "#d5f5e3"},
                    {"range": [20, 40], "color": "#fef9e7"},
                    {"range": [40, 60], "color": "#fdebd0"},
                    {"range": [60, 80], "color": "#fadbd8"},
                    {"range": [80, 100],"color": "#f1948a"},
                ],
                "threshold": {"value": 50, "line": {"color": "black", "width": 3}}
            }
        ))
        gauge.update_layout(height=300, margin=dict(t=50, b=10))
        st.plotly_chart(gauge, use_container_width=True)

        # ── Tabs for details ───────────────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs(["🔍 SHAP Explanation", "💰 Cost Analysis", "🎯 Recommendations"])

        with tab1:
            if HAS_SHAP and feature_names:
                try:
                    _, shap_values = compute_shap_values(model, X, feature_names)
                    if shap_values is not None:
                        sv = shap_values[0]
                        fv = X[0]

                        # Waterfall-style bar chart in Plotly
                        indices     = np.argsort(np.abs(sv))[::-1][:10]
                        top_features = [feature_names[i] for i in indices]
                        top_shap    = sv[indices]
                        colors      = ["#e74c3c" if v > 0 else "#2ecc71" for v in top_shap]

                        fig_shap = go.Figure(go.Bar(
                            x    = top_shap[::-1],
                            y    = top_features[::-1],
                            orientation = "h",
                            marker_color = colors[::-1]
                        ))
                        fig_shap.update_layout(
                            title  = "SHAP Values — Why This Prediction?",
                            xaxis_title = "SHAP Value (Red = increases risk, Green = reduces risk)",
                            height = 400
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)

                        explanation = explain_in_plain_english(sv, feature_names, fv)
                        st.code(explanation)
                except Exception as e:
                    st.warning(f"SHAP explanation unavailable: {e}")
            else:
                st.info("Install SHAP for per-prediction explanations: `pip install shap`")

        with tab2:
            cost_report = calculate_replacement_cost(income, dept, years)
            roi         = calculate_retention_roi(
                cost_report.total_replacement_cost,
                income * 3, 0.70
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("💼 Recruitment",     f"₹{cost_report.recruitment_cost:,.0f}")
            c2.metric("📚 Onboarding",      f"₹{cost_report.onboarding_cost:,.0f}")
            c3.metric("📉 Productivity Gap", f"₹{cost_report.productivity_loss:,.0f}")

            st.metric(
                "⚠️ TOTAL REPLACEMENT COST",
                f"₹{cost_report.total_replacement_cost:,.0f}",
                delta=f"Risk Category: {cost_report.cost_category}"
            )

            # ROI chart
            roi_fig = go.Figure(go.Bar(
                x      = ["Retention Cost", "Expected Savings", "Net Benefit"],
                y      = [roi["retention_cost"], roi["expected_savings"], roi["net_benefit"]],
                marker_color = ["#3498db", "#2ecc71",
                                "#2ecc71" if roi["net_benefit"] > 0 else "#e74c3c"]
            ))
            roi_fig.update_layout(title="Retention Program ROI Analysis", height=300)
            st.plotly_chart(roi_fig, use_container_width=True)

            verdict = "✅ WORTH IT" if roi["worthwhile"] else "❌ NOT RECOMMENDED"
            st.success(f"ROI: **{roi['roi_percent']:.1f}%** — {verdict}")

        with tab3:
            recs = generate_recommendations(employee, prob)
            if recs:
                st.markdown("#### 🎯 Personalized Actionable Retention Plan")
                for rec in recs:
                    badge_style = {
                        "Critical": {"bg": "#dc2626", "text": "#ffffff"},
                        "High":     {"bg": "#ea580c", "text": "#ffffff"},
                        "Medium":   {"bg": "#d97706", "text": "#ffffff"},
                        "Low":      {"bg": "#16a34a", "text": "#ffffff"}
                    }.get(rec.priority, {"bg": "#475569", "text": "#ffffff"})

                    st.markdown(f"""
                    <div style="background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 10px; padding: 1.1rem; margin-bottom: 0.9rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);">
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                            <span style="font-size: 1.05rem; font-weight: 700; color: #ffffff;">{rec.emoji} {rec.category}</span>
                            <span style="background-color: {badge_style['bg']}; color: {badge_style['text']}; font-size: 0.75rem; font-weight: 700; padding: 0.25rem 0.65rem; border-radius: 9999px; letter-spacing: 0.05em; text-transform: uppercase;">
                                {rec.priority}
                            </span>
                        </div>
                        <div style="background-color: #0f172a; border-left: 4px solid #38bdf8; border-radius: 6px; padding: 0.75rem 0.9rem; margin-bottom: 0.5rem;">
                            <div style="font-size: 0.75rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; margin-bottom: 0.2rem;">⚡ Recommended Action</div>
                            <div style="font-size: 0.92rem; color: #f1f5f9; line-height: 1.45;">{rec.action}</div>
                        </div>
                        <div style="background-color: #0f172a; border-left: 4px solid #4ade80; border-radius: 6px; padding: 0.75rem 0.9rem;">
                            <div style="font-size: 0.75rem; font-weight: 700; color: #4ade80; text-transform: uppercase; margin-bottom: 0.2rem;">📈 Expected Impact</div>
                            <div style="font-size: 0.92rem; color: #e2e8f0; line-height: 1.45;">{rec.expected_impact}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No critical risk factors detected.")

        # ── Log + PDF ──────────────────────────────────────────────────────────
        try:
            log_prediction(employee, label_str, prob, risk_tier,
                           cost_report.total_replacement_cost)
        except Exception:
            pass

        if st.button("📄 Download PDF Report"):
            recs_text  = format_recommendations_text(recs, emp_name or "Employee")
            report_path = generate_pdf_report(
                employee, label_str, prob, risk_tier,
                cost_report.total_replacement_cost, recs_text
            )
            with open(report_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Report", f.read(),
                    file_name=os.path.basename(report_path),
                    mime="application/pdf"
                )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2: Batch Prediction
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📦 Batch Prediction":
    st.header("📦 Batch Employee Attrition Prediction")
    st.markdown("Upload a CSV of employees and get predictions for all of them at once.")

    st.info("""
    **Required CSV columns:**
    Age, Department, MonthlyIncome, YearsAtCompany, JobSatisfaction,
    WorkLifeBalance, PerformanceRating, OverTime, DistanceFromHome
    """)

    uploaded = st.file_uploader("Upload Employee CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"**Loaded {len(df)} employees**")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", type="primary"):
            with st.spinner(f"Processing {len(df)} employees..."):
                results = []
                for _, row in df.iterrows():
                    try:
                        emp = sanitize_input(row.to_dict())
                        X   = preprocess_single_input(emp, preprocessor)
                        lbl = model.predict(X)[0]
                        prb = float(model.predict_proba(X)[0][1])
                        tier, _ = get_risk_tier(prb)
                        cost = calculate_replacement_cost(
                            emp["MonthlyIncome"], emp["Department"], emp["YearsAtCompany"]
                        ).total_replacement_cost
                        results.append({
                            **row.to_dict(),
                            "Predicted_Attrition": "Yes" if lbl == 1 else "No",
                            "Attrition_Probability": round(prb, 4),
                            "Risk_Tier": tier,
                            "Replacement_Cost_INR": round(cost, 0)
                        })
                    except Exception as e:
                        results.append({**row.to_dict(), "Error": str(e)})

            results_df = pd.DataFrame(results)

            # Summary stats
            s1, s2, s3, s4 = st.columns(4)
            n_yes = (results_df["Predicted_Attrition"] == "Yes").sum()
            s1.metric("Total Employees",    len(results_df))
            s2.metric("At-Risk Employees",  n_yes)
            s3.metric("Attrition Rate",     f"{n_yes/len(results_df)*100:.1f}%")
            s4.metric("Total Risk Cost",    f"₹{results_df['Replacement_Cost_INR'].sum()/1e6:.1f}M")

            # Risk distribution pie chart
            risk_counts = results_df["Risk_Tier"].value_counts()
            pie_fig = px.pie(
                names  = risk_counts.index,
                values = risk_counts.values,
                title  = "Risk Tier Distribution",
                color_discrete_sequence = px.colors.qualitative.Set2
            )
            st.plotly_chart(pie_fig, use_container_width=True)

            # Full results table
            st.dataframe(results_df, use_container_width=True)

            # Download
            csv_out = results_df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Results CSV",
                csv_out, "attrition_predictions.csv", "text/csv"
            )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3: What-If Simulator
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔄 What-If Simulator":
    st.header("🔄 What-If Scenario Simulator")
    st.markdown("""
    Explore how changing specific factors affects attrition risk.
    *"What if we gave this employee a raise? Would they stay?"*
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Base Employee Profile")
        base_age    = st.slider("Age",               18, 65, 27)
        base_dept   = st.selectbox("Department", DEPARTMENTS)
        base_income = st.number_input("Monthly Income (₹)", 5000, 300000, 22000, 1000)
        base_years  = st.slider("Years at Company",  0, 40, 1)
        base_jsat   = st.slider("Job Satisfaction",  1, 5, 2)
        base_wlb    = st.slider("Work-Life Balance", 1, 5, 2)
        base_perf   = st.slider("Performance Rating",1, 5, 4)
        base_ot     = st.radio("OverTime", ["Yes", "No"])
        base_dist   = st.slider("Distance (km)",     0, 100, 35)

    base_emp = {
        "Age": base_age, "Department": base_dept, "MonthlyIncome": base_income,
        "YearsAtCompany": base_years, "JobSatisfaction": base_jsat,
        "WorkLifeBalance": base_wlb, "PerformanceRating": base_perf,
        "OverTime": base_ot, "DistanceFromHome": base_dist
    }

    with col2:
        st.subheader("🔧 Change These Factors")
        new_income   = st.number_input("New Monthly Income (₹)",    5000, 300000, base_income + 10000, 1000)
        new_ot       = st.radio("New OverTime Status",              ["Yes", "No"], index=1)
        new_jsat     = st.slider("New Job Satisfaction",            1, 5, min(base_jsat + 1, 5))
        new_wlb      = st.slider("New Work-Life Balance",           1, 5, min(base_wlb + 1, 5))
        new_dist     = st.slider("New Distance from Home (km)",     0, 100, base_dist)

    new_emp = {
        **base_emp,
        "MonthlyIncome": new_income, "OverTime": new_ot,
        "JobSatisfaction": new_jsat, "WorkLifeBalance": new_wlb,
        "DistanceFromHome": new_dist
    }

    if st.button("⚡ Compare Scenarios", type="primary", use_container_width=True):
        _, base_prob, _ = run_prediction(base_emp, model, preprocessor)
        _, new_prob,  _ = run_prediction(new_emp,  model, preprocessor)

        delta     = new_prob - base_prob
        delta_pct = delta * 100

        b1, b2, b3 = st.columns(3)
        b1.metric("📊 Base Attrition Risk",  f"{base_prob*100:.1f}%")
        b2.metric("✨ New Attrition Risk",    f"{new_prob*100:.1f}%",
                  delta=f"{delta_pct:+.1f}%",
                  delta_color="inverse")
        verdict = "🎉 Employee more likely to STAY!" if delta < 0 else "⚠️ Risk has INCREASED!"
        b3.metric("Outcome", verdict)

        # Side-by-side gauge
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=base_prob*100,
            title={"text": "Before"},
            domain={"x": [0, 0.45], "y": [0, 1]},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#e74c3c" if base_prob > 0.5 else "#2ecc71"}}
        ))
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=new_prob*100,
            title={"text": "After"},
            domain={"x": [0.55, 1], "y": [0, 1]},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#e74c3c" if new_prob > 0.5 else "#2ecc71"}}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Sensitivity: scan salary range
        st.subheader("💰 Salary Sensitivity Analysis")
        salaries = range(15000, 100001, 5000)
        probs    = []
        for sal in salaries:
            test_emp = {**base_emp, "MonthlyIncome": sal}
            _, p, _ = run_prediction(test_emp, model, preprocessor)
            probs.append(p * 100)

        sens_fig = px.line(
            x    = list(salaries),
            y    = probs,
            labels = {"x": "Monthly Income (₹)", "y": "Attrition Probability (%)"},
            title  = "How Salary Affects Attrition Risk"
        )
        sens_fig.add_hline(y=50, line_dash="dash", line_color="red",
                           annotation_text="50% threshold")
        st.plotly_chart(sens_fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4: Department Heatmap
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🌡️ Department Heatmap":
    st.header("🌡️ Department Risk Heatmap")
    st.markdown("Interactive heatmap showing average attrition risk by department and other factors.")

    # Generate grid of predictions
    depts       = DEPARTMENTS
    sat_levels  = [1, 2, 3, 4, 5]
    ot_options  = ["Yes", "No"]

    selected_ot = st.radio("OverTime filter:", ot_options, horizontal=True)

    with st.spinner("Computing risk matrix..."):
        matrix = []
        for dept in depts:
            row = []
            for sat in sat_levels:
                avg_emp = {
                    "Age": 32, "Department": dept, "MonthlyIncome": 40000,
                    "YearsAtCompany": 4, "JobSatisfaction": sat,
                    "WorkLifeBalance": 3, "PerformanceRating": 3,
                    "OverTime": selected_ot, "DistanceFromHome": 15
                }
                _, prob, _ = run_prediction(avg_emp, model, preprocessor)
                row.append(round(prob * 100, 1))
            matrix.append(row)

    heatmap_df = pd.DataFrame(matrix, index=depts,
                               columns=[f"Satisfaction {s}" for s in sat_levels])

    fig_heat = px.imshow(
        heatmap_df,
        text_auto     = True,
        color_continuous_scale = "RdYlGn_r",
        title         = f"Attrition Risk % by Department × Job Satisfaction (OverTime: {selected_ot})",
        labels        = {"color": "Risk %"}
    )
    fig_heat.update_layout(height=450)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("""
    > **Reading this chart:** Darker red = higher attrition risk.
    > Use this to identify which department × satisfaction combinations need urgent HR intervention.
    """)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5: Model Performance
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Model Performance":
    st.header("📊 Model Performance Dashboard")

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        st.warning("No metrics found. Train the model first: `python src/train_model.py`")
    else:
        with open(metrics_path) as f:
            data = json.load(f)

        all_metrics   = data.get("all_metrics", [])
        best_model    = data.get("best_model", "")

        df_metrics = pd.DataFrame(all_metrics).set_index("Model")

        st.subheader(f"🏆 Best Model: {best_model}")

        # Metrics table
        st.dataframe(
            df_metrics.style.highlight_max(axis=0, color="#d5f5e3")
                            .format("{:.4f}"),
            use_container_width=True
        )

        # Radar chart comparison
        categories = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
        fig_radar = go.Figure()
        colors    = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

        for i, (_, row) in enumerate(df_metrics.iterrows()):
            vals   = [row[c] for c in categories]
            vals  += vals[:1]   # Close the radar polygon
            cats   = categories + [categories[0]]
            marker = "🏆 " if row.name == best_model else ""
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats,
                name=f"{marker}{row.name}",
                line={"color": colors[i % len(colors)], "width": 2},
                fill="toself", opacity=0.3
            ))

        fig_radar.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 1]}},
            title="Model Performance Radar Chart",
            height=500
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Saved charts
        st.subheader("📈 Saved Visualizations")
        visuals_dir = os.path.join(BASE_DIR, "visuals")
        if os.path.exists(visuals_dir):
            charts = sorted([f for f in os.listdir(visuals_dir) if f.endswith(".png")])
            for i in range(0, len(charts), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(charts):
                        img_path = os.path.join(visuals_dir, charts[i + j])
                        col.image(img_path, caption=charts[i + j], use_column_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 6: Prediction History
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🗄️ Prediction History":
    st.header("🗄️ Prediction History (Audit Log)")
    st.markdown("All predictions are logged to SQLite for traceability and HR reporting.")

    try:
        history = get_prediction_history(limit=50)
        if history:
            df_hist = pd.DataFrame(history)
            st.write(f"**{len(df_hist)} recent predictions:**")
            st.dataframe(df_hist, use_container_width=True)

            # Summary
            h1, h2 = st.columns(2)
            h1.metric("Total Logged", len(df_hist))
            yes_count = (df_hist["predicted_label"] == "Yes").sum()
            h2.metric("High-Risk Predictions", yes_count)

            # Download history
            csv = df_hist.to_csv(index=False).encode()
            st.download_button("⬇️ Export History CSV", csv,
                               "prediction_history.csv", "text/csv")
        else:
            st.info("No predictions logged yet. Make some predictions first!")
    except Exception as e:
        st.error(f"Could not load history: {e}")
