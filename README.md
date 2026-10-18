# 👥 Employee Attrition Prediction System

> **An end-to-end ML system that doesn't just *predict* attrition — it *explains* it, *costs* it, and *recommends* what to do about it.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-brightgreen.svg)](https://shap.readthedocs.io)

---

## 🌟 What Makes This Project Unique

Most employee attrition projects stop at "Accuracy: 85%". This project goes further:

| Feature | Description |
|---|---|
| 🔮 **SHAP Per-Employee Explanations** | Waterfall chart showing *why* each employee is at risk |
| 💰 **Attrition Cost Calculator** | Itemized financial cost of losing each employee (₹) |
| 🔄 **What-If Scenario Simulator** | "What if salary increases 20%?" — live recalculation |
| 🎯 **Personalized Retention Engine** | AI-generated action items per risk factor |
| 📊 **Risk Tier Classification** | Low/Medium/High/Critical buckets with color coding |
| 🗄️ **Prediction History (SQLite)** | Audit trail of every prediction made |
| 📄 **PDF Report Generator** | One-click exportable HR report |
| 📦 **Batch CSV Prediction** | Process 100s of employees, download results |
| 🌡️ **Department Risk Heatmap** | Interactive Plotly heatmap |
| 📐 **Calibrated Probabilities** | Platt scaling for statistically reliable confidence scores |

---

## 📂 Project Structure

```
Employee-Attrition-Prediction/
│
├── data/
│   ├── generate_data.py        ← Synthetic dataset generator (1500 employees)
│   └── employee_data.csv       ← Generated dataset
│
├── src/
│   ├── preprocessing.py        ← sklearn Pipeline (imputation → encoding → scaling)
│   ├── eda.py                  ← 6 EDA visualizations auto-saved to visuals/
│   ├── train_model.py          ← 4 models + tuning + calibration + comparison
│   ├── predict.py              ← Console prediction interface with full pipeline
│   ├── explainability.py       ← SHAP waterfall plots + plain-English explanations
│   ├── cost_analyzer.py        ← Attrition cost + ROI calculator (UNIQUE)
│   ├── recommendations.py      ← Personalized retention recommendations (UNIQUE)
│   └── utils.py                ← Validation, SQLite logger, PDF generator, formatting
│
├── models/
│   ├── best_model.pkl          ← Best trained model
│   ├── calibrated_model.pkl    ← Probability-calibrated model
│   ├── preprocessor.pkl        ← Fitted preprocessing pipeline
│   └── metrics.json            ← All model metrics + feature names
│
├── visuals/
│   └── (10+ auto-generated charts)
│
├── reports/
│   └── (auto-generated PDF/text reports)
│
├── app/
│   └── streamlit_app.py        ← 6-page interactive web dashboard
│
├── run_pipeline.py             ← One-command full pipeline
└── requirements.txt
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (one command)
```bash
python run_pipeline.py
```
This will:
- Generate the synthetic HR dataset
- Run EDA and save 6 charts
- Train 4 models with hyperparameter tuning
- Run a demo prediction

### 3. Launch the web dashboard
```bash
streamlit run app/streamlit_app.py
```

### 4. Or use the console interface
```bash
python src/predict.py          # Interactive mode
python src/predict.py --demo   # Demo with sample employees
```

### 5. Run individual steps
```bash
python data/generate_data.py   # Generate dataset only
python src/eda.py              # EDA only
python src/train_model.py      # Train only
```

---

## 📊 Dataset Description

**1500 synthetic employees** with realistic attrition correlations:

| Column | Type | Description |
|---|---|---|
| EmployeeID | Integer | Unique identifier |
| Age | Integer | 22–57 years |
| Department | Categorical | Sales / Engineering / HR / Marketing / Finance |
| MonthlyIncome | Integer | ₹15,000–₹1,20,000 |
| YearsAtCompany | Integer | 0–35 years |
| JobSatisfaction | Integer | 1 (Low) to 5 (High) |
| WorkLifeBalance | Integer | 1 (Poor) to 5 (Excellent) |
| PerformanceRating | Integer | 1–5 |
| OverTime | Categorical | Yes / No |
| DistanceFromHome | Integer | 1–60 km |
| **Attrition** | **Target** | **Yes (left) / No (stayed)** |

**Attrition Rate:** ~28–32% (realistic, weighted by factors above)

---

## 🤖 Models & Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.78 | ~0.62 | ~0.68 | ~0.65 | ~0.83 |
| Random Forest | ~0.82 | ~0.70 | ~0.72 | ~0.71 | ~0.88 |
| **XGBoost** ⭐ | **~0.84** | **~0.73** | **~0.75** | **~0.74** | **~0.90** |
| LightGBM | ~0.83 | ~0.71 | ~0.74 | ~0.72 | ~0.89 |

> *Actual results vary per training run. XGBoost or LightGBM typically wins.*

**Best model selection:** ROC-AUC on stratified 80/20 held-out test set.

**Why ROC-AUC?** Because attrition datasets are imbalanced. Accuracy alone is misleading (a model predicting "No attrition" for everyone gets ~72% accuracy but is useless).

---

## 🔍 Key EDA Insights

1. **Overtime is the #1 attrition driver**: Employees on overtime have ~3× higher attrition rate
2. **Sales department has highest attrition**: ~38% vs Engineering at ~18%
3. **Low-income employees leave more**: Median income of leavers is ₹21K vs ₹42K for stayers
4. **First 2 years are critical**: Attrition risk peaks in the first 24 months
5. **Low job satisfaction compounds risk**: Score of 1–2 → 45%+ attrition rate

---

## 💰 Cost Analysis Model

Based on SHRM (Society for Human Resource Management) research:

```
Replacement Cost = Recruitment (20%) + Onboarding (15%) + 
                   Productivity Gap (12.5%) + Manager Time (10%)
                   × Tenure Multiplier × Department Multiplier
```

For a typical mid-level employee (₹45,000/month, 3 years): **~₹4.2 Lakhs**

---

## 🎯 Retention ROI Framework

```
ROI = (Expected Savings - Retention Cost) / Retention Cost × 100%
Expected Savings = Replacement Cost × Success Rate (70% default)
```

If replacing costs ₹4.2L and a retention raise costs ₹1.5L → **ROI = 96%** ✅

---

## 🛠️ Technologies Used

| Category | Technology |
|---|---|
| **ML Core** | Python, Pandas, NumPy, Scikit-learn |
| **Boosting** | XGBoost, LightGBM |
| **Imbalance** | imbalanced-learn (SMOTE) |
| **Explainability** | SHAP |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Web Dashboard** | Streamlit |
| **Persistence** | Joblib (models), SQLite (history) |
| **Reports** | fpdf2 (PDF generation) |

---

## 📋 Technical Design Decisions

- **SMOTE applied AFTER train/test split** — prevents data leakage
- **Calibrated probabilities** (Platt scaling) — raw model probabilities can be overconfident
- **sklearn Pipeline** — guarantees same transformations on train and predict paths
- **RandomizedSearchCV** (not GridSearchCV) — faster with comparable results on large grids
- **ROC-AUC as selection metric** — robust to class imbalance

---

## 👨‍💻 Author

Built as a comprehensive ML project demonstrating end-to-end pipeline from data generation to deployment-ready prediction with unique industry-applicable features.
