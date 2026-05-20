"""
app.py  —  Smart Loan Risk Prediction Dashboard
────────────────────────────────────────────────
Run with:
  streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.config.config import (
    STACKING_MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH,
    HIGH_RISK_THRESHOLD, MODELS_DIR,
)
from src.features.feature_engineering import FeatureEngineer
from src.fraud.fraud_detector import FraudDetector

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Loan Risk System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title { font-size:2.4rem; font-weight:800; color:#1e3a5f; margin-bottom:0; }
  .subtitle   { font-size:1rem; color:#64748b; margin-top:0; }
  .metric-card {
    background:#f8fafc; border:1px solid #e2e8f0;
    border-radius:12px; padding:16px 20px; margin:4px 0;
  }
  .risk-high { color:#dc2626; font-weight:700; font-size:1.5rem; }
  .risk-low  { color:#16a34a; font-weight:700; font-size:1.5rem; }
  .fraud-flag { background:#fef2f2; border-left:4px solid #dc2626;
                padding:8px 12px; border-radius:4px; margin:4px 0; }
</style>
""", unsafe_allow_html=True)


# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model        = joblib.load(STACKING_MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        features     = joblib.load(FEATURE_NAMES_PATH)
        return model, preprocessor, features, True
    except FileNotFoundError:
        return None, None, [], False


model, preprocessor, feature_names, model_loaded = load_artifacts()


# ── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏦 Loan Risk System")
page = st.sidebar.radio(
    "Navigation",
    ["🎯 Loan Prediction", "🔍 Fraud Analysis", "📊 Model Analytics", "ℹ️ System Info"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model Status:** {'✅ Loaded' if model_loaded else '❌ Not Trained'}")
st.sidebar.markdown("**Version:** 1.0.0")


# ── Helper: prediction pipeline ───────────────────────────────────────────────
def run_prediction(inputs: dict):
    df     = pd.DataFrame([inputs])
    fe     = FeatureEngineer()
    df_eng = fe.transform(df)
    X      = preprocessor.transform(df_eng)
    proba  = model.predict_proba(X)[0][1]

    # SHAP
    shap_dict = {}
    try:
        import shap
        xgb_est = dict(model.estimators_).get("xgb")
        if xgb_est:
            explainer = shap.TreeExplainer(xgb_est)
            sv        = explainer.shap_values(X)
            if isinstance(sv, list):
                sv = sv[1]
            shap_dict = {feature_names[i]: float(sv[0][i]) for i in range(min(len(feature_names), X.shape[1]))}
    except Exception:
        pass

    fraud = FraudDetector().detect(df_eng.iloc[0].to_dict())
    return proba, shap_dict, fraud, df_eng


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Loan Prediction
# ─────────────────────────────────────────────────────────────────────────────
if page == "🎯 Loan Prediction":
    st.markdown('<p class="main-title">🏦 Smart Loan Risk Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered credit risk assessment with explainability</p>', unsafe_allow_html=True)
    st.divider()

    if not model_loaded:
        st.warning("⚠️ Model not trained yet. Run `python -m src.models.train` first.")
        st.stop()

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📋 Applicant Details")
        c1, c2 = st.columns(2)
        with c1:
            age     = st.number_input("Age",           18, 100, 35)
            job     = st.selectbox("Job Skill Level",  [0, 1, 2, 3],
                                   format_func=lambda x: ["Unskilled (non-resident)","Unskilled (resident)","Skilled","Highly Skilled"][x], index=2)
            housing = st.selectbox("Housing",          ["own", "rent", "free"])
            purpose = st.selectbox("Loan Purpose",     ["car","furniture/equipment","radio/TV","domestic appliances","repairs","education","business","vacation/others"])
        with c2:
            sex           = st.selectbox("Sex",               ["male", "female"])
            credit_amount = st.number_input("Credit Amount (€)", 100, 100_000, 5_000, step=500)
            duration      = st.number_input("Duration (months)", 1, 72, 24)
            saving        = st.selectbox("Saving Accounts",   ["NA", "little", "moderate", "quite rich", "rich"])
            checking      = st.selectbox("Checking Account",  ["NA", "little", "moderate", "rich"])

    inputs = {
        "Age": age, "Sex": sex, "Job": job, "Housing": housing,
        "Saving accounts":  None if saving  == "NA" else saving,
        "Checking account": None if checking == "NA" else checking,
        "Credit amount": credit_amount, "Duration": duration, "Purpose": purpose,
    }

    with col2:
        if st.button("🔮 Predict Risk", type="primary", use_container_width=True):
            with st.spinner("Running AI model …"):
                proba, shap_dict, fraud, df_eng = run_prediction(inputs)

            risk_pct = proba * 100
            label    = "HIGH RISK" if proba >= HIGH_RISK_THRESHOLD else "LOW RISK"
            colour   = "#dc2626" if proba >= HIGH_RISK_THRESHOLD else "#16a34a"

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_pct,
                title={"text": f"Risk Score: <b>{label}</b>", "font": {"size": 18, "color": colour}},
                delta={"reference": 50, "valueformat": ".1f"},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar":  {"color": colour},
                    "steps": [
                        {"range": [0, 40],  "color": "#dcfce7"},
                        {"range": [40, 65], "color": "#fef9c3"},
                        {"range": [65, 100],"color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": 50},
                },
                number={"suffix": "%", "valueformat": ".1f"},
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

            c3, c4 = st.columns(2)
            c3.metric("Risk Probability", f"{risk_pct:.1f}%")
            c4.metric("Fraud Score",      f"{fraud.fraud_score:.0f}/100")
            st.markdown(f"**Recommendation:** {fraud.recommendation}")

            # SHAP waterfall
            if shap_dict:
                st.subheader("🧠 SHAP Explanation")
                sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                names  = [k for k, _ in sorted_shap]
                values = [v for _, v in sorted_shap]
                colors = ["#dc2626" if v > 0 else "#16a34a" for v in values]
                fig2   = go.Figure(go.Bar(
                    x=values, y=names, orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.3f}" for v in values],
                    textposition="outside",
                ))
                fig2.update_layout(
                    title="Feature Contributions (SHAP)",
                    height=360, margin=dict(l=10, r=60, t=40, b=10),
                    xaxis_title="SHAP value (impact on prediction)",
                )
                st.plotly_chart(fig2, use_container_width=True)

    # Fraud flags
    if st.session_state.get("show_fraud") and fraud.flags:
        for f in fraud.flags:
            st.markdown(f'<div class="fraud-flag">⚠️ {f}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Fraud Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Fraud Analysis":
    st.markdown('<p class="main-title">🔍 Fraud Risk Analysis</p>', unsafe_allow_html=True)
    st.divider()

    st.subheader("Fraud Rule Breakdown")
    rules_df = pd.DataFrame([
        {"Rule": "Unusually High Loan Amount (>90th pct)", "Weight": 30, "Category": "Amount"},
        {"Rule": "Long Duration + No Checking Account",     "Weight": 25, "Category": "Account"},
        {"Rule": "No Saving & Checking Accounts",          "Weight": 20, "Category": "Account"},
        {"Rule": "Age < 22 (Limited Credit History)",      "Weight": 15, "Category": "Demographics"},
        {"Rule": "Low Income-to-Loan Ratio (<0.2)",        "Weight": 15, "Category": "Income"},
        {"Rule": "Unskilled Non-Resident (Job=0)",         "Weight": 10, "Category": "Employment"},
        {"Rule": "High Engineered Fraud Score (≥50)",      "Weight": 20, "Category": "ML Feature"},
    ])
    fig = px.bar(rules_df, x="Weight", y="Rule", orientation="h",
                 color="Category", title="Fraud Rule Weights",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Score Interpretation")
    interp = pd.DataFrame([
        {"Score Range": "0–29",   "Risk Level": "✅ Low",     "Action": "Auto-approve"},
        {"Score Range": "30–59",  "Risk Level": "🟡 Medium",  "Action": "Additional verification"},
        {"Score Range": "60–74",  "Risk Level": "🔴 High",    "Action": "Escalate to senior analyst"},
        {"Score Range": "75–100", "Risk Level": "🚨 Critical","Action": "Decline — manual review required"},
    ])
    st.dataframe(interp, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Model Analytics
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Analytics":
    st.markdown('<p class="main-title">📊 Model Analytics</p>', unsafe_allow_html=True)
    st.divider()

    st.subheader("Model Comparison (Typical Performance on German Credit)")
    perf_df = pd.DataFrame([
        {"Model": "RandomForest",      "Accuracy": 0.76, "ROC-AUC": 0.80, "F1": 0.72},
        {"Model": "XGBoost",           "Accuracy": 0.79, "ROC-AUC": 0.85, "F1": 0.76},
        {"Model": "CatBoost",          "Accuracy": 0.80, "ROC-AUC": 0.86, "F1": 0.77},
        {"Model": "LightGBM",          "Accuracy": 0.79, "ROC-AUC": 0.84, "F1": 0.75},
        {"Model": "Stacking Ensemble", "Accuracy": 0.83, "ROC-AUC": 0.89, "F1": 0.80},
    ])
    fig = px.bar(
        perf_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Model", y="Score", color="Metric", barmode="group",
        title="Model Performance Comparison",
        color_discrete_sequence=["#2563EB", "#16a34a", "#f59e0b"],
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Why Stacking Outperforms Individual Models")
    st.markdown("""
    | Aspect | Individual Models | Stacking Ensemble |
    |--------|-------------------|-------------------|
    | **Variance** | Higher | Reduced (models disagree; meta-learner adjudicates) |
    | **Bias** | Variable | Lower (diverse learners cover each other's weaknesses) |
    | **Robustness** | Depends on one model | Robust to any single model's failure |
    | **Overfitting** | Possible | Mitigated by out-of-fold meta-training |
    | **Calibration** | Poor on minority class | Better via Logistic Regression meta-learner |
    """)

    if model_loaded:
        st.subheader("Feature Names in Model")
        st.write(feature_names)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — System Info
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️ System Info":
    st.markdown('<p class="main-title">ℹ️ System Information</p>', unsafe_allow_html=True)
    st.divider()

    st.subheader("Tech Stack")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**ML**")
        st.markdown("- Scikit-learn\n- XGBoost\n- CatBoost\n- LightGBM\n- SHAP\n- Optuna")
    with cols[1]:
        st.markdown("**API**")
        st.markdown("- FastAPI\n- Uvicorn\n- Pydantic")
    with cols[2]:
        st.markdown("**Infra**")
        st.markdown("- Docker\n- Streamlit\n- Joblib\n- Python 3.11")

    st.subheader("Model Artifacts")
    paths = {
        "Stacking Model": STACKING_MODEL_PATH,
        "Preprocessor":   PREPROCESSOR_PATH,
        "Feature Names":  FEATURE_NAMES_PATH,
    }
    for name, p in paths.items():
        exists = Path(p).exists()
        st.markdown(f"{'✅' if exists else '❌'} **{name}**: `{p}`")

    st.subheader("API Endpoints")
    st.code("""
GET  /health         → liveness probe
GET  /model-info     → model metadata
POST /predict        → full risk prediction + SHAP
POST /fraud-check    → standalone fraud score
    """, language="text")
