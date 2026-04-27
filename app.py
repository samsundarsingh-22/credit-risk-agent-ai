import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from agent import CreditRiskAgent   # ✅ FIX 1

# ============================================================
# STYLE (UI POLISH)
# ============================================================

st.markdown("""
<style>
.main {background-color: #f5f7fb;}
h1, h2, h3 {color: #1f3b73;}
.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}
.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL FILES
# ============================================================

model = joblib.load("credit_model.pkl")
calibrator = joblib.load("calibrator.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(model)

# ============================================================
# AGENT
# ============================================================

agent = CreditRiskAgent(model, calibrator, scaler, feature_columns)

# ============================================================
# HEADER
# ============================================================

st.markdown("# 🏦 Agentic AI Credit Risk System")
st.caption("AI-powered risk scoring • Explainable decisions • Expected loss modeling")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🔐 Session")
st.sidebar.success("Active")

st.sidebar.markdown("---")
st.sidebar.markdown("## 📡 System Info")
st.sidebar.write("Model: LightGBM")
st.sidebar.write("Explainability: SHAP")
st.sidebar.write("Mode: Agentic AI")

# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("👤 Customer Profile")

repayment_map = {
    "Paid on time": 0,
    "1 month delay": 1,
    "2 months delay": 2,
    "3+ months delay": 3
}

LIMIT_BAL = st.number_input("Credit Limit (₹)", value=200000)
AGE = st.number_input("Age", value=35)

PAY_0 = repayment_map[st.selectbox("Recent Repayment", list(repayment_map.keys()))]
PAY_2 = repayment_map[st.selectbox("Previous Month", list(repayment_map.keys()))]

BILL_AMT1 = st.number_input("Latest Bill Amount", value=50000)
PAY_AMT1 = st.number_input("Latest Payment", value=5000)

# ============================================================
# RUN AGENT
# ============================================================

if st.button("🚀 Run AI Credit Assessment"):

    input_data = {
        "LIMIT_BAL": LIMIT_BAL,
        "AGE": AGE,
        "PAY_0": PAY_0,
        "PAY_2": PAY_2,
        "BILL_AMT1": BILL_AMT1,
        "PAY_AMT1": PAY_AMT1
    }

    # Fill missing features
    for col in feature_columns:
        if col not in input_data:
            input_data[col] = 0

    input_df = pd.DataFrame([input_data])[feature_columns]

    # ================================
    # AGENT OUTPUT
    # ================================

    agent_output = agent.run(input_df)

    result = agent_output["result"]
    recommendations = agent_output["recommendations"]
    reasons = agent_output.get("reasons", [])
    summary = agent_output["summary"]

    # ================================
    # DASHBOARD
    # ================================

    st.markdown("## 🧠 AI Decision Dashboard")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Score", result["credit_score"])
    c2.metric("PD", f"{result['pd']:.2%}")
    c3.metric("Loss", f"₹{result['expected_loss']:,.0f}")
    c4.metric("Risk", result["risk_bucket"])
    c5.metric("Decision", result["decision"])

    if result["decision"] == "Approve":
        st.success("✅ Safe to approve")
    elif result["decision"] == "Manual Review":
        st.warning("⚠️ Needs review")
    else:
        st.error("❌ High risk")

    # ================================
    # SHAP EXPLANATION (FIXED)
    # ================================

    st.markdown("## 🔍 Feature Impact")

    # ✅ FIX 2: Use scaled data
    input_scaled = scaler.transform(input_df)

    shap_values = explainer.shap_values(input_scaled)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Bar plot
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_scaled, plot_type="bar", show=False)
    st.pyplot(fig)

    # ================================
    # WATERFALL (SAFE VERSION)
    # ================================

    st.markdown("### 📊 Detailed Explanation")

    try:
        base_val = explainer.expected_value
        if isinstance(base_val, list):
            base_val = base_val[1]

        fig2 = plt.figure()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=base_val,
                data=input_scaled[0]
            )
        )
        st.pyplot(fig2)
    except Exception:
        st.warning("Detailed explanation not available for this input.")

    # ================================
    # TEXT EXPLANATION
    # ================================

    st.markdown("### 🧠 Key Drivers")

    top_features = sorted(
        zip(feature_columns, shap_values[0]),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    for f, v in top_features:
        if v > 0:
            st.write(f"🔺 {f} increased risk")
        else:
            st.write(f"🔻 {f} reduced risk")

    # ================================
    # REASONS & RECOMMENDATIONS
    # ================================

    st.markdown("### 🔍 Why this decision?")
    for r in reasons:
        st.write("✔", r)

    st.markdown("### 📌 Recommendations")
    for r in recommendations:
        st.write("➡️", r)

    # ================================
    # REPORT
    # ================================

    st.markdown("### 📄 AI Risk Report")
    st.text_area("Summary", summary, height=250)
