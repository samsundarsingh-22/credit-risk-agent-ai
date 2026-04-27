import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from agent import CreditRiskAgent
from fpdf import FPDF

# ============================================================
# STYLE
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
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD FILES
# ============================================================

model = joblib.load("credit_model.pkl")
calibrator = joblib.load("calibrator.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

agent = CreditRiskAgent(model, calibrator, scaler, feature_columns)
explainer = shap.TreeExplainer(model)

# ============================================================
# FEATURE LABELS
# ============================================================

feature_name_map = {
    "LIMIT_BAL": "Credit Limit",
    "AGE": "Age",
    "PAY_0": "Recent Payment Delay",
    "PAY_2": "Previous Payment Delay",
    "BILL_AMT1": "Latest Bill Amount",
    "PAY_AMT1": "Latest Payment"
}

# ============================================================
# UI HEADER
# ============================================================

st.title("🏦 Agentic AI Credit Risk System")
st.caption("Explainable AI • Decision Engine • Risk Intelligence")

# ============================================================
# INPUTS
# ============================================================

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
# RUN MODEL
# ============================================================

if st.button("Run AI Credit Assessment"):

    input_data = {
        "LIMIT_BAL": LIMIT_BAL,
        "AGE": AGE,
        "PAY_0": PAY_0,
        "PAY_2": PAY_2,
        "BILL_AMT1": BILL_AMT1,
        "PAY_AMT1": PAY_AMT1
    }

    for col in feature_columns:
        if col not in input_data:
            input_data[col] = 0

    input_df = pd.DataFrame([input_data])[feature_columns]

    agent_output = agent.run(input_df)

    result = agent_output["result"]
    recommendations = agent_output["recommendations"]
    reasons = agent_output["reasons"]
    summary = agent_output["summary"]

    # ============================================================
    # DASHBOARD
    # ============================================================

    st.markdown("## 🧠 Decision Dashboard")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Score", result["credit_score"])
    c2.metric("PD", f"{result['pd']:.2%}")
    c3.metric("Loss", f"₹{result['expected_loss']:,.0f}")
    c4.metric("Risk", result["risk_bucket"])
    c5.metric("Decision", result["decision"])

    # ============================================================
    # SHAP
    # ============================================================

    st.markdown("## 🔍 Feature Impact")

    input_scaled = scaler.transform(input_df)

    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_columns)
    input_scaled_df.rename(columns=feature_name_map, inplace=True)

    shap_values = explainer.shap_values(input_scaled_df)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig, ax = plt.subplots()

    shap.summary_plot(
        shap_values,
        input_scaled_df,
        plot_type="bar",
        show=False
    )

    ax.set_xlabel("Impact on Risk")
    ax.set_ylabel("Customer Features")

    st.pyplot(fig)

    # ============================================================
    # KEY DRIVERS
    # ============================================================

    st.markdown("### Key Drivers")

    top_features = sorted(
        zip(feature_columns, shap_values[0]),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    for f, v in top_features:
        name = feature_name_map.get(f, f)

        if v > 0:
            st.write(f"🔴 {name} increased risk")
        else:
            st.write(f"🟢 {name} reduced risk")

    # ============================================================
    # REASONS
    # ============================================================

    st.markdown("### 🔍 Why this decision?")
    for r in reasons:
        st.write("✔", r)

    # ============================================================
    # RECOMMENDATIONS
    # ============================================================

    st.markdown("### Recommendations")
    for r in recommendations:
        st.write("➡️", r)

    # ============================================================
    # WHAT-IF SIMULATOR
    # ============================================================

    st.markdown("## 🔄 What-if Simulator")

    sim_payment = st.slider("Change Payment Amount", 0, int(BILL_AMT1), PAY_AMT1)

    if st.button("Simulate Scenario"):

        temp_df = input_df.copy()
        temp_df["PAY_AMT1"] = sim_payment

        sim_out = agent.run(temp_df)

        st.write("New PD:", f"{sim_out['result']['pd']:.2%}")
        st.write("New Score:", sim_out['result']['credit_score'])

    # ============================================================
    # PDF REPORT
    # ============================================================

    st.markdown("## 📄 Download Report")

    if st.button("Generate PDF"):

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, "Credit Risk Report", ln=True)
        pdf.cell(200, 10, f"Score: {result['credit_score']}", ln=True)
        pdf.cell(200, 10, f"PD: {result['pd']:.2%}", ln=True)
        pdf.cell(200, 10, f"Decision: {result['decision']}", ln=True)

        pdf.output("report.pdf")

        with open("report.pdf", "rb") as f:
            st.download_button("Download PDF", f, "credit_report.pdf")

    # ============================================================
    # INSIGHT
    # ============================================================

    st.markdown("## 📊 Model Insight")

    st.write(
        "The model shows that repayment delays and high bill utilization "
        "increase default risk, while higher payments reduce risk."
    )
