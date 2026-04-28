import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="AI Credit Risk System", layout="wide")

# ============================================================
# SESSION (LOGIN / SIGNUP)
# ============================================================

if "users" not in st.session_state:
    st.session_state.users = {"admin": "1234"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = None


def auth():
    st.title("🔐 AI Credit Risk System")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.info("Demo → admin / 1234")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        if st.button("Login"):
            if user in st.session_state.users and st.session_state.users[user] == pwd:
                st.session_state.logged_in = True
                st.session_state.current_user = user
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_user = st.text_input("New Username")
        new_pwd = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            if new_user in st.session_state.users:
                st.warning("User exists")
            else:
                st.session_state.users[new_user] = new_pwd
                st.success("Account created")


if not st.session_state.logged_in:
    auth()
    st.stop()

# ============================================================
# LOAD MODELS (IMPORTANT)
# ============================================================

qsvc_model = joblib.load("qsvc_model.pkl")
lgb_model = joblib.load("credit_model.pkl")
calibrator = joblib.load("calibrator.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

explainer = shap.TreeExplainer(lgb_model)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏦 System Panel")
st.sidebar.write(f"User: {st.session_state.current_user}")
st.sidebar.success("System Active")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ============================================================
# HEADER
# ============================================================

st.title("🏦 AI Credit Risk Decision System")
st.caption("QSVC + LightGBM + Calibration • Explainable AI • Simulator")

# ============================================================
# INPUTS
# ============================================================

repayment_map = {
    "Paid on time": 0,
    "1 month delay": 1,
    "2 months delay": 2,
    "3+ months delay": 3
}

st.subheader("👤 Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    LIMIT_BAL = st.slider("💳 Credit Limit (₹)", 10000, 10000000, 200000)
    AGE = st.slider("Age", 18, 70, 30)

with col2:
    PAY_0 = repayment_map[st.selectbox("Last Month Repayment", repayment_map.keys())]
    PAY_2 = repayment_map[st.selectbox("2 Months Ago Repayment", repayment_map.keys())]

with col3:
    BILL_AMT1 = st.slider("Outstanding Bill (₹)", 0, 1000000, 50000)
    PAY_AMT1 = st.slider("Payment Made (₹)", 0, 500000, 5000)

# ============================================================
# EVALUATION
# ============================================================

if st.button("🚀 Evaluate Risk"):

    # Prepare input
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

    # ================================
    # 3-LAYER MODEL PIPELINE
    # ================================

    input_scaled = scaler.transform(input_df)

    # QSVC
    qsvc_pred = qsvc_model.predict(input_scaled)[0]

    # LightGBM
    lgb_proba = lgb_model.predict_proba(input_scaled)[:, 1][0]

    # Calibrated PD
    base_pd = calibrator.predict([lgb_proba])[0]

    # ================================
    # DECISION LOGIC
    # ================================

    if qsvc_pred == 1:
        if base_pd > 0.20:
            decision = "Reject"
        else:
            decision = "Manual Review"
    else:
        if base_pd < 0.10:
            decision = "Approve"
        elif base_pd < 0.25:
            decision = "Manual Review"
        else:
            decision = "Reject"

    score = int(300 + (1 - base_pd) * 600)
    expected_loss = base_pd * LIMIT_BAL * 0.45

    # ============================================================
    # OUTPUT DASHBOARD
    # ============================================================

    st.subheader("📊 Decision Dashboard")

    d1, d2, d3, d4 = st.columns(4)

    d1.metric("PD", f"{base_pd:.2%}")
    d2.metric("Score", score)
    d3.metric("Expected Loss", f"₹{expected_loss:,.0f}")
    d4.metric("Decision", decision)

    # ============================================================
    # MODEL LAYERS
    # ============================================================

    st.subheader("🧠 Model Outputs")

    m1, m2, m3 = st.columns(3)

    m1.metric("QSVC", "High Risk" if qsvc_pred else "Normal")
    m2.metric("LightGBM", f"{lgb_proba:.2%}")
    m3.metric("Calibrated PD", f"{base_pd:.2%}")

    # ============================================================
    # SCORE GAUGE
    # ============================================================

    st.subheader("🎯 Credit Score")

    fig, ax = plt.subplots()
    ax.barh([0], [600], left=300)
    ax.axvline(score)
    ax.set_xlim(300, 900)
    ax.set_yticks([])
    st.pyplot(fig)

    # ============================================================
    # SHAP
    # ============================================================

    st.subheader("🔍 Explainability")

    shap_values = explainer.shap_values(input_scaled)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_scaled, plot_type="bar", show=False)
    st.pyplot(fig)

    # ============================================================
    # SIMULATOR
    # ============================================================

    st.subheader("🔄 Simulator")

    new_payment = st.slider("Increase Payment", 0, int(BILL_AMT1), PAY_AMT1)

    temp_df = input_df.copy()
    temp_df["PAY_AMT1"] = new_payment

    sim_scaled = scaler.transform(temp_df)
    sim_pd = calibrator.predict(
        lgb_model.predict_proba(sim_scaled)[:, 1]
    )[0]

    st.write(f"New PD: {sim_pd:.2%}")

    # ============================================================
    # PDF REPORT
    # ============================================================

    def create_pdf():
        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "AI CREDIT REPORT", ln=True)

        pdf.cell(200, 10, f"PD: {base_pd:.2%}", ln=True)
        pdf.cell(200, 10, f"Score: {score}", ln=True)
        pdf.cell(200, 10, f"Decision: {decision}", ln=True)

        return BytesIO(pdf.output(dest="S").encode("latin-1"))

    st.download_button("📄 Download Report", create_pdf(), "report.pdf")
