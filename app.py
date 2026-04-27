import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from agent import CreditRiskAgent
from io import BytesIO
from fpdf import FPDF

# ============================================================
# SESSION INIT (LOGIN / SIGNUP)
# ============================================================

if "users" not in st.session_state:
    st.session_state.users = {"admin": "1234"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = None

# ============================================================
# AUTH PAGE
# ============================================================

def auth_page():
    st.title("🔐 AI Credit Risk System")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
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
        new_user = st.text_input("Create Username")
        new_pwd = st.text_input("Create Password", type="password")

        if st.button("Sign Up"):
            if new_user in st.session_state.users:
                st.warning("User already exists")
            else:
                st.session_state.users[new_user] = new_pwd
                st.success("Signup successful")

# Stop if not logged in
if not st.session_state.logged_in:
    auth_page()
    st.stop()

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("credit_model.pkl")
calibrator = joblib.load("calibrator.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

agent = CreditRiskAgent(model, calibrator, scaler, feature_columns)
explainer = shap.TreeExplainer(model)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏦 Dashboard")
st.sidebar.write(f"User: {st.session_state.current_user}")
st.sidebar.success("System Active")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ============================================================
# FEATURE LABELS
# ============================================================

feature_map = {
    "LIMIT_BAL": "Credit Limit",
    "AGE": "Age",
    "PAY_0": "Recent Payment Delay",
    "PAY_2": "Previous Payment Delay",
    "BILL_AMT1": "Latest Bill",
    "PAY_AMT1": "Latest Payment"
}

# ============================================================
# UI
# ============================================================

st.title("🏦 AI Credit Risk Decision System")
st.caption("Explainable AI • Decision Support • Simulation")

repayment_map = {
    "Paid on time": 0,
    "1 month delay": 1,
    "2 months delay": 2,
    "3+ months delay": 3
}

LIMIT_BAL = st.number_input("Credit Limit (₹)", value=200000)
AGE = st.number_input("Age", value=35)

PAY_0 = repayment_map[st.selectbox("Recent Repayment", list(repayment_map.keys()))]
PAY_2 = repayment_map[st.selectbox("Previous Repayment", list(repayment_map.keys()))]

BILL_AMT1 = st.number_input("Latest Bill Amount", value=50000)
PAY_AMT1 = st.number_input("Latest Payment", value=5000)

# ============================================================
# MAIN BUTTON
# ============================================================

if st.button("🚀 Evaluate Credit Risk"):

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

    # Run agent
    output = agent.run(input_df)
    result = output["result"]
    reasons = output["reasons"]
    recs = output["recommendations"]

    # ============================================================
    # DASHBOARD
    # ============================================================

    st.markdown("## 🧠 Decision Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("PD", f"{result['pd']:.2%}")
    c2.metric("Score", result["credit_score"])
    c3.metric("Decision", result["decision"])

    # ============================================================
    # SHAP
    # ============================================================

    st.markdown("## 🔍 Feature Impact")

    scaled = scaler.transform(input_df)
    df_scaled = pd.DataFrame(scaled, columns=feature_columns)
    df_scaled.rename(columns=feature_map, inplace=True)

    shap_values = explainer.shap_values(df_scaled)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, df_scaled, plot_type="bar", show=False)
    ax.set_xlabel("Impact on Default Risk")
    ax.set_ylabel("Customer Features")
    st.pyplot(fig)

    # ============================================================
    # BUSINESS INTERPRETATION
    # ============================================================

    st.markdown("## 🧠 Business Interpretation")

    top = sorted(zip(feature_columns, shap_values[0]),
                 key=lambda x: abs(x[1]), reverse=True)[:3]

    for f, v in top:
        name = feature_map.get(f, f)
        if v > 0:
            st.write(f"🔴 High {name} increases risk")
        else:
            st.write(f"🟢 Lower {name} reduces risk")

    # ============================================================
    # SIMULATOR
    # ============================================================

    st.markdown("## 🔄 What-if Simulator")

    new_payment = st.slider("Change Payment", 0, int(BILL_AMT1), PAY_AMT1)

    temp_df = input_df.copy()
    temp_df["PAY_AMT1"] = new_payment

    sim = agent.run(temp_df)

    col1, col2 = st.columns(2)
    col1.metric("Current PD", f"{result['pd']:.2%}")
    col2.metric("New PD", f"{sim['result']['pd']:.2%}")

    # ============================================================
    # DECISION LOGIC
    # ============================================================

    st.markdown("## ⚖️ Decision Logic")

    st.write(f"""
    - PD < 20% → Approve  
    - 20–40% → Review  
    - >40% → Reject  

    Current PD: {result['pd']:.2%} → {result['decision']}
    """)

    # ============================================================
    # PDF DOWNLOAD (CORRECT)
    # ============================================================

    def create_pdf(result):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, "Credit Risk Report", ln=True)
        pdf.cell(200, 10, f"PD: {result['pd']:.2%}", ln=True)
        pdf.cell(200, 10, f"Score: {result['credit_score']}", ln=True)
        pdf.cell(200, 10, f"Decision: {result['decision']}", ln=True)

        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        return BytesIO(pdf_bytes)

    st.download_button(
        "📄 Download Report",
        create_pdf(result),
        file_name="credit_report.pdf",
        mime="application/pdf"
    )
