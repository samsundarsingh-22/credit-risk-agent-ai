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

st.set_page_config(
    page_title="AI Credit Risk Decision System",
    layout="wide"
)

# ============================================================
# PREMIUM UI STYLE
# ============================================================

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    background-color: #f5f7fb;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #102A43;
}

.sub-title {
    font-size: 17px;
    color: #486581;
    margin-bottom: 20px;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}

.bank-header {
    background: linear-gradient(90deg, #102A43, #1f77b4);
    color: white;
    padding: 24px;
    border-radius: 18px;
    margin-bottom: 20px;
}

.low-box {
    background-color: #D8F3DC;
    color: #1B4332;
    padding: 16px;
    border-radius: 14px;
    font-weight: bold;
}

.medium-box {
    background-color: #FFF3B0;
    color: #7A5901;
    padding: 16px;
    border-radius: 14px;
    font-weight: bold;
}

.high-box {
    background-color: #FFD6D6;
    color: #8B0000;
    padding: 16px;
    border-radius: 14px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION: LOGIN / SIGNUP
# ============================================================

if "users" not in st.session_state:
    st.session_state.users = {"admin": "1234"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = None


def auth_page():
    st.markdown("<div class='main-title'>🔐 AI Credit Risk System</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Secure banking-style credit assessment dashboard</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

    with tab1:
        st.info("Demo Login: Username = admin | Password = 1234")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pwd")

        if st.button("Sign In"):
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.logged_in = True
                st.session_state.current_user = username
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_user = st.text_input("Create Username", key="signup_user")
        new_pwd = st.text_input("Create Password", type="password", key="signup_pwd")

        if st.button("Create Account"):
            if not new_user or not new_pwd:
                st.warning("Please enter username and password")
            elif new_user in st.session_state.users:
                st.warning("User already exists")
            else:
                st.session_state.users[new_user] = new_pwd
                st.success("Account created. Please sign in.")


if not st.session_state.logged_in:
    auth_page()
    st.stop()

# ============================================================
# LOAD MODEL OBJECTS
# ============================================================

qsvc_available = True

try:
    qsvc_model = joblib.load("qsvc_model.pkl")
except Exception:
    qsvc_model = None
    qsvc_available = False

lgb_model = joblib.load("credit_model.pkl")
calibrator = joblib.load("calibrator.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

explainer = shap.TreeExplainer(lgb_model)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def rupee(x):
    return f"₹{x:,.0f}"


def get_decision(pd_value, qsvc_signal):
    if qsvc_signal == 1:
        if pd_value >= 0.20:
            return "Reject"
        return "Manual Review"

    if pd_value < 0.10:
        return "Approve"
    elif pd_value < 0.25:
        return "Manual Review"
    else:
        return "Reject"


def get_risk_band(pd_value):
    if pd_value < 0.10:
        return "Low Risk"
    elif pd_value < 0.25:
        return "Medium Risk"
    return "High Risk"


def get_score_band(score):
    if score >= 750:
        return "Excellent"
    elif score >= 700:
        return "Good"
    elif score >= 650:
        return "Fair"
    elif score >= 600:
        return "Poor"
    return "Very Poor"


def apply_financial_adjustment(pd_value, emi, savings, mutual_funds, stocks, property_status, vehicle_status):
    adjusted_pd = pd_value
    total_assets = savings + mutual_funds + stocks

    if total_assets >= 1000000:
        adjusted_pd *= 0.90

    if total_assets >= 3000000:
        adjusted_pd *= 0.85

    if property_status == "Self Owned":
        adjusted_pd *= 0.85
    elif property_status == "Family Owned":
        adjusted_pd *= 0.93

    if vehicle_status == "Car":
        adjusted_pd *= 0.95
    elif vehicle_status == "Two Wheeler":
        adjusted_pd *= 0.98

    if emi >= 50000:
        adjusted_pd *= 1.20

    if emi >= 100000:
        adjusted_pd *= 1.35

    return min(max(adjusted_pd, 0.01), 0.95)


def create_score_gauge(score):
    fig, ax = plt.subplots(figsize=(7, 2.5))

    ax.barh([0], [150], left=300, label="Poor")
    ax.barh([0], [100], left=450, label="Fair")
    ax.barh([0], [100], left=550, label="Good")
    ax.barh([0], [150], left=650, label="Very Good")
    ax.barh([0], [100], left=800, label="Excellent")

    ax.axvline(score, linewidth=4)
    ax.set_xlim(300, 900)
    ax.set_yticks([])
    ax.set_xlabel("Credit Score Range")
    ax.set_title(f"CIBIL-style Score Gauge: {score}")

    return fig


def create_risk_meter(pd_value):
    fig, ax = plt.subplots(figsize=(7, 2.5))

    ax.barh([0], [0.10], left=0.00, label="Low Risk")
    ax.barh([0], [0.15], left=0.10, label="Medium Risk")
    ax.barh([0], [0.75], left=0.25, label="High Risk")

    ax.axvline(pd_value, linewidth=4)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Probability of Default")
    ax.set_title(f"Risk Meter: {pd_value:.2%}")
    ax.legend(loc="upper center", ncol=3)

    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏦 Credit Risk Console")
st.sidebar.write(f"User: **{st.session_state.current_user}**")
st.sidebar.success("System Online")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Model Stack")
st.sidebar.write("Layer 1: QSVC - Best PR-AUC")
st.sidebar.write("Layer 2: LightGBM - Best Accuracy")
st.sidebar.write("Layer 3: Calibrated LightGBM - Final PD")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔎 Deployment Status")
st.sidebar.write(f"QSVC Available: {'Yes' if qsvc_available else 'Fallback Mode'}")
st.sidebar.write("Explainability: SHAP")
st.sidebar.write("Report: CIBIL-style PDF")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.rerun()

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class='bank-header'>
    <h1>🏦 AI Credit Risk Decision System</h1>
    <p>CIBIL-style dashboard • Quantum risk screening • LightGBM prediction • Calibrated default probability</p>
</div>
""", unsafe_allow_html=True)

st.info("""
This dashboard estimates customer credit risk using a three-layer intelligence system:
**QSVC** for rare-risk screening, **LightGBM** for accurate classification, and **calibrated LightGBM** for final default probability.
Additional asset and liability information is used as an agentic financial-strength adjustment layer.
""")

# ============================================================
# INPUT SECTION
# ============================================================

st.markdown("## 👤 Customer Financial Profile")

st.markdown("""
<div class='card'>
<b>Input Guide</b><br>
• Credit Limit: Maximum approved exposure.<br>
• Credit Card Repayment: Whether recent payments were delayed.<br>
• Outstanding Balance: Unpaid credit card bill.<br>
• Amount Paid: Actual payment made toward the card bill.<br>
• EMI, savings, investments and ownership details help the agent assess financial strength.
</div>
""", unsafe_allow_html=True)

repayment_map = {
    "Paid on time": 0,
    "1 month delay": 1,
    "2 months delay": 2,
    "3+ months delay": 3
}

# ---------------- CREDIT PROFILE ----------------

st.markdown("### 1. Credit Card Profile")

c1, c2, c3 = st.columns(3)

with c1:
    LIMIT_BAL = st.slider(
        "💳 Credit Limit / Exposure (₹)",
        min_value=10000,
        max_value=10000000,
        step=10000,
        value=200000
    )

    AGE = st.slider("🎂 Age", 18, 70, 30)

with c2:
    PAY_0_WORD = st.selectbox("📅 Credit Card Repayment - Last Month", list(repayment_map.keys()))
    PAY_2_WORD = st.selectbox("📅 Credit Card Repayment - 2 Months Ago", list(repayment_map.keys()))

    PAY_0 = repayment_map[PAY_0_WORD]
    PAY_2 = repayment_map[PAY_2_WORD]

with c3:
    BILL_AMT1 = st.slider(
        "📊 Credit Card Outstanding Balance (₹)",
        min_value=0,
        max_value=1000000,
        step=5000,
        value=50000
    )

    PAY_AMT1 = st.slider(
        "💸 Amount Paid Towards Credit Card (₹)",
        min_value=0,
        max_value=500000,
        step=5000,
        value=5000
    )

# ---------------- PERSONAL LOAN ----------------

st.markdown("### 2. Personal Loan Information")

l1, l2, l3 = st.columns(3)

with l1:
    EMI = st.slider("📉 Monthly Personal Loan EMI (₹)", 0, 200000, 10000, step=5000)

with l2:
    INTEREST_RATE = st.slider("📊 Personal Loan Interest Rate (%)", 5.0, 36.0, 12.0, step=0.5)

with l3:
    TENURE = st.slider("⏳ Loan Tenure (Years)", 1, 30, 5)

# ---------------- ASSETS ----------------

st.markdown("### 3. Savings, Investments and Assets")

a1, a2, a3 = st.columns(3)

with a1:
    SAVINGS = st.slider("💰 Savings Balance (₹)", 0, 5000000, 100000, step=50000)

with a2:
    MUTUAL_FUNDS = st.slider("📈 Mutual Fund Holdings (₹)", 0, 5000000, 50000, step=50000)

with a3:
    STOCKS = st.slider("📊 Stock Holdings (₹)", 0, 5000000, 50000, step=50000)

o1, o2 = st.columns(2)

with o1:
    PROPERTY = st.selectbox("🏠 Property Ownership", ["None", "Family Owned", "Self Owned"])

with o2:
    VEHICLE = st.selectbox("🚗 Vehicle Ownership", ["None", "Two Wheeler", "Car"])

# ============================================================
# EVALUATION
# ============================================================

if st.button("🚀 Evaluate Credit Risk"):

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
    input_scaled = scaler.transform(input_df)

    # ========================================================
    # THREE-LAYER MODEL PIPELINE
    # ========================================================

    lgb_proba = lgb_model.predict_proba(input_scaled)[:, 1][0]
    base_pd = calibrator.predict([lgb_proba])[0]

    try:
        if qsvc_available:
            qsvc_pred = qsvc_model.predict(input_scaled)[0]
        else:
            qsvc_pred = 1 if base_pd > 0.25 else 0
    except Exception:
        qsvc_pred = 1 if base_pd > 0.25 else 0

    adjusted_pd = apply_financial_adjustment(
        base_pd,
        EMI,
        SAVINGS,
        MUTUAL_FUNDS,
        STOCKS,
        PROPERTY,
        VEHICLE
    )

    decision = get_decision(adjusted_pd, qsvc_pred)
    risk_band = get_risk_band(adjusted_pd)
    score = int(300 + (1 - adjusted_pd) * 600)
    score_band = get_score_band(score)
    expected_loss = adjusted_pd * LIMIT_BAL * 0.45

    # ========================================================
    # MODEL STACK OUTPUT
    # ========================================================

    st.markdown("## 🧠 Three-Layer Model Intelligence")

    m1, m2, m3 = st.columns(3)

    m1.metric("QSVC Risk Signal", "High Risk" if qsvc_pred == 1 else "Normal")
    m2.metric("LightGBM Probability", f"{lgb_proba:.2%}")
    m3.metric("Final Calibrated PD", f"{base_pd:.2%}")

    st.write("""
**Model role explanation:**  
- **QSVC** is used as the rare-risk screening layer because it achieved the best PR-AUC.  
- **LightGBM** is used as the main predictive model because it achieved the best accuracy.  
- **Calibrated LightGBM** is used for final PD because calibrated probability is required for financial decision-making.
""")

    # ========================================================
    # DASHBOARD
    # ========================================================

    st.markdown("## 📊 Credit Decision Dashboard")

    d1, d2, d3, d4 = st.columns(4)

    d1.metric("Adjusted PD", f"{adjusted_pd:.2%}")
    d2.metric("AI Credit Score", score)
    d3.metric("Expected Loss", rupee(expected_loss))
    d4.metric("Decision", decision)

    if risk_band == "Low Risk":
        st.markdown("<div class='low-box'>LOW RISK: Customer appears suitable for approval.</div>", unsafe_allow_html=True)
    elif risk_band == "Medium Risk":
        st.markdown("<div class='medium-box'>MEDIUM RISK: Manual review is recommended.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='high-box'>HIGH RISK: Reject or request stronger verification.</div>", unsafe_allow_html=True)

    # ========================================================
    # GAUGE + RISK METER
    # ========================================================

    st.markdown("## 🎯 CIBIL-style Score Gauge and Risk Meter")

    g1, g2 = st.columns(2)

    with g1:
        st.pyplot(create_score_gauge(score))
        st.write(f"**Score Band:** {score_band}")

    with g2:
        st.pyplot(create_risk_meter(adjusted_pd))
        st.write(f"**Risk Band:** {risk_band}")

    # ========================================================
    # OUTPUT INTERPRETATION
    # ========================================================

    st.markdown("## 📘 Output Interpretation")

    st.write(f"""
- The LightGBM model estimated a raw default probability of **{lgb_proba:.2%}**.
- The calibrated probability of default is **{base_pd:.2%}**.
- After financial-strength adjustment, the final PD is **{adjusted_pd:.2%}**.
- The AI credit score is **{score}**, which falls under the **{score_band}** band.
- Final decision: **{decision}**.
""")

    st.markdown("## ⚖️ Decision Logic")

    st.write("""
- Adjusted PD < 10% and no QSVC risk flag → Approve  
- Adjusted PD between 10% and 25% → Manual Review  
- Adjusted PD > 25% or QSVC high-risk signal with high PD → Reject  
""")

    # ========================================================
    # SHAP EXPLAINABILITY
    # ========================================================

    st.markdown("## 🔍 SHAP Risk Drivers")

    feature_name_map = {
        "LIMIT_BAL": "Credit Limit",
        "AGE": "Age",
        "PAY_0": "Credit Card Repayment Last Month",
        "PAY_2": "Credit Card Repayment 2 Months Ago",
        "BILL_AMT1": "Credit Card Outstanding Balance",
        "PAY_AMT1": "Amount Paid Towards Credit Card"
    }

    shap_values = explainer.shap_values(input_scaled)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_df = pd.DataFrame({
        "Feature": feature_columns,
        "Impact": shap_values[0]
    })

    shap_df["Feature"] = shap_df["Feature"].map(lambda x: feature_name_map.get(x, x))
    shap_df["Risk Direction"] = np.where(
        shap_df["Impact"] >= 0,
        "Increases risk",
        "Reduces risk"
    )

    shap_df = shap_df.sort_values(by="Impact", key=np.abs, ascending=False).head(8)

    st.dataframe(shap_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(shap_df["Feature"], shap_df["Impact"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("SHAP Impact on Default Risk")
    ax.set_ylabel("Customer Features")
    ax.set_title("Top Risk Drivers")
    ax.invert_yaxis()
    st.pyplot(fig)

    st.info("""
Positive SHAP values increase default risk. Negative SHAP values reduce default risk.
This explains which financial variables influenced the model prediction.
""")

    # ========================================================
    # WHAT-IF SIMULATOR
    # ========================================================

    st.markdown("## 🔄 What-if Simulator")

    s_col1, s_col2 = st.columns(2)

    with s_col1:
        sim_payment = st.slider(
            "Simulate Higher Credit Card Payment (₹)",
            min_value=0,
            max_value=int(max(BILL_AMT1, 1)),
            value=int(PAY_AMT1),
            step=5000
        )

    with s_col2:
        sim_repayment_word = st.selectbox(
            "Simulate Repayment Behaviour",
            list(repayment_map.keys()),
            index=0
        )

    temp_df = input_df.copy()
    temp_df["PAY_AMT1"] = sim_payment
    temp_df["PAY_0"] = repayment_map[sim_repayment_word]

    sim_scaled = scaler.transform(temp_df)
    sim_lgb_proba = lgb_model.predict_proba(sim_scaled)[:, 1][0]
    sim_base_pd = calibrator.predict([sim_lgb_proba])[0]

    sim_adjusted_pd = apply_financial_adjustment(
        sim_base_pd,
        EMI,
        SAVINGS,
        MUTUAL_FUNDS,
        STOCKS,
        PROPERTY,
        VEHICLE
    )

    sim_decision = get_decision(sim_adjusted_pd, qsvc_pred)
    sim_score = int(300 + (1 - sim_adjusted_pd) * 600)

    sim1, sim2, sim3 = st.columns(3)

    sim1.metric("Current Adjusted PD", f"{adjusted_pd:.2%}")
    sim2.metric("Simulated Adjusted PD", f"{sim_adjusted_pd:.2%}", delta=f"{sim_adjusted_pd - adjusted_pd:.2%}")
    sim3.metric("Simulated Decision", sim_decision)

    st.write(f"Simulated Score: **{sim_score}**")

    # ========================================================
    # PROFESSIONAL PDF
    # ========================================================

    def create_pdf():
        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "AI CREDIT RISK REPORT", ln=True, align="C")

        pdf.set_font("Arial", "", 10)
        pdf.cell(200, 8, f"Customer/User: {st.session_state.current_user}", ln=True)
        pdf.cell(200, 8, f"Generated: {pd.Timestamp.now()}", ln=True)
        pdf.ln(5)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "1. CUSTOMER INFORMATION", ln=True)

        pdf.set_font("Arial", "", 10)
        customer_rows = [
            ("Credit Limit", rupee(LIMIT_BAL)),
            ("Age", AGE),
            ("Outstanding Card Balance", rupee(BILL_AMT1)),
            ("Card Payment Made", rupee(PAY_AMT1)),
            ("Personal Loan EMI", rupee(EMI)),
            ("Interest Rate", f"{INTEREST_RATE}%"),
            ("Tenure", f"{TENURE} years"),
            ("Savings", rupee(SAVINGS)),
            ("Mutual Funds", rupee(MUTUAL_FUNDS)),
            ("Stocks", rupee(STOCKS)),
            ("Property", PROPERTY),
            ("Vehicle", VEHICLE)
        ]

        for k, v in customer_rows:
            pdf.cell(95, 8, str(k), 1)
            pdf.cell(95, 8, str(v), 1, ln=True)

        pdf.ln(5)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "2. RISK SUMMARY", ln=True)

        risk_rows = [
            ("QSVC Signal", "High Risk" if qsvc_pred == 1 else "Normal"),
            ("LightGBM Probability", f"{lgb_proba:.2%}"),
            ("Calibrated PD", f"{base_pd:.2%}"),
            ("Adjusted PD", f"{adjusted_pd:.2%}"),
            ("AI Credit Score", score),
            ("Score Band", score_band),
            ("Risk Band", risk_band),
            ("Expected Loss", rupee(expected_loss)),
            ("Decision", decision)
        ]

        pdf.set_font("Arial", "", 10)
        for k, v in risk_rows:
            pdf.cell(95, 8, str(k), 1)
            pdf.cell(95, 8, str(v), 1, ln=True)

        pdf.ln(5)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "3. MODEL SELECTION SUMMARY", ln=True)

        model_rows = [
            ("Best PR-AUC Model", "QSVC (Quantum Kernel)"),
            ("Best Accuracy Model", "LightGBM"),
            ("Best Calibrated Model", "LightGBM"),
            ("Deployment Probability Layer", "Calibrated LightGBM")
        ]

        pdf.set_font("Arial", "", 10)
        for k, v in model_rows:
            pdf.cell(95, 8, str(k), 1)
            pdf.cell(95, 8, str(v), 1, ln=True)

        pdf.ln(5)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "4. INTERPRETATION", ln=True)

        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(
            0,
            7,
            f"The final adjusted probability of default is {adjusted_pd:.2%}. "
            f"The customer receives an AI credit score of {score}, classified as {score_band}. "
            f"The recommended decision is {decision}. This assessment combines QSVC risk screening, "
            f"LightGBM prediction, calibration, and financial-strength adjustment."
        )

        pdf.ln(3)

        pdf.set_font("Arial", "I", 9)
        pdf.multi_cell(
            0,
            6,
            "Disclaimer: This AI-generated report is for research and demonstration purposes only. "
            "It does not replace an official credit bureau report."
        )

        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        return BytesIO(pdf_bytes)

    st.download_button(
        "📄 Download CIBIL-style Credit Report",
        create_pdf(),
        file_name="credit_risk_report.pdf",
        mime="application/pdf"
    )
