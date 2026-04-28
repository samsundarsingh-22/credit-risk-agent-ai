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
    page_title="AI Credit Risk System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>
.block-container {padding-top: 0.8rem;}
html, body, [class*="css"] {font-family: "Segoe UI", sans-serif;}

.main-header {
    background: linear-gradient(90deg,#08203e,#1f77b4);
    color: white;
    padding: 24px;
    border-radius: 18px;
    margin-bottom: 20px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.18);
}
.card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}
.low {background:#d8f3dc;color:#1b4332;padding:14px;border-radius:12px;font-weight:700;}
.medium {background:#fff3b0;color:#7a5901;padding:14px;border-radius:12px;font-weight:700;}
.high {background:#ffd6d6;color:#8b0000;padding:14px;border-radius:12px;font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION
# ============================================================

if "users" not in st.session_state:
    st.session_state.users = {"admin": "1234"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

if "input_data" not in st.session_state:
    st.session_state.input_data = None

if "result" not in st.session_state:
    st.session_state.result = None

# ============================================================
# LOGIN
# ============================================================

def login_page():
    st.markdown("""
    <div class="main-header">
        <h1>AI Credit Risk System</h1>
        <p>Secure credit analyst dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    tab_login, tab_signup = st.tabs(["Sign In", "Sign Up"])

    with tab_login:
        st.info("Demo Login: Username = admin | Password = 1234")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pwd")

        if st.button("Sign In"):
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.logged_in = True
                st.session_state.user = username
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab_signup:
        new_user = st.text_input("Create Username", key="signup_user")
        new_pwd = st.text_input("Create Password", type="password", key="signup_pwd")

        if st.button("Create Account"):
            if not new_user or not new_pwd:
                st.warning("Please enter username and password.")
            elif new_user in st.session_state.users:
                st.warning("User already exists.")
            else:
                st.session_state.users[new_user] = new_pwd
                st.success("Account created. Please sign in.")

if not st.session_state.logged_in:
    login_page()
    st.stop()

# ============================================================
# LOAD MODELS
# ============================================================

lgb_model = joblib.load("credit_model.pkl")
calibrator = joblib.load("calibrator.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

try:
    qsvc_model = joblib.load("qsvc_model.pkl")
    qsvc_available = True
except Exception:
    qsvc_model = None
    qsvc_available = False

explainer = shap.TreeExplainer(lgb_model)

# ============================================================
# HELPERS
# ============================================================

def rupee(x):
    return f"Rs. {x:,.0f}"

def safe_text(x):
    return str(x).encode("latin-1", "replace").decode("latin-1")

def calculate_emi(principal, annual_rate, tenure_years):
    monthly_rate = annual_rate / 12 / 100
    months = tenure_years * 12

    if principal <= 0 or months <= 0:
        return 0

    if monthly_rate == 0:
        return principal / months

    emi = (principal * monthly_rate * (1 + monthly_rate) ** months) / (
        ((1 + monthly_rate) ** months) - 1
    )
    return emi

def calculate_foir(existing_emi, new_emi, monthly_income):
    if monthly_income <= 0:
        return 100
    return ((existing_emi + new_emi) / monthly_income) * 100

def calculate_utilisation(outstanding, limit):
    return (outstanding / max(limit, 1)) * 100

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

def get_risk_band(pd_value):
    if pd_value < 0.10:
        return "Low Risk"
    elif pd_value < 0.25:
        return "Medium Risk"
    return "High Risk"

def final_decision_engine(pd_value, qsvc_signal, foir, utilisation):
    if foir > 55:
        return "Reject", "High FOIR indicates weak repayment capacity"

    if utilisation > 80:
        return "Reject", "Credit utilisation is very high"

    if pd_value > 0.40:
        return "Reject", "High probability of default"

    if qsvc_signal == 1 and pd_value >= 0.20:
        return "Reject", "QSVC high-risk signal with elevated PD"

    if pd_value > 0.20 or foir > 40:
        return "Manual Review", "Moderate risk or repayment burden"

    if qsvc_signal == 1:
        return "Manual Review", "QSVC high-risk signal requires analyst review"

    return "Approve", "Meets risk and affordability criteria"

def apply_financial_adjustment(pd_value, savings, mutual_funds, stocks, property, vehicle):
    adjusted_pd = pd_value
    total_assets = savings + mutual_funds + stocks

    if total_assets >= 1000000:
        adjusted_pd *= 0.90
    if total_assets >= 3000000:
        adjusted_pd *= 0.85

    if property == "Self Owned":
        adjusted_pd *= 0.85
    elif property == "Family Owned":
        adjusted_pd *= 0.93

    if vehicle == "Car":
        adjusted_pd *= 0.95
    elif vehicle == "Two Wheeler":
        adjusted_pd *= 0.98

    return min(max(adjusted_pd, 0.01), 0.95)

def build_model_input(data):
    model_data = {
        "LIMIT_BAL": data["LIMIT_BAL"],
        "AGE": data["AGE"],
        "PAY_0": data["PAY_0"],
        "PAY_2": data["PAY_2"],
        "BILL_AMT1": data["BILL_AMT1"],
        "PAY_AMT1": data["PAY_AMT1"]
    }

    for col in feature_columns:
        if col not in model_data:
            model_data[col] = 0

    return pd.DataFrame([model_data])[feature_columns]

def run_model(data):
    input_df = build_model_input(data)
    scaled = scaler.transform(input_df)

    lgb_probability = lgb_model.predict_proba(scaled)[:, 1][0]
    base_pd = calibrator.predict([lgb_probability])[0]
    base_pd = min(max(base_pd, 0.01), 0.95)

    try:
        if qsvc_available:
            qsvc_signal = qsvc_model.predict(scaled)[0]
        else:
            qsvc_signal = 1 if base_pd > 0.25 else 0
    except Exception:
        qsvc_signal = 1 if base_pd > 0.25 else 0

    adjusted_pd = apply_financial_adjustment(
        base_pd,
        data["SAVINGS"],
        data["MUTUAL_FUNDS"],
        data["STOCKS"],
        data["PROPERTY"],
        data["VEHICLE"]
    )

    new_emi = calculate_emi(
        data["REQUESTED_LOAN_AMOUNT"],
        data["INTEREST_RATE"],
        data["TENURE"]
    )

    foir = calculate_foir(
        data["EXISTING_EMI"],
        new_emi,
        data["MONTHLY_INCOME"]
    )

    utilisation = calculate_utilisation(
        data["BILL_AMT1"],
        data["LIMIT_BAL"]
    )

    decision, decision_reason = final_decision_engine(
        adjusted_pd,
        qsvc_signal,
        foir,
        utilisation
    )

    score = int(300 + (1 - adjusted_pd) * 600)
    expected_loss = adjusted_pd * data["LIMIT_BAL"] * 0.45

    eligible_loan_multiplier = data["MONTHLY_INCOME"] * 15
    max_new_emi_40_rule = max((data["MONTHLY_INCOME"] * 0.40) - data["EXISTING_EMI"], 0)

    return {
        "input_df": input_df,
        "scaled": scaled,
        "lgb_probability": lgb_probability,
        "base_pd": base_pd,
        "adjusted_pd": adjusted_pd,
        "qsvc_signal": qsvc_signal,
        "score": score,
        "expected_loss": expected_loss,
        "risk_band": get_risk_band(adjusted_pd),
        "score_band": get_score_band(score),
        "decision": decision,
        "decision_reason": decision_reason,
        "new_emi": new_emi,
        "foir": foir,
        "utilisation": utilisation,
        "eligible_loan_multiplier": eligible_loan_multiplier,
        "max_new_emi_40_rule": max_new_emi_40_rule
    }

def create_score_gauge(score):
    fig, ax = plt.subplots(figsize=(7.5, 2.7))
    segments = [
        (300, 500, "Very Poor"),
        (500, 650, "Poor"),
        (650, 700, "Fair"),
        (700, 750, "Good"),
        (750, 900, "Excellent")
    ]

    for start, end, label in segments:
        ax.barh(0, end - start, left=start, height=0.45)
        ax.text((start + end) / 2, 0, label, ha="center", va="center", fontsize=8)

    ax.axvline(score, linewidth=4)
    ax.text(score, 0.45, str(score), ha="center", fontsize=12, fontweight="bold")
    ax.set_xlim(300, 900)
    ax.set_yticks([])
    ax.set_xlabel("Credit Score Range")
    ax.set_title("CIBIL-style Credit Score Gauge")
    ax.spines[["top", "right", "left"]].set_visible(False)
    return fig

def create_risk_meter(pd_value):
    fig, ax = plt.subplots(figsize=(7.5, 2.7))
    segments = [
        (0.00, 0.10, "Low Risk"),
        (0.10, 0.25, "Medium Risk"),
        (0.25, 1.00, "High Risk")
    ]

    for start, end, label in segments:
        ax.barh(0, end - start, left=start, height=0.45)
        ax.text((start + end) / 2, 0, label, ha="center", va="center", fontsize=8)

    ax.axvline(pd_value, linewidth=4)
    ax.text(pd_value, 0.45, f"{pd_value:.2%}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Probability of Default")
    ax.set_title("Default Risk Meter")
    ax.spines[["top", "right", "left"]].set_visible(False)
    return fig

def generate_cibil_report_data(data):
    utilisation_pct = round(calculate_utilisation(data["BILL_AMT1"], data["LIMIT_BAL"]), 2)
    new_emi = calculate_emi(data["REQUESTED_LOAN_AMOUNT"], data["INTEREST_RATE"], data["TENURE"])

    accounts = pd.DataFrame([
        {
            "Account Type": "Credit Card",
            "Institution": "HDFC Bank",
            "Sanctioned / Limit": rupee(data["LIMIT_BAL"]),
            "Current Balance": rupee(data["BILL_AMT1"]),
            "Overdue": rupee(0 if data["PAY_0"] == 0 else data["PAY_AMT1"]),
            "Status": "Active",
            "Utilisation": f"{utilisation_pct}%"
        },
        {
            "Account Type": "Existing Personal Loan",
            "Institution": "ICICI Bank",
            "Sanctioned / Limit": rupee(data["EXISTING_EMI"] * data["TENURE"] * 12),
            "Current Balance": rupee(data["EXISTING_EMI"] * data["TENURE"] * 12),
            "Overdue": rupee(0),
            "Status": "Active",
            "Utilisation": "-"
        },
        {
            "Account Type": "Requested Personal Loan",
            "Institution": "AI Bank",
            "Sanctioned / Limit": rupee(data["REQUESTED_LOAN_AMOUNT"]),
            "Current Balance": rupee(data["REQUESTED_LOAN_AMOUNT"]),
            "Overdue": rupee(0),
            "Status": "Proposed",
            "Utilisation": "-"
        }
    ])

    payment_history = pd.DataFrame({
        "Month": ["M-6", "M-5", "M-4", "M-3", "M-2", "M-1"],
        "Payment Status": [
            "On Time",
            "On Time",
            "On Time",
            "Delay" if data["PAY_2"] > 0 else "On Time",
            "Delay" if data["PAY_0"] > 0 else "On Time",
            "Current"
        ]
    })

    affordability = pd.DataFrame({
        "Metric": [
            "Monthly Income",
            "Existing EMI",
            "Requested Loan Amount",
            "New EMI",
            "FOIR",
            "Credit Utilisation",
            "Max EMI by 40% Rule",
            "Multiplier Eligibility"
        ],
        "Value": [
            rupee(data["MONTHLY_INCOME"]),
            rupee(data["EXISTING_EMI"]),
            rupee(data["REQUESTED_LOAN_AMOUNT"]),
            rupee(new_emi),
            f"{calculate_foir(data['EXISTING_EMI'], new_emi, data['MONTHLY_INCOME']):.2f}%",
            f"{utilisation_pct}%",
            rupee(max((data["MONTHLY_INCOME"] * 0.40) - data["EXISTING_EMI"], 0)),
            rupee(data["MONTHLY_INCOME"] * 15)
        ]
    })

    enquiries = pd.DataFrame([
        {"Date": "2026-04-01", "Institution": "ABC Bank", "Purpose": "Credit Card", "Amount": rupee(50000)},
        {"Date": "2026-03-18", "Institution": "XYZ Finance", "Purpose": "Personal Loan", "Amount": rupee(200000)}
    ])

    signals = []
    if data["PAY_0"] > 0:
        signals.append("Recent repayment delay detected.")
    if utilisation_pct > 70:
        signals.append("High credit utilisation above 70%.")
    if data["PAY_AMT1"] < 0.1 * max(data["BILL_AMT1"], 1):
        signals.append("Low payment relative to outstanding balance.")
    if not signals:
        signals.append("No adverse credit behaviour detected.")

    return accounts, payment_history, affordability, enquiries, signals

def auto_approval_optimizer(data, current_result):
    if current_result["decision"] == "Approve":
        return {
            "status": "Already Eligible",
            "message": "Customer already satisfies approval threshold.",
            "scenario": data,
            "result": current_result
        }

    best = None

    candidate_outstanding = range(data["BILL_AMT1"], -1, -5000)
    candidate_payment = range(data["PAY_AMT1"], max(data["BILL_AMT1"], data["PAY_AMT1"]) + 1, 5000)
    candidate_loan = range(data["REQUESTED_LOAN_AMOUNT"], 50000 - 1, -50000)

    for outstanding in list(candidate_outstanding)[:80]:
        for payment in list(candidate_payment)[:50]:
            for loan_amount in list(candidate_loan)[:50]:
                scenario = data.copy()
                scenario["BILL_AMT1"] = outstanding
                scenario["PAY_AMT1"] = min(payment, max(outstanding, 1))
                scenario["PAY_0"] = 0
                scenario["PAY_0_WORD"] = "Paid on time"
                scenario["REQUESTED_LOAN_AMOUNT"] = loan_amount

                result = run_model(scenario)

                if result["decision"] == "Approve":
                    change_cost = (
                        abs(data["BILL_AMT1"] - scenario["BILL_AMT1"]) +
                        abs(data["REQUESTED_LOAN_AMOUNT"] - scenario["REQUESTED_LOAN_AMOUNT"]) * 0.3 +
                        abs(data["PAY_AMT1"] - scenario["PAY_AMT1"])
                    )

                    if best is None or change_cost < best["change_cost"]:
                        best = {
                            "status": "Approval Path Found",
                            "message": "Minimum practical changes identified for approval.",
                            "scenario": scenario,
                            "result": result,
                            "change_cost": change_cost
                        }

    if best is None:
        return {
            "status": "Approval Not Found",
            "message": "Approval may require higher income, lower obligations, collateral, or manual underwriting.",
            "scenario": data,
            "result": current_result
        }

    return best

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Credit Console")
st.sidebar.write(f"User: **{st.session_state.user}**")
st.sidebar.success("System Online")
st.sidebar.markdown("---")
st.sidebar.write("Layer 1: QSVC - Best PR-AUC")
st.sidebar.write("Layer 2: LightGBM - Best Accuracy")
st.sidebar.write("Layer 3: Calibrated LightGBM - Final PD")
st.sidebar.markdown("---")
st.sidebar.write("Underwriting: FOIR + EMI + Utilisation")
st.sidebar.write(f"QSVC: {'Available' if qsvc_available else 'Fallback Mode'}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.input_data = None
    st.session_state.result = None
    st.rerun()

# ============================================================
# HEADER
# ============================================================

st.markdown(f"""
<div class="main-header">
    <h1>AI Credit Risk Decision System</h1>
    <p>CIBIL-style dashboard | QSVC + LightGBM + Calibrated PD | FOIR Underwriting | Analyst: {st.session_state.user}</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Customer Profile",
    "Decision Dashboard",
    "Risk Analysis",
    "Simulator",
    "CIBIL Report",
    "PDF Download"
])

repayment_map = {
    "Paid on time": 0,
    "1 month delay": 1,
    "2 months delay": 2,
    "3+ months delay": 3
}

# ============================================================
# TAB 1
# ============================================================

with tab1:
    st.markdown("## Customer Financial Profile")

    st.markdown("""
    <div class="card">
    <b>Input Guide</b><br>
    The final decision is not based only on machine learning probability.
    It uses calibrated PD, FOIR/DTI, EMI affordability, credit utilisation, repayment behaviour and asset strength.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        MONTHLY_INCOME = st.slider("Monthly Income (Rs.)", 10000, 500000, 50000, step=5000)
        LIMIT_BAL = st.slider("Credit Limit / Exposure (Rs.)", 10000, 10000000, 200000, step=10000)
        AGE = st.slider("Age", 18, 70, 30)

    with c2:
        REQUESTED_LOAN_AMOUNT = st.slider("Requested Loan Amount (Rs.)", 50000, 5000000, 300000, step=50000)
        EXISTING_EMI = st.slider("Existing Monthly EMI (Rs.)", 0, 300000, 10000, step=5000)
        INTEREST_RATE = st.slider("Loan Interest Rate (%)", 5.0, 36.0, 12.0, step=0.5)
        TENURE = st.slider("Loan Tenure (Years)", 1, 30, 5)

    with c3:
        PAY_0_WORD = st.selectbox("Credit Card Repayment - Last Month", list(repayment_map.keys()))
        PAY_2_WORD = st.selectbox("Credit Card Repayment - 2 Months Ago", list(repayment_map.keys()))
        PAY_0 = repayment_map[PAY_0_WORD]
        PAY_2 = repayment_map[PAY_2_WORD]
        BILL_AMT1 = st.slider("Credit Card Outstanding Balance (Rs.)", 0, 1000000, 50000, step=5000)
        PAY_AMT1 = st.slider("Amount Paid Towards Credit Card (Rs.)", 0, 500000, 5000, step=5000)

    st.markdown("### Assets and Financial Strength")

    a1, a2, a3 = st.columns(3)

    with a1:
        SAVINGS = st.slider("Savings Balance (Rs.)", 0, 5000000, 100000, step=50000)

    with a2:
        MUTUAL_FUNDS = st.slider("Mutual Fund Holdings (Rs.)", 0, 5000000, 50000, step=50000)

    with a3:
        STOCKS = st.slider("Stock Holdings (Rs.)", 0, 5000000, 50000, step=50000)

    o1, o2 = st.columns(2)

    with o1:
        PROPERTY = st.selectbox("Property Ownership", ["None", "Family Owned", "Self Owned"])

    with o2:
        VEHICLE = st.selectbox("Vehicle Ownership", ["None", "Two Wheeler", "Car"])

    if st.button("Evaluate Credit Risk"):
        st.session_state.input_data = {
            "MONTHLY_INCOME": MONTHLY_INCOME,
            "LIMIT_BAL": LIMIT_BAL,
            "AGE": AGE,
            "REQUESTED_LOAN_AMOUNT": REQUESTED_LOAN_AMOUNT,
            "EXISTING_EMI": EXISTING_EMI,
            "INTEREST_RATE": INTEREST_RATE,
            "TENURE": TENURE,
            "PAY_0": PAY_0,
            "PAY_2": PAY_2,
            "PAY_0_WORD": PAY_0_WORD,
            "PAY_2_WORD": PAY_2_WORD,
            "BILL_AMT1": BILL_AMT1,
            "PAY_AMT1": PAY_AMT1,
            "SAVINGS": SAVINGS,
            "MUTUAL_FUNDS": MUTUAL_FUNDS,
            "STOCKS": STOCKS,
            "PROPERTY": PROPERTY,
            "VEHICLE": VEHICLE
        }

        st.session_state.result = run_model(st.session_state.input_data)
        st.success("Assessment completed. Open the Decision Dashboard tab.")

if st.session_state.input_data is not None:
    st.session_state.result = run_model(st.session_state.input_data)

# ============================================================
# TAB 2
# ============================================================

with tab2:
    if st.session_state.result is None:
        st.warning("Please evaluate customer first.")
    else:
        r = st.session_state.result

        st.markdown("## Decision Dashboard")

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Adjusted PD", f"{r['adjusted_pd']:.2%}")
        d2.metric("FOIR", f"{r['foir']:.2f}%")
        d3.metric("AI Credit Score", r["score"])
        d4.metric("Decision", r["decision"])

        if r["decision"] == "Approve":
            st.markdown('<div class="low">APPROVED: Customer satisfies risk and affordability criteria.</div>', unsafe_allow_html=True)
        elif r["decision"] == "Manual Review":
            st.markdown('<div class="medium">MANUAL REVIEW: Profile requires analyst review.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="high">REJECTED: Customer fails one or more underwriting rules.</div>', unsafe_allow_html=True)

        st.markdown("## Underwriting Summary")

        u1, u2, u3, u4 = st.columns(4)
        u1.metric("New EMI", rupee(r["new_emi"]))
        u2.metric("Max EMI by 40% Rule", rupee(r["max_new_emi_40_rule"]))
        u3.metric("Credit Utilisation", f"{r['utilisation']:.2f}%")
        u4.metric("Eligible Loan by Multiplier", rupee(r["eligible_loan_multiplier"]))

        st.markdown(f"**Decision Reason:** {r['decision_reason']}")

        st.markdown("## Three-Layer Model Intelligence")
        m1, m2, m3 = st.columns(3)
        m1.metric("QSVC Signal", "High Risk" if r["qsvc_signal"] == 1 else "Normal")
        m2.metric("LightGBM Probability", f"{r['lgb_probability']:.2%}")
        m3.metric("Calibrated PD", f"{r['base_pd']:.2%}")

        g1, g2 = st.columns(2)
        with g1:
            st.pyplot(create_score_gauge(r["score"]))
            st.write(f"Score Band: **{r['score_band']}**")
        with g2:
            st.pyplot(create_risk_meter(r["adjusted_pd"]))
            st.write(f"Risk Band: **{r['risk_band']}**")

# ============================================================
# TAB 3
# ============================================================

with tab3:
    if st.session_state.result is None:
        st.warning("Please evaluate customer first.")
    else:
        r = st.session_state.result

        st.markdown("## SHAP Risk Drivers")

        feature_name_map = {
            "LIMIT_BAL": "Credit Limit",
            "AGE": "Age",
            "PAY_0": "Repayment Last Month",
            "PAY_2": "Repayment 2 Months Ago",
            "BILL_AMT1": "Outstanding Balance",
            "PAY_AMT1": "Payment Made"
        }

        shap_values = explainer.shap_values(r["scaled"])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_df = pd.DataFrame({
            "Feature": feature_columns,
            "Impact": shap_values[0]
        })

        shap_df["Feature"] = shap_df["Feature"].map(lambda x: feature_name_map.get(x, x))
        shap_df["Direction"] = np.where(shap_df["Impact"] >= 0, "Increases risk", "Reduces risk")
        shap_df = shap_df.sort_values(by="Impact", key=np.abs, ascending=False).head(8)

        st.dataframe(shap_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.barh(shap_df["Feature"], shap_df["Impact"])
        ax.axvline(0)
        ax.set_xlabel("SHAP Impact on Default Risk")
        ax.set_title("Top Risk Drivers")
        ax.invert_yaxis()
        st.pyplot(fig)

# ============================================================
# TAB 4
# ============================================================

with tab4:
    if st.session_state.result is None or st.session_state.input_data is None:
        st.warning("Please evaluate customer first.")
    else:
        r = st.session_state.result
        d = st.session_state.input_data

        st.markdown("## Smart What-if Simulator")

        s1, s2, s3 = st.columns(3)

        with s1:
            sim_income = st.slider("Simulate Monthly Income (Rs.)", 10000, 500000, int(d["MONTHLY_INCOME"]), step=5000)
            sim_loan = st.slider("Simulate Requested Loan Amount (Rs.)", 50000, 5000000, int(d["REQUESTED_LOAN_AMOUNT"]), step=50000)

        with s2:
            sim_existing_emi = st.slider("Simulate Existing EMI (Rs.)", 0, 300000, int(d["EXISTING_EMI"]), step=5000)
            sim_outstanding = st.slider("Simulate Outstanding Balance (Rs.)", 0, int(d["LIMIT_BAL"]), int(d["BILL_AMT1"]), step=5000)

        with s3:
            sim_payment = st.slider("Simulate Card Payment (Rs.)", 0, int(max(d["BILL_AMT1"], 1)), int(d["PAY_AMT1"]), step=5000)
            sim_repayment_word = st.selectbox("Simulate Repayment Behaviour", list(repayment_map.keys()), index=list(repayment_map.keys()).index(d["PAY_0_WORD"]))

        sim_data = d.copy()
        sim_data["MONTHLY_INCOME"] = sim_income
        sim_data["REQUESTED_LOAN_AMOUNT"] = sim_loan
        sim_data["EXISTING_EMI"] = sim_existing_emi
        sim_data["BILL_AMT1"] = sim_outstanding
        sim_data["PAY_AMT1"] = sim_payment
        sim_data["PAY_0"] = repayment_map[sim_repayment_word]
        sim_data["PAY_0_WORD"] = sim_repayment_word

        sim_result = run_model(sim_data)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Current FOIR", f"{r['foir']:.2f}%")
        c2.metric("Simulated FOIR", f"{sim_result['foir']:.2f}%", delta=f"{sim_result['foir'] - r['foir']:.2f}%")
        c3.metric("Simulated PD", f"{sim_result['adjusted_pd']:.2%}", delta=f"{sim_result['adjusted_pd'] - r['adjusted_pd']:.2%}")
        c4.metric("Simulated Decision", sim_result["decision"])

        st.markdown("## Auto-Approval Optimizer")

        if st.button("Find Minimum Changes for Approval"):
            opt = auto_approval_optimizer(d, r)

            st.subheader(opt["status"])
            st.write(opt["message"])

            opt_result = opt["result"]
            opt_scenario = opt["scenario"]

            o1, o2, o3, o4 = st.columns(4)
            o1.metric("Optimized PD", f"{opt_result['adjusted_pd']:.2%}")
            o2.metric("Optimized FOIR", f"{opt_result['foir']:.2f}%")
            o3.metric("Optimized Score", opt_result["score"])
            o4.metric("Optimized Decision", opt_result["decision"])

            st.markdown("### Recommended Changes")
            st.write(f"- Requested Loan Amount: {rupee(d['REQUESTED_LOAN_AMOUNT'])} → {rupee(opt_scenario['REQUESTED_LOAN_AMOUNT'])}")
            st.write(f"- Outstanding Balance: {rupee(d['BILL_AMT1'])} → {rupee(opt_scenario['BILL_AMT1'])}")
            st.write(f"- Card Payment: {rupee(d['PAY_AMT1'])} → {rupee(opt_scenario['PAY_AMT1'])}")
            st.write("- Repayment behaviour: Paid on time")

# ============================================================
# TAB 5
# ============================================================

with tab5:
    if st.session_state.result is None or st.session_state.input_data is None:
        st.warning("Please evaluate customer first.")
    else:
        r = st.session_state.result
        d = st.session_state.input_data

        accounts, payment_history, affordability, enquiries, signals = generate_cibil_report_data(d)

        st.markdown("## Credit Bureau Report - CIBIL Style")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Credit Score", r["score"])
        c2.metric("Risk Band", r["risk_band"])
        c3.metric("Decision", r["decision"])
        c4.metric("FOIR", f"{r['foir']:.2f}%")

        st.markdown("### Credit Accounts")
        st.dataframe(accounts, use_container_width=True)

        st.markdown("### Payment History - Last 6 Months")
        st.dataframe(payment_history, use_container_width=True)

        st.markdown("### Affordability and Eligibility")
        st.dataframe(affordability, use_container_width=True)

        st.markdown("### Recent Enquiries")
        st.dataframe(enquiries, use_container_width=True)

        st.markdown("### Risk Signals")
        for signal in signals:
            st.write("•", signal)

# ============================================================
# TAB 6
# ============================================================

with tab6:
    if st.session_state.result is None or st.session_state.input_data is None:
        st.warning("Please evaluate customer first.")
    else:
        r = st.session_state.result
        d = st.session_state.input_data
        accounts, payment_history, affordability, enquiries, signals = generate_cibil_report_data(d)

        st.markdown("## Download Full CIBIL-style Credit Report")

        def create_pdf():
            pdf = FPDF()
            pdf.add_page()

            pdf.set_font("Arial", "B", 16)
            pdf.cell(190, 10, safe_text("AI CREDIT RISK REPORT"), ln=True, align="C")

            pdf.set_font("Arial", "", 10)
            pdf.cell(190, 8, safe_text(f"Customer/User: {st.session_state.user}"), ln=True)
            pdf.cell(190, 8, safe_text(f"Report Generated: {pd.Timestamp.now()}"), ln=True)
            pdf.ln(5)

            def section(title):
                pdf.set_font("Arial", "B", 12)
                pdf.cell(190, 10, safe_text(title), ln=True)
                pdf.set_font("Arial", "", 9)

            def table_rows(rows):
                for k, v in rows:
                    pdf.cell(90, 7, safe_text(k), 1)
                    pdf.cell(100, 7, safe_text(v), 1, ln=True)
                pdf.ln(4)

            section("1. Customer and Loan Information")
            table_rows([
                ("Monthly Income", rupee(d["MONTHLY_INCOME"])),
                ("Credit Limit", rupee(d["LIMIT_BAL"])),
                ("Requested Loan", rupee(d["REQUESTED_LOAN_AMOUNT"])),
                ("Existing EMI", rupee(d["EXISTING_EMI"])),
                ("New EMI", rupee(r["new_emi"])),
                ("FOIR", f"{r['foir']:.2f}%"),
                ("Credit Utilisation", f"{r['utilisation']:.2f}%"),
                ("Max EMI by 40% Rule", rupee(r["max_new_emi_40_rule"])),
                ("Multiplier Eligibility", rupee(r["eligible_loan_multiplier"]))
            ])

            section("2. Risk Summary")
            table_rows([
                ("QSVC Signal", "High Risk" if r["qsvc_signal"] == 1 else "Normal"),
                ("LightGBM Probability", f"{r['lgb_probability']:.2%}"),
                ("Calibrated PD", f"{r['base_pd']:.2%}"),
                ("Adjusted PD", f"{r['adjusted_pd']:.2%}"),
                ("AI Credit Score", r["score"]),
                ("Score Band", r["score_band"]),
                ("Risk Band", r["risk_band"]),
                ("Expected Loss", rupee(r["expected_loss"])),
                ("Decision", r["decision"]),
                ("Decision Reason", r["decision_reason"])
            ])

            section("3. Credit Accounts")
            for _, row in accounts.iterrows():
                pdf.multi_cell(0, 6, safe_text(
                    f"{row['Account Type']} | {row['Institution']} | "
                    f"Limit/Loan: {row['Sanctioned / Limit']} | "
                    f"Balance: {row['Current Balance']} | Overdue: {row['Overdue']} | "
                    f"Status: {row['Status']} | Utilisation: {row['Utilisation']}"
                ))
            pdf.ln(4)

            section("4. Payment History")
            for _, row in payment_history.iterrows():
                pdf.cell(90, 7, safe_text(row["Month"]), 1)
                pdf.cell(100, 7, safe_text(row["Payment Status"]), 1, ln=True)
            pdf.ln(4)

            section("5. Affordability and Eligibility")
            for _, row in affordability.iterrows():
                pdf.cell(90, 7, safe_text(row["Metric"]), 1)
                pdf.cell(100, 7, safe_text(row["Value"]), 1, ln=True)
            pdf.ln(4)

            section("6. Recent Enquiries")
            for _, row in enquiries.iterrows():
                pdf.multi_cell(0, 6, safe_text(
                    f"{row['Date']} | {row['Institution']} | {row['Purpose']} | Amount: {row['Amount']}"
                ))
            pdf.ln(4)

            section("7. Risk Signals")
            for signal in signals:
                pdf.multi_cell(0, 6, safe_text(f"- {signal}"))

            pdf.ln(4)
            pdf.set_font("Arial", "I", 8)
            pdf.multi_cell(
                0,
                6,
                safe_text(
                    "Disclaimer: This AI-generated report is for research and demonstration purposes only. "
                    "It does not replace an official credit bureau report."
                )
            )

            pdf_bytes = pdf.output(dest="S").encode("latin-1", "replace")
            return BytesIO(pdf_bytes)

        st.download_button(
            "Download Full CIBIL-style PDF Report",
            create_pdf(),
            file_name="credit_risk_report.pdf",
            mime="application/pdf"
        )
