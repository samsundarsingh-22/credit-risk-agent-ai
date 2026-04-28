import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
from datetime import date

from agent import CreditRiskAgent
from tools import (
    rupee,
    safe_text,
    calculate_utilisation,
    derive_dpd_from_history,
    apply_stress_scenario,
    simulate_credit_losses,
    BASEL_CAR_THRESHOLD,
    LGD_DEFAULT
)

st.set_page_config(
    page_title="AI Credit Risk System",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
# SESSION STATE
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

if "reasons" not in st.session_state:
    st.session_state.reasons = []

if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

if "accounts_df" not in st.session_state:
    st.session_state.accounts_df = pd.DataFrame(columns=[
        "Account Type", "Institution", "Sanctioned Amount",
        "Current Balance", "Overdue", "Status", "Utilisation %"
    ])

if "enquiry_df" not in st.session_state:
    st.session_state.enquiry_df = pd.DataFrame(columns=[
        "Date", "Institution", "Purpose", "Amount"
    ])

if "payment_df" not in st.session_state:
    st.session_state.payment_df = pd.DataFrame({
        "Month": ["M-6", "M-5", "M-4", "M-3", "M-2", "M-1"],
        "Payment Status": ["On Time", "On Time", "On Time", "On Time", "On Time", "Current"]
    })

if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame(columns=[
        "Customer", "PD", "Lifetime PD", "LGD", "EAD", "ECL", "Stage",
        "Standardised RWA", "Standardised CAR", "IRB RWA", "IRB CAR", "Decision"
    ])

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

model = joblib.load("credit_model.pkl")
calibrator = joblib.load("calibrator.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

try:
    qsvc_model = joblib.load("qsvc_model.pkl")
    qsvc_available = True
except Exception:
    qsvc_model = None
    qsvc_available = False

agent = CreditRiskAgent(
    model=model,
    calibrator=calibrator,
    scaler=scaler,
    feature_columns=feature_columns,
    qsvc_model=qsvc_model
)

explainer = shap.TreeExplainer(model)

# ============================================================
# CONSTANTS
# ============================================================

account_types = [
    "Credit Card", "Personal Loan", "Home Loan", "Auto Loan",
    "Education Loan", "Gold Loan", "Loan Against Property",
    "Consumer Durable Loan", "Business Loan"
]

banks = [
    "HDFC Bank", "ICICI Bank", "State Bank of India",
    "Axis Bank", "Kotak Mahindra Bank", "Bajaj Finance",
    "Tata Capital", "IDFC First Bank", "Bank of Baroda",
    "Canara Bank", "Union Bank of India", "Punjab National Bank",
    "Federal Bank", "IndusInd Bank", "Yes Bank"
]

purposes = [
    "Credit Card", "Personal Loan", "Home Loan", "Auto Loan",
    "Education Loan", "Gold Loan", "Loan Against Property", "Business Loan"
]

repayment_map = {
    "Paid on time": 0,
    "1 month delay": 1,
    "2 months delay": 2,
    "3+ months delay": 3
}

# ============================================================
# APP HELPERS
# ============================================================

def bureau_totals():
    df = st.session_state.accounts_df.copy()

    if df.empty:
        return 200000, 50000, 0, 25.0

    total_limit = float(df["Sanctioned Amount"].sum())
    total_balance = float(df["Current Balance"].sum())
    total_overdue = float(df["Overdue"].sum())
    util = calculate_utilisation(total_balance, total_limit)

    return total_limit, total_balance, total_overdue, util


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
    ax.set_title("Credit Score Gauge")
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


def auto_approval_optimizer(data, current_result):
    if current_result["decision"] == "Approve":
        return {
            "status": "Already Eligible",
            "message": "Customer already satisfies approval threshold.",
            "scenario": data,
            "result": current_result
        }

    best = None
    candidate_outstanding = range(int(data["BILL_AMT1"]), -1, -5000)
    candidate_payment = range(int(data["PAY_AMT1"]), int(max(data["BILL_AMT1"], data["PAY_AMT1"])) + 1, 5000)
    candidate_loan = range(int(data["REQUESTED_LOAN_AMOUNT"]), 50000 - 1, -50000)

    for outstanding in list(candidate_outstanding)[:60]:
        for payment in list(candidate_payment)[:40]:
            for loan_amount in list(candidate_loan)[:40]:
                scenario = data.copy()
                scenario["BILL_AMT1"] = outstanding
                scenario["PAY_AMT1"] = min(payment, max(outstanding, 1))
                scenario["PAY_0"] = 0
                scenario["PAY_0_WORD"] = "Paid on time"
                scenario["REQUESTED_LOAN_AMOUNT"] = loan_amount

                output = agent.run(scenario)
                result = output["result"]

                if result["decision"] == "Approve":
                    change_cost = (
                        abs(data["BILL_AMT1"] - scenario["BILL_AMT1"]) +
                        abs(data["REQUESTED_LOAN_AMOUNT"] - scenario["REQUESTED_LOAN_AMOUNT"]) * 0.30 +
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
st.sidebar.write("Underwriting: FOIR + EMI + Utilisation")
st.sidebar.write("IFRS 9: Stage + Lifetime ECL")
st.sidebar.write("Basel: Standardised RWA + IRB RWA + CAR")
st.sidebar.write("Advanced: VaR + Economic Capital")
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
    <p>CIBIL-style bureau input | QSVC + LightGBM + Calibrated PD | IFRS 9 ECL | Basel IRB | VaR | Analyst: {st.session_state.user}</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Customer Profile",
    "Bureau Data",
    "Decision Dashboard",
    "Risk Analysis",
    "Simulator",
    "IFRS 9 & Basel",
    "Stress & VaR",
    "CIBIL Report",
    "FAQs & PDF"
])

# ============================================================
# TAB 1
# ============================================================

with tab1:
    st.markdown("## Customer Financial Profile")

    st.markdown("""
    <div class="card">
    <b>Input Guide</b><br>
    This system uses calibrated PD, bureau data, FOIR, EMI affordability, credit utilisation, IFRS 9 staging, lifetime ECL, Basel capital metrics, stress testing and VaR-based economic capital.
    </div>
    """, unsafe_allow_html=True)

    bureau_limit, bureau_balance, bureau_overdue, bureau_utilisation = bureau_totals()

    c1, c2, c3 = st.columns(3)

    with c1:
        MONTHLY_INCOME = st.slider("Monthly Income (Rs.)", 10000, 500000, 50000, step=5000)
        AGE = st.slider("Age", 18, 70, 30)
        DPD_MANUAL = st.slider("Manual Days Past Due Override", 0, 180, 0)

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
        PAY_AMT1 = st.slider("Amount Paid Towards Credit Card (Rs.)", 0, 500000, 5000, step=5000)
        CAPITAL = st.slider("Bank Capital Allocation (Rs.)", 100000, 50000000, 1000000, step=100000)

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

    derived_dpd = derive_dpd_from_history(st.session_state.payment_df)
    final_dpd = max(DPD_MANUAL, derived_dpd)

    st.info(
        f"Bureau-derived Credit Limit: {rupee(bureau_limit)} | "
        f"Current Balance: {rupee(bureau_balance)} | "
        f"Overall Utilisation: {bureau_utilisation:.2f}% | "
        f"Derived DPD: {final_dpd}"
    )

    if st.button("Evaluate Credit Risk"):
        st.session_state.input_data = {
            "MONTHLY_INCOME": MONTHLY_INCOME,
            "LIMIT_BAL": bureau_limit,
            "AGE": AGE,
            "REQUESTED_LOAN_AMOUNT": REQUESTED_LOAN_AMOUNT,
            "EXISTING_EMI": EXISTING_EMI,
            "INTEREST_RATE": INTEREST_RATE,
            "TENURE": TENURE,
            "PAY_0": PAY_0,
            "PAY_2": PAY_2,
            "PAY_0_WORD": PAY_0_WORD,
            "PAY_2_WORD": PAY_2_WORD,
            "BILL_AMT1": bureau_balance,
            "PAY_AMT1": PAY_AMT1,
            "SAVINGS": SAVINGS,
            "MUTUAL_FUNDS": MUTUAL_FUNDS,
            "STOCKS": STOCKS,
            "PROPERTY": PROPERTY,
            "VEHICLE": VEHICLE,
            "DPD": final_dpd,
            "CAPITAL": CAPITAL
        }

        output = agent.run(st.session_state.input_data)
        st.session_state.result = output["result"]
        st.session_state.reasons = output["reasons"]
        st.session_state.recommendations = output["recommendations"]

        portfolio_row = {
            "Customer": st.session_state.user,
            "PD": st.session_state.result["adjusted_pd"],
            "Lifetime PD": st.session_state.result["lifetime_pd"],
            "LGD": LGD_DEFAULT,
            "EAD": st.session_state.result["ead"],
            "ECL": st.session_state.result["ecl"],
            "Stage": st.session_state.result["stage"],
            "Standardised RWA": st.session_state.result["standardised_rwa"],
            "Standardised CAR": st.session_state.result["standardised_car"],
            "IRB RWA": st.session_state.result["irb_rwa"],
            "IRB CAR": st.session_state.result["irb_car"],
            "Decision": st.session_state.result["decision"]
        }

        st.session_state.portfolio_df = st.session_state.portfolio_df[
            st.session_state.portfolio_df["Customer"] != st.session_state.user
        ]

        st.session_state.portfolio_df = pd.concat(
            [st.session_state.portfolio_df, pd.DataFrame([portfolio_row])],
            ignore_index=True
        )

        st.success("Assessment completed. Open the Decision Dashboard tab.")

if st.session_state.input_data is not None:
    output = agent.run(st.session_state.input_data)
    st.session_state.result = output["result"]
    st.session_state.reasons = output["reasons"]
    st.session_state.recommendations = output["recommendations"]

# ============================================================
# TAB 2
# ============================================================

with tab2:
    st.markdown("## User-entered Credit Bureau Data")

    st.warning("""
Real CIBIL/Experian data cannot be fetched without licensed bureau APIs.
This module captures user-entered bureau-style data using structured dropdowns and date fields.
""")

    st.markdown("### Credit Accounts")

    with st.form("add_account"):
        col1, col2 = st.columns(2)

        with col1:
            acc_type = st.selectbox("Account Type", account_types)
            bank = st.selectbox("Institution", banks)
            sanction = st.number_input("Sanctioned / Limit (Rs.)", 0, 10000000, 100000)

        with col2:
            balance = st.number_input("Current Balance (Rs.)", 0, 10000000, 50000)
            overdue = st.number_input("Overdue Amount (Rs.)", 0, 500000, 0)
            status = st.selectbox("Status", ["Active", "Closed", "Written-off", "Settled", "Proposed"])

        if st.form_submit_button("Add Account"):
            util = calculate_utilisation(balance, sanction)
            new_row = pd.DataFrame([{
                "Account Type": acc_type,
                "Institution": bank,
                "Sanctioned Amount": sanction,
                "Current Balance": balance,
                "Overdue": overdue,
                "Status": status,
                "Utilisation %": round(util, 2)
            }])
            st.session_state.accounts_df = pd.concat(
                [st.session_state.accounts_df, new_row],
                ignore_index=True
            )

    st.dataframe(st.session_state.accounts_df, use_container_width=True)

    if st.button("Clear Accounts"):
        st.session_state.accounts_df = st.session_state.accounts_df.iloc[0:0]
        st.rerun()

    st.markdown("### Payment History - Last 6 Months")

    payment_data = []
    status_options = ["On Time", "Delay", "Missed", "Current"]

    for m in ["M-6", "M-5", "M-4", "M-3", "M-2", "M-1"]:
        current_val = st.session_state.payment_df.loc[
            st.session_state.payment_df["Month"] == m, "Payment Status"
        ].iloc[0]

        status = st.selectbox(
            f"{m} Payment Status",
            status_options,
            index=status_options.index(current_val) if current_val in status_options else 0,
            key=f"payhist_{m}"
        )
        payment_data.append({"Month": m, "Payment Status": status})

    st.session_state.payment_df = pd.DataFrame(payment_data)
    st.dataframe(st.session_state.payment_df, use_container_width=True)

    st.markdown("### Recent Credit Enquiries")

    with st.form("add_enquiry"):
        col1, col2 = st.columns(2)

        with col1:
            enquiry_date = st.date_input("Enquiry Date", value=date.today())
            inst = st.selectbox("Institution", banks, key="enq_bank")

        with col2:
            purpose = st.selectbox("Purpose", purposes)
            amount = st.number_input("Requested Amount (Rs.)", 0, 5000000, 100000)

        if st.form_submit_button("Add Enquiry"):
            new_row = pd.DataFrame([{
                "Date": enquiry_date,
                "Institution": inst,
                "Purpose": purpose,
                "Amount": amount
            }])
            st.session_state.enquiry_df = pd.concat(
                [st.session_state.enquiry_df, new_row],
                ignore_index=True
            )

    st.dataframe(st.session_state.enquiry_df, use_container_width=True)

    if st.button("Clear Enquiries"):
        st.session_state.enquiry_df = st.session_state.enquiry_df.iloc[0:0]
        st.rerun()

    total_limit, total_balance, total_overdue, overall_util = bureau_totals()

    st.markdown("### Credit Utilisation")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sanctioned Limit", rupee(total_limit))
    c2.metric("Current Balance", rupee(total_balance))
    c3.metric("Total Overdue", rupee(total_overdue))
    c4.metric("Overall Utilisation", f"{overall_util:.2f}%")

# ============================================================
# TAB 3
# ============================================================

with tab3:
    if st.session_state.result is None:
        st.warning("Please evaluate customer first.")
    else:
        r = st.session_state.result

        st.markdown("## Decision Dashboard")

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Adjusted PD", f"{r['adjusted_pd']:.2%}")
        d2.metric("Lifetime PD", f"{r['lifetime_pd']:.2%}")
        d3.metric("AI Credit Score", r["score"])
        d4.metric("Decision", r["decision"])

        if r["decision"] == "Approve":
            st.markdown('<div class="low">APPROVED: Customer satisfies risk, affordability, and capital criteria.</div>', unsafe_allow_html=True)
        elif r["decision"] == "Manual Review":
            st.markdown('<div class="medium">MANUAL REVIEW: Profile requires analyst review.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="high">REJECTED: Customer fails one or more underwriting or risk rules.</div>', unsafe_allow_html=True)

        st.markdown("## Underwriting Summary")

        u1, u2, u3, u4 = st.columns(4)
        u1.metric("FOIR", f"{r['foir']:.2f}%")
        u2.metric("New EMI", rupee(r["new_emi"]))
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

        st.markdown("## Agent Reasoning")
        for item in st.session_state.reasons:
            st.write("•", item)

        st.markdown("## Recommendations")
        for item in st.session_state.recommendations:
            st.write("•", item)

# ============================================================
# TAB 4
# ============================================================

with tab4:
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
# TAB 5
# ============================================================

with tab5:
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
            sim_outstanding = st.slider("Simulate Outstanding Balance (Rs.)", 0, int(max(d["LIMIT_BAL"], 1)), int(d["BILL_AMT1"]), step=5000)

        with s3:
            sim_payment = st.slider("Simulate Card Payment (Rs.)", 0, int(max(d["BILL_AMT1"], 1)), int(d["PAY_AMT1"]), step=5000)
            sim_repayment_word = st.selectbox(
                "Simulate Repayment Behaviour",
                list(repayment_map.keys()),
                index=list(repayment_map.keys()).index(d["PAY_0_WORD"])
            )

        sim_data = d.copy()
        sim_data["MONTHLY_INCOME"] = sim_income
        sim_data["REQUESTED_LOAN_AMOUNT"] = sim_loan
        sim_data["EXISTING_EMI"] = sim_existing_emi
        sim_data["BILL_AMT1"] = sim_outstanding
        sim_data["PAY_AMT1"] = sim_payment
        sim_data["PAY_0"] = repayment_map[sim_repayment_word]
        sim_data["PAY_0_WORD"] = sim_repayment_word

        sim_output = agent.run(sim_data)
        sim_result = sim_output["result"]

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
            st.write(f"- Requested Loan Amount: {rupee(d['REQUESTED_LOAN_AMOUNT'])} -> {rupee(opt_scenario['REQUESTED_LOAN_AMOUNT'])}")
            st.write(f"- Outstanding Balance: {rupee(d['BILL_AMT1'])} -> {rupee(opt_scenario['BILL_AMT1'])}")
            st.write(f"- Card Payment: {rupee(d['PAY_AMT1'])} -> {rupee(opt_scenario['PAY_AMT1'])}")
            st.write("- Repayment behaviour: Paid on time")

# ============================================================
# TAB 6
# ============================================================

with tab6:
    if st.session_state.result is None:
        st.warning("Please evaluate customer first.")
    else:
        r = st.session_state.result

        st.markdown("## IFRS 9 ECL, Stage Migration, PD Term Structure and Basel Capital")

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("IFRS 9 Stage", r["stage"])
        b2.metric("ECL Type", r["ecl_type"])
        b3.metric("ECL", rupee(r["ecl"]))
        b4.metric("Lifetime PD", f"{r['lifetime_pd']:.2%}")

        st.markdown("### PD Term Structure")
        pd_term_display = r["pd_term_df"].copy()
        pd_term_display["Marginal PD"] = pd_term_display["Marginal PD"].map(lambda x: f"{x:.2%}")
        pd_term_display["Cumulative PD"] = pd_term_display["Cumulative PD"].map(lambda x: f"{x:.2%}")
        st.dataframe(pd_term_display, use_container_width=True)

        st.markdown("### Stage Migration Matrix")
        st.dataframe(r["transition_matrix"], use_container_width=True)
        st.info(f"Current exposure belongs to: {r['current_stage_row']}")

        st.markdown("### Basel Capital Metrics")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EAD", rupee(r["ead"]))
        c2.metric("LGD", f"{r['lgd']:.0%}")
        c3.metric("Standardised RWA", rupee(r["standardised_rwa"]))
        c4.metric("Standardised CAR", f"{r['standardised_car']:.2f}%")

        i1, i2, i3, i4 = st.columns(4)
        i1.metric("IRB Capital %", f"{r['irb_capital_requirement_pct']:.2%}")
        i2.metric("IRB Capital Amount", rupee(r["irb_capital_amount"]))
        i3.metric("IRB RWA", rupee(r["irb_rwa"]))
        i4.metric("IRB CAR", f"{r['irb_car']:.2f}%")

        j1, j2 = st.columns(2)
        j1.metric("Asset Correlation", f"{r['asset_correlation']:.4f}")
        j2.metric("Maturity Adjustment", f"{r['maturity_adjustment']:.4f}")

        if r["standardised_car"] >= BASEL_CAR_THRESHOLD:
            st.success("Capital adequacy is above the simplified Basel threshold.")
        else:
            st.error("Capital adequacy is below the simplified Basel threshold and needs review.")

        st.markdown("## Portfolio Risk")
        st.dataframe(st.session_state.portfolio_df, use_container_width=True)

        if not st.session_state.portfolio_df.empty:
            p = st.session_state.portfolio_df
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Total EAD", rupee(p["EAD"].sum()))
            p2.metric("Total ECL", rupee(p["ECL"].sum()))
            p3.metric("Total Standardised RWA", rupee(p["Standardised RWA"].sum()))
            p4.metric("Average PD", f"{p['PD'].mean():.2%}")

            st.bar_chart(p["Stage"].value_counts())

# ============================================================
# TAB 7
# ============================================================

with tab7:
    if st.session_state.result is None:
        st.warning("Please evaluate customer first.")
    else:
        r = st.session_state.result

        st.markdown("## Macroeconomic Stress Testing")

        scenario = st.selectbox(
            "Select Stress Scenario",
            ["Base Case", "Mild Stress", "Recession", "Severe Recession", "High Interest Rate Shock"]
        )

        stress = apply_stress_scenario(r["adjusted_pd"], LGD_DEFAULT, r["ead"], scenario)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Stressed PD", f"{stress['Stressed PD']:.2%}")
        s2.metric("Stressed LGD", f"{stress['Stressed LGD']:.2%}")
        s3.metric("Stressed EAD", rupee(stress["Stressed EAD"]))
        s4.metric("Stressed ECL", rupee(stress["Stressed ECL"]))

        st.write(stress["Description"])

        stress_rows = []
        for sc in ["Base Case", "Mild Stress", "Recession", "Severe Recession", "High Interest Rate Shock"]:
            sr = apply_stress_scenario(r["adjusted_pd"], LGD_DEFAULT, r["ead"], sc)
            stress_rows.append({
                "Scenario": sr["Scenario"],
                "PD": f"{sr['Stressed PD']:.2%}",
                "LGD": f"{sr['Stressed LGD']:.2%}",
                "EAD": rupee(sr["Stressed EAD"]),
                "ECL": rupee(sr["Stressed ECL"])
            })

        st.dataframe(pd.DataFrame(stress_rows), use_container_width=True)

        st.markdown("## Economic Capital and Credit VaR")

        confidence_level = st.selectbox("Confidence Level", [0.95, 0.975, 0.99], index=2)
        n_sim = st.slider("Monte Carlo Simulations", 1000, 50000, 10000, step=1000)

        var_result = simulate_credit_losses(r["adjusted_pd"], LGD_DEFAULT, r["ead"], n_sim, confidence_level)

        v1, v2, v3 = st.columns(3)
        v1.metric("Expected Loss", rupee(var_result["expected_loss"]))
        v2.metric(f"Credit VaR {confidence_level:.1%}", rupee(var_result["var_loss"]))
        v3.metric("Economic Capital", rupee(var_result["economic_capital"]))

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(var_result["losses"], bins=30)
        ax.axvline(var_result["expected_loss"], linestyle="--", label="Expected Loss")
        ax.axvline(var_result["var_loss"], linestyle="--", label="Credit VaR")
        ax.set_title("Monte Carlo Credit Loss Distribution")
        ax.set_xlabel("Loss")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

# ============================================================
# TAB 8
# ============================================================

with tab8:
    if st.session_state.result is None:
        st.warning("Please evaluate customer first.")
    else:
        r = st.session_state.result

        st.markdown("## Credit Bureau Report - User-entered CIBIL-style Data")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Credit Score", r["score"])
        c2.metric("Risk Band", r["risk_band"])
        c3.metric("Decision", r["decision"])
        c4.metric("FOIR", f"{r['foir']:.2f}%")

        st.markdown("### Credit Accounts")
        st.dataframe(st.session_state.accounts_df, use_container_width=True)

        st.markdown("### Payment History")
        st.dataframe(st.session_state.payment_df, use_container_width=True)

        st.markdown("### Recent Enquiries")
        st.dataframe(st.session_state.enquiry_df, use_container_width=True)

        total_limit, total_balance, total_overdue, overall_util = bureau_totals()

        st.markdown("### Bureau Summary")
        summary_df = pd.DataFrame({
            "Metric": ["Total Sanctioned Limit", "Current Balance", "Total Overdue", "Overall Utilisation"],
            "Value": [rupee(total_limit), rupee(total_balance), rupee(total_overdue), f"{overall_util:.2f}%"]
        })
        st.dataframe(summary_df, use_container_width=True)

# ============================================================
# TAB 9
# ============================================================

with tab9:
    st.markdown("## FAQs and Regulatory Positioning")

    with st.expander("Are we fully Basel compliant?"):
        st.write("""
This is a Basel-inspired academic prototype, not a certified regulatory implementation.
It includes PD, LGD, EAD, ECL, IFRS 9 staging, lifetime ECL, transition matrix, PD term structure, standardised RWA, simplified IRB capital, CAR, stress testing and VaR.
It does not perform supervisory reporting or regulatory model validation.
""")

    with st.expander("What is lifetime ECL?"):
        st.write("""
Lifetime ECL estimates expected loss over the remaining life of the exposure.
In this prototype, Stage 2 and Stage 3 exposures use lifetime ECL, while Stage 1 uses 12-month ECL.
""")

    with st.expander("What is PD term structure?"):
        st.write("""
The PD term structure shows marginal and cumulative default probability over multiple future years.
It supports lifetime ECL calculation for Stage 2 and Stage 3 exposures.
""")

    with st.expander("What is stage migration?"):
        st.write("""
Stage migration represents the probability that an exposure moves from Stage 1 to Stage 2 or Stage 3 over time.
This prototype uses a simplified transition matrix for academic demonstration.
""")

    with st.expander("What is Basel IRB capital?"):
        st.write("""
Basel IRB capital links PD, LGD, EAD, asset correlation and maturity adjustment to estimate regulatory-style capital.
This implementation is simplified and not valid for supervisory reporting.
""")

    with st.expander("Is the CIBIL report real?"):
        st.write("""
No. Real CIBIL data requires licensed bureau APIs. This app uses structured user-entered bureau-style data.
""")

    st.markdown("---")

    if st.session_state.result is not None and st.session_state.input_data is not None:
        r = st.session_state.result

        st.markdown("## Download Full Report")

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

            section("1. Risk Summary")
            table_rows([
                ("Adjusted PD", f"{r['adjusted_pd']:.2%}"),
                ("Lifetime PD", f"{r['lifetime_pd']:.2%}"),
                ("Score", r["score"]),
                ("Decision", r["decision"]),
                ("Decision Reason", r["decision_reason"]),
                ("FOIR", f"{r['foir']:.2f}%"),
                ("Utilisation", f"{r['utilisation']:.2f}%")
            ])

            section("2. IFRS 9 ECL")
            table_rows([
                ("Stage", r["stage"]),
                ("ECL Type", r["ecl_type"]),
                ("EAD", rupee(r["ead"])),
                ("LGD", f"{r['lgd']:.0%}"),
                ("ECL", rupee(r["ecl"]))
            ])

            section("3. Basel Capital")
            table_rows([
                ("Standardised RWA", rupee(r["standardised_rwa"])),
                ("Risk Weight", f"{r['risk_weight']:.0%}"),
                ("Standardised CAR", f"{r['standardised_car']:.2f}%"),
                ("IRB Capital %", f"{r['irb_capital_requirement_pct']:.2%}"),
                ("IRB Capital Amount", rupee(r["irb_capital_amount"])),
                ("IRB RWA", rupee(r["irb_rwa"])),
                ("IRB CAR", f"{r['irb_car']:.2f}%")
            ])

            section("4. VaR and Economic Capital")
            table_rows([
                ("VaR 99%", rupee(r["var_99"])),
                ("Economic Capital", rupee(r["economic_capital"]))
            ])

            section("5. Credit Accounts")
            if st.session_state.accounts_df.empty:
                pdf.multi_cell(0, 6, safe_text("No user-entered credit account data."))
            else:
                for _, row in st.session_state.accounts_df.iterrows():
                    pdf.multi_cell(0, 6, safe_text(
                        f"{row['Account Type']} | {row['Institution']} | "
                        f"Sanctioned: {rupee(row['Sanctioned Amount'])} | "
                        f"Balance: {rupee(row['Current Balance'])} | "
                        f"Overdue: {rupee(row['Overdue'])} | "
                        f"Status: {row['Status']} | Utilisation: {row['Utilisation %']}%"
                    ))

            section("6. Payment History")
            for _, row in st.session_state.payment_df.iterrows():
                pdf.cell(90, 7, safe_text(row["Month"]), 1)
                pdf.cell(100, 7, safe_text(row["Payment Status"]), 1, ln=True)
            pdf.ln(4)

            section("7. Recent Enquiries")
            if st.session_state.enquiry_df.empty:
                pdf.multi_cell(0, 6, safe_text("No user-entered enquiries."))
            else:
                for _, row in st.session_state.enquiry_df.iterrows():
                    pdf.multi_cell(0, 6, safe_text(
                        f"{row['Date']} | {row['Institution']} | {row['Purpose']} | Amount: {rupee(row['Amount'])}"
                    ))

            section("8. Regulatory Note")
            pdf.multi_cell(0, 6, safe_text(
                "This is a Basel-inspired academic prototype. It includes PD, LGD, EAD, IFRS 9 staging, lifetime ECL, transition matrix, PD term structure, standardised RWA, simplified IRB capital, CAR, stress testing, VaR and user-entered bureau-style data. It is not a certified regulatory system."
            ))

            pdf_bytes = pdf.output(dest="S").encode("latin-1", "replace")
            return BytesIO(pdf_bytes)

        st.download_button(
            "Download Full Credit Risk Report",
            create_pdf(),
            file_name="credit_risk_report.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("Please evaluate customer first to generate PDF.")
