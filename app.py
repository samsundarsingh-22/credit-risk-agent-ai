import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from agent import CreditRiskAgent
from io import BytesIO
from fpdf import FPDF

# ============================================================
# PAGE CONFIG + STYLE
# ============================================================
st.set_page_config(page_title="AI Credit Risk System", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1rem;}
h1, h2, h3 {color:#1f3b73;}
.metric-card {background:#fff;border-radius:12px;padding:10px;box-shadow:0 2px 6px rgba(0,0,0,0.08);}
.small {color:#666;font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

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
# AUTH PAGE (Login / Sign Up)
# ============================================================
def auth_page():
    st.title("🔐 AI Credit Risk System")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pwd")
        if st.button("Login"):
            if u in st.session_state.users and st.session_state.users[u] == p:
                st.session_state.logged_in = True
                st.session_state.current_user = u
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        nu = st.text_input("Create Username", key="signup_user")
        npw = st.text_input("Create Password", type="password", key="signup_pwd")
        if st.button("Sign Up"):
            if nu in st.session_state.users:
                st.warning("User already exists")
            else:
                st.session_state.users[nu] = npw
                st.success("Signup successful. Please login.")

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
st.sidebar.write(f"User: **{st.session_state.current_user}**")
st.sidebar.success("System Active")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 Engine")
st.sidebar.write("Model: LightGBM")
st.sidebar.write("Explainability: SHAP")
st.sidebar.write("Mode: Agentic AI")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ============================================================
# MAIN UI
# ============================================================
st.title("🏦 AI Credit Risk Decision System")
st.caption("CIBIL-style Report • Explainable AI • What-if Simulation")

st.info("""
**How to use**
- Enter customer financials below
- Click **Evaluate**
- See decision, drivers, and simulate improvements
""")

# ============================================================
# INPUTS
# ============================================================
repayment_map = {
    "Paid on time": 0,
    "1 month delay": 1,
    "2 months delay": 2,
    "3+ months delay": 3
}

colA, colB, colC = st.columns(3)

with colA:
    LIMIT_BAL = st.number_input("Credit Limit (₹)", value=200000, step=10000)
    AGE = st.number_input("Age", value=35, min_value=18, max_value=100)

with colB:
    PAY_0 = repayment_map[st.selectbox("Recent Repayment", list(repayment_map.keys()))]
    PAY_2 = repayment_map[st.selectbox("Previous Repayment", list(repayment_map.keys()))]

with colC:
    BILL_AMT1 = st.number_input("Latest Bill Amount (₹)", value=50000, step=1000)
    PAY_AMT1 = st.number_input("Latest Payment (₹)", value=5000, step=1000)

# ============================================================
# RUN MODEL
# ============================================================
if st.button("🚀 Evaluate Credit Risk"):

    # Build full feature vector
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

    # Agent output
    out = agent.run(input_df)
    result = out["result"]
    reasons = out.get("reasons", [])
    recs = out.get("recommendations", [])

    # ========================================================
    # DECISION DASHBOARD
    # ========================================================
    st.markdown("## 🧠 Decision Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PD", f"{result['pd']:.2%}")
    c2.metric("Score", result["credit_score"])
    c3.metric("Expected Loss", f"₹{result.get('expected_loss',0):,.0f}")
    c4.metric("Decision", result["decision"])

    # Color cue
    if result["decision"] == "Approve":
        st.success("✅ Recommended: Approve")
    elif result["decision"] == "Manual Review":
        st.warning("⚠️ Recommended: Manual Review")
    else:
        st.error("❌ Recommended: Reject")

    # ========================================================
    # DECISION LOGIC
    # ========================================================
    st.markdown("### ⚖️ Decision Logic")
    st.write(f"""
- **PD < 20% → Approve**  
- **20–40% → Review**  
- **> 40% → Reject**  

Current PD: **{result['pd']:.2%} → {result['decision']}**
""")

    # ========================================================
    # SHAP (Clear + Creative)
    # ========================================================
    st.markdown("## 🔍 Risk Drivers (Explainability)")

    scaled = scaler.transform(input_df)
    shap_values = explainer.shap_values(scaled)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Top features table
    shap_df = pd.DataFrame({
        "Feature": feature_columns,
        "Impact": shap_values[0]
    })
    shap_df["Direction"] = np.where(shap_df["Impact"] >= 0, "↑ Increase Risk", "↓ Decrease Risk")
    shap_df = shap_df.sort_values(by="Impact", key=np.abs, ascending=False).head(6)

    st.dataframe(shap_df, use_container_width=True)

    # Bar chart
    fig, ax = plt.subplots()
    ax.barh(shap_df["Feature"], shap_df["Impact"])
    ax.set_xlabel("Impact on Default Probability")
    ax.set_ylabel("Features")
    ax.invert_yaxis()
    st.pyplot(fig)

    # ========================================================
    # SIMULATOR
    # ========================================================
    st.markdown("## 🔄 What-if Simulator")

    new_payment = st.slider("Increase Monthly Payment (₹)", 0, int(BILL_AMT1), PAY_AMT1)

    temp = input_df.copy()
    temp["PAY_AMT1"] = new_payment

    sim = agent.run(temp)

    s1, s2 = st.columns(2)
    s1.metric("Current PD", f"{result['pd']:.2%}")
    s2.metric("New PD", f"{sim['result']['pd']:.2%}", delta=f"{sim['result']['pd']-result['pd']:.2%}")

    # ========================================================
    # REASONS & RECOMMENDATIONS
    # ========================================================
    st.markdown("## 📌 Key Risk Factors")
    for r in reasons:
        st.write("•", r)

    st.markdown("## 💡 Recommendations")
    for r in recs:
        st.write("•", r)

    # ========================================================
    # PDF (CIBIL-style with tables)
    # ========================================================
    def create_pdf():
        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "CREDIT RISK REPORT", ln=True, align="C")

        pdf.set_font("Arial", "", 10)
        pdf.cell(200, 8, f"Customer: {st.session_state.current_user}", ln=True)
        pdf.cell(200, 8, f"Generated: {pd.Timestamp.now()}", ln=True)
        pdf.ln(5)

        # Customer table
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "Customer Information", ln=True)

        pdf.set_font("Arial", "", 10)
        rows = [
            ("Credit Limit", LIMIT_BAL),
            ("Age", AGE),
            ("Latest Bill", BILL_AMT1),
            ("Latest Payment", PAY_AMT1)
        ]
        for k,v in rows:
            pdf.cell(100,8,k,1)
            pdf.cell(100,8,str(v),1,ln=True)

        pdf.ln(5)

        # Risk summary
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "Risk Summary", ln=True)

        rows = [
            ("PD", f"{result['pd']:.2%}"),
            ("Score", result["credit_score"]),
            ("Decision", result["decision"])
        ]
        for k,v in rows:
            pdf.cell(100,8,k,1)
            pdf.cell(100,8,str(v),1,ln=True)

        pdf.ln(5)

        # Factors
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "Risk Factors", ln=True)
        pdf.set_font("Arial", "", 10)
        for r in reasons:
            pdf.multi_cell(0,6,f"- {r}")

        pdf.ln(3)

        # Recommendations
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "Recommendations", ln=True)
        pdf.set_font("Arial", "", 10)
        for r in recs:
            pdf.multi_cell(0,6,f"- {r}")

        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        return BytesIO(pdf_bytes)

    st.download_button(
        "📄 Download Report",
        create_pdf(),
        file_name="credit_report.pdf",
        mime="application/pdf"
    )
