import streamlit as st
import pandas as pd
import joblib
from agent import CreditRiskAgent

# ============================================================
# LOAD MODEL FILES
# ============================================================

model = joblib.load("credit_model.pkl")
calibrator = joblib.load("calibrator.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ============================================================
# INITIALIZE AGENT
# ============================================================

agent = CreditRiskAgent(
    model=model,
    calibrator=calibrator,
    scaler=scaler,
    feature_columns=feature_columns
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Agentic AI Credit Risk System",
    layout="wide"
)

# ============================================================
# SIDEBAR SESSION PANEL
# ============================================================

st.sidebar.title("🔐 Customer Session")

customer_name = st.sidebar.text_input("Customer Name", value="Demo Customer")
application_id = st.sidebar.text_input("Application ID", value="APP-001")

st.sidebar.success("Session Active")

st.sidebar.markdown("---")
st.sidebar.title("📡 Live Risk Engine")

st.sidebar.write("✅ Model: LightGBM")
st.sidebar.write("✅ Probability Calibration: Isotonic")
st.sidebar.write("✅ Output: PD, Score, EL, Decision")
st.sidebar.write("✅ Agent: Reasoning + Recommendations")

# ============================================================
# TITLE
# ============================================================

st.title("🏦 Agentic AI Credit Risk Decision System")

st.write(
    "This system evaluates credit risk using a calibrated machine learning model "
    "and provides decision recommendations using an agentic AI layer."
)

st.warning(
    "Disclaimer: This is an AI-based risk estimate for research and educational purposes only. "
    "It is not an official credit bureau score."
)

st.info("""
### 📘 How to Fill This Form

- **Credit Limit**: Maximum approved credit amount.
- **Repayment Status**: Whether the customer paid on time or delayed payment.
- **Bill Amount**: Outstanding bill amount for each month.
- **Payment Amount**: Amount actually paid by the customer.

📅 Month 1 = Most recent month  
📅 Month 6 = Six months ago
""")

# ============================================================
# OPTION MAPS
# ============================================================

sex_map = {
    "Male": 1,
    "Female": 2
}

education_map = {
    "Graduate School": 1,
    "University": 2,
    "High School": 3,
    "Others": 4
}

marriage_map = {
    "Married": 1,
    "Single": 2,
    "Others": 3
}

repayment_map = {
    "Paid on time": 0,
    "1 month delay": 1,
    "2 months delay": 2,
    "3 or more months delay": 3
}

# ============================================================
# USER INPUTS
# ============================================================

st.header("👤 Customer Financial Profile")

col1, col2, col3 = st.columns(3)

with col1:
    LIMIT_BAL = st.number_input(
        "Approved Credit Limit / Exposure (₹)",
        min_value=0,
        value=200000,
        step=10000,
        help="Maximum credit limit approved for the customer."
    )

    SEX_WORD = st.selectbox("Sex", list(sex_map.keys()))
    SEX = sex_map[SEX_WORD]

    EDUCATION_WORD = st.selectbox("Education Level", list(education_map.keys()))
    EDUCATION = education_map[EDUCATION_WORD]

with col2:
    MARRIAGE_WORD = st.selectbox("Marriage Status", list(marriage_map.keys()))
    MARRIAGE = marriage_map[MARRIAGE_WORD]

    AGE = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

with col3:
    st.markdown("### 🧠 Quick Guide")
    st.write("A lower repayment delay generally means lower credit risk.")
    st.write("Higher bill amount with low payment may increase risk.")

# ============================================================
# REPAYMENT STATUS
# ============================================================

st.header("⏱️ Repayment Status - Last 6 Months")

r1, r2, r3 = st.columns(3)

with r1:
    PAY_0_WORD = st.selectbox("Month 1: Most Recent Repayment Status", list(repayment_map.keys()))
    PAY_2_WORD = st.selectbox("Month 2: Previous Month Repayment Status", list(repayment_map.keys()))

with r2:
    PAY_3_WORD = st.selectbox("Month 3: Repayment Status", list(repayment_map.keys()))
    PAY_4_WORD = st.selectbox("Month 4: Repayment Status", list(repayment_map.keys()))

with r3:
    PAY_5_WORD = st.selectbox("Month 5: Repayment Status", list(repayment_map.keys()))
    PAY_6_WORD = st.selectbox("Month 6: Oldest Repayment Status", list(repayment_map.keys()))

PAY_0 = repayment_map[PAY_0_WORD]
PAY_2 = repayment_map[PAY_2_WORD]
PAY_3 = repayment_map[PAY_3_WORD]
PAY_4 = repayment_map[PAY_4_WORD]
PAY_5 = repayment_map[PAY_5_WORD]
PAY_6 = repayment_map[PAY_6_WORD]

# ============================================================
# BILL AMOUNTS
# ============================================================

st.header("📊 Outstanding Bill Amounts - Last 6 Months")

b1, b2, b3 = st.columns(3)

with b1:
    BILL_AMT1 = st.number_input("Month 1: Most Recent Bill Amount (₹)", value=50000)
    BILL_AMT2 = st.number_input("Month 2: Bill Amount (₹)", value=48000)

with b2:
    BILL_AMT3 = st.number_input("Month 3: Bill Amount (₹)", value=45000)
    BILL_AMT4 = st.number_input("Month 4: Bill Amount (₹)", value=43000)

with b3:
    BILL_AMT5 = st.number_input("Month 5: Bill Amount (₹)", value=40000)
    BILL_AMT6 = st.number_input("Month 6: Oldest Bill Amount (₹)", value=38000)

# ============================================================
# PAYMENT AMOUNTS
# ============================================================

st.header("💳 Amount Paid - Last 6 Months")

p1, p2, p3 = st.columns(3)

with p1:
    PAY_AMT1 = st.number_input("Month 1: Most Recent Amount Paid (₹)", value=5000)
    PAY_AMT2 = st.number_input("Month 2: Amount Paid (₹)", value=5000)

with p2:
    PAY_AMT3 = st.number_input("Month 3: Amount Paid (₹)", value=5000)
    PAY_AMT4 = st.number_input("Month 4: Amount Paid (₹)", value=5000)

with p3:
    PAY_AMT5 = st.number_input("Month 5: Amount Paid (₹)", value=5000)
    PAY_AMT6 = st.number_input("Month 6: Oldest Amount Paid (₹)", value=5000)

# ============================================================
# SESSION STATE
# ============================================================

if "agent_output" not in st.session_state:
    st.session_state.agent_output = None

if "input_df" not in st.session_state:
    st.session_state.input_df = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================================================
# RUN AGENT
# ============================================================

if st.button("🚀 Run Agentic AI Decision"):

    input_data = {
        "LIMIT_BAL": LIMIT_BAL,
        "SEX": SEX,
        "EDUCATION": EDUCATION,
        "MARRIAGE": MARRIAGE,
        "AGE": AGE,
        "PAY_0": PAY_0,
        "PAY_2": PAY_2,
        "PAY_3": PAY_3,
        "PAY_4": PAY_4,
        "PAY_5": PAY_5,
        "PAY_6": PAY_6,
        "BILL_AMT1": BILL_AMT1,
        "BILL_AMT2": BILL_AMT2,
        "BILL_AMT3": BILL_AMT3,
        "BILL_AMT4": BILL_AMT4,
        "BILL_AMT5": BILL_AMT5,
        "BILL_AMT6": BILL_AMT6,
        "PAY_AMT1": PAY_AMT1,
        "PAY_AMT2": PAY_AMT2,
        "PAY_AMT3": PAY_AMT3,
        "PAY_AMT4": PAY_AMT4,
        "PAY_AMT5": PAY_AMT5,
        "PAY_AMT6": PAY_AMT6
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_columns]

    agent_output = agent.run(input_df)

    st.session_state.agent_output = agent_output
    st.session_state.input_df = input_df

# ============================================================
# DISPLAY AGENT OUTPUT
# ============================================================

if st.session_state.agent_output is not None:

    agent_output = st.session_state.agent_output

    result = agent_output["result"]
    recommendations = agent_output["recommendations"]
    summary = agent_output["summary"]
    reasons = agent_output.get("reasons", [])

    st.header("🧠 Agentic AI Decision Dashboard")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("AI Credit Score", result["credit_score"])
    c2.metric("Probability of Default", f"{result['pd']:.2%}")
    c3.metric("Expected Loss", f"₹{result['expected_loss']:,.2f}")
    c4.metric("Risk Category", result["risk_bucket"])
    c5.metric("Decision", result["decision"])

    if result["decision"] == "Approve":
        st.success("✅ Recommended Decision: Approve")
    elif result["decision"] == "Manual Review":
        st.warning("⚠️ Recommended Decision: Manual Review")
    else:
        st.error("❌ Recommended Decision: Reject / Further Review Required")

    st.subheader("🔍 Why this decision?")

    if reasons:
        for reason in reasons:
            st.write("•", reason)
    else:
        st.write("• No major risk factors detected.")

    st.subheader("✅ Agent Recommendations")

    for rec in recommendations:
        st.write("•", rec)

    st.subheader("📄 AI Generated Credit Risk Report")

    st.text_area("Decision Summary", summary, height=260)

    # ========================================================
    # CHATBOT UI
    # ========================================================

    st.header("💬 Credit Risk Chatbot Assistant")

    st.write(
        "Ask questions such as: "
        "**Why is my risk high?**, **How can I improve my score?**, "
        "**What is expected loss?**, or **Should the bank approve this customer?**"
    )

    def chatbot_response(user_question, result, reasons, recommendations):
        question = user_question.lower()

        if "why" in question or "reason" in question:
            response = "The decision is mainly based on these risk factors:\n"
            for r in reasons:
                response += f"- {r}\n"
            if not reasons:
                response += "- No major risk factors were detected.\n"
            return response

        elif "improve" in question or "increase score" in question or "better score" in question:
            response = "To improve the credit score, the customer should:\n"
            for r in recommendations:
                response += f"- {r}\n"
            return response

        elif "expected loss" in question or "loss" in question:
            return (
                f"Expected Loss is ₹{result['expected_loss']:,.2f}. "
                "It is calculated as PD × Exposure at Default × LGD."
            )

        elif "pd" in question or "probability" in question or "default" in question:
            return (
                f"The predicted probability of default is {result['pd']:.2%}. "
                "A higher PD indicates higher credit risk."
            )

        elif "approve" in question or "decision" in question or "reject" in question:
            return (
                f"The recommended decision is: {result['decision']}. "
                f"The customer is classified as {result['risk_bucket']}."
            )

        elif "score" in question:
            return (
                f"The AI credit score is {result['credit_score']}. "
                "The score is derived from calibrated default probability."
            )

        else:
            return (
                "I can help explain the credit decision, risk score, probability of default, "
                "expected loss, and ways to improve the credit profile."
            )

    user_question = st.chat_input("Ask the Credit Risk Assistant...")

    if user_question:
        st.session_state.chat_history.append(("user", user_question))

        bot_answer = chatbot_response(
            user_question,
            result,
            reasons,
            recommendations
        )

        st.session_state.chat_history.append(("assistant", bot_answer))

    for role, message in st.session_state.chat_history:
        if role == "user":
            with st.chat_message("user"):
                st.write(message)
        else:
            with st.chat_message("assistant"):
                st.write(message)
