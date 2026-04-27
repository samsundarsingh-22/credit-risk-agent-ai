import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from agent import CreditRiskAgent

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="Agentic AI Credit Risk System", layout="wide")

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
# SIDEBAR
# ============================================================

st.sidebar.title("🔐 Customer Session")

customer_name = st.sidebar.text_input("Customer Name", value="Demo Customer")
application_id = st.sidebar.text_input("Application ID", value="APP-001")

st.sidebar.success("Session Active")

st.sidebar.markdown("---")
st.sidebar.title("📡 Live Risk Engine")

st.sidebar.write("Model: LightGBM")
st.sidebar.write("Calibration: Isotonic")
st.sidebar.write("Explainability: SHAP")

# ============================================================
# TITLE
# ============================================================

st.title("🏦 Agentic AI Credit Risk Decision System")

st.info("AI-powered credit risk evaluation with explainable decisions.")

# ============================================================
# OPTION MAPS
# ============================================================

sex_map = {"Male": 1, "Female": 2}
education_map = {
    "Graduate School": 1,
    "University": 2,
    "High School": 3,
    "Others": 4
}
marriage_map = {"Married": 1, "Single": 2, "Others": 3}
repayment_map = {
    "Paid on time": 0,
    "1 month delay": 1,
    "2 months delay": 2,
    "3+ months delay": 3
}

# ============================================================
# INPUTS
# ============================================================

st.header("👤 Customer Profile")

col1, col2 = st.columns(2)

with col1:
    LIMIT_BAL = st.number_input("Credit Limit (₹)", value=200000)
    AGE = st.number_input("Age", value=35)

with col2:
    SEX = sex_map[st.selectbox("Sex", sex_map.keys())]
    EDUCATION = education_map[st.selectbox("Education", education_map.keys())]
    MARRIAGE = marriage_map[st.selectbox("Marriage", marriage_map.keys())]

# Repayment
st.header("⏱️ Repayment Behavior")

PAY_0 = repayment_map[st.selectbox("Recent Month", repayment_map.keys())]
PAY_2 = repayment_map[st.selectbox("Previous Month", repayment_map.keys())]

# Financial
st.header("💳 Financials")

BILL_AMT1 = st.number_input("Latest Bill Amount", value=50000)
PAY_AMT1 = st.number_input("Latest Payment Amount", value=5000)

# ============================================================
# SESSION STATE
# ============================================================

if "agent_output" not in st.session_state:
    st.session_state.agent_output = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================================================
# RUN MODEL
# ============================================================

if st.button("Run AI Decision"):

    input_data = {
        "LIMIT_BAL": LIMIT_BAL,
        "SEX": SEX,
        "EDUCATION": EDUCATION,
        "MARRIAGE": MARRIAGE,
        "AGE": AGE,
        "PAY_0": PAY_0,
        "PAY_2": PAY_2,
        "BILL_AMT1": BILL_AMT1,
        "PAY_AMT1": PAY_AMT1
    }

    # Fill missing columns safely (VERY IMPORTANT FIX)
    for col in feature_columns:
        if col not in input_data:
            input_data[col] = 0

    input_df = pd.DataFrame([input_data])[feature_columns]

    agent_output = agent.run(input_df)

    st.session_state.agent_output = agent_output
    st.session_state.input_df = input_df

# ============================================================
# DISPLAY OUTPUT
# ============================================================

if st.session_state.agent_output:

    agent_output = st.session_state.agent_output

    result = agent_output["result"]
    recommendations = agent_output["recommendations"]
    summary = agent_output["summary"]
    reasons = agent_output.get("reasons", [])

    # ======================
    # METRICS
    # ======================

    st.header("AI Decision Dashboard")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Score", result["credit_score"])
    c2.metric("PD", f"{result['pd']:.2%}")
    c3.metric("Loss", f"₹{result['expected_loss']:,.0f}")
    c4.metric("Risk", result["risk_bucket"])
    c5.metric("Decision", result["decision"])

    # Decision color
    if result["decision"] == "Approve":
        st.success("✅ Low Risk")
    elif result["decision"] == "Manual Review":
        st.warning("⚠️ Medium Risk")
    else:
        st.error("❌ High Risk")

    # ======================
    # SHAP (FIXED)
    # ======================

    st.header("🔍 Feature Impact (Explainability)")

    input_scaled = scaler.transform(st.session_state.input_df)

    shap_values = explainer.shap_values(input_scaled)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Bar plot
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_scaled, plot_type="bar", show=False)
    st.pyplot(fig)

    # ======================
    # TEXT EXPLANATION
    # ======================

    st.subheader("Key Drivers")

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

    # ======================
    # REASONS
    # ======================

    st.subheader("🔍 Why this decision?")
    for r in reasons:
        st.write("•", r)

    # ======================
    # RECOMMENDATIONS
    # ======================

    st.subheader("📌 Recommendations")
    for r in recommendations:
        st.write("•", r)

    # ======================
    # REPORT
    # ======================

    st.subheader("📄 AI Report")
    st.text_area("Summary", summary, height=200)

    # ======================
    # CHATBOT
    # ======================

    st.header("💬 AI Assistant")

    user_question = st.chat_input("Ask about your credit risk...")

    def chatbot_response(q):
        q = q.lower()

        if "why" in q:
            return "\n".join(reasons)
        elif "improve" in q:
            return "\n".join(recommendations)
        elif "score" in q:
            return f"Your score is {result['credit_score']}"
        elif "risk" in q:
            return f"Risk level: {result['risk_bucket']}"
        else:
            return "Ask about risk, score, or improvements."

    if user_question:
        st.session_state.chat_history.append(("user", user_question))
        answer = chatbot_response(user_question)
        st.session_state.chat_history.append(("bot", answer))

    for role, msg in st.session_state.chat_history:
        if role == "user":
            with st.chat_message("user"):
                st.write(msg)
        else:
            with st.chat_message("assistant"):
                st.write(msg)
