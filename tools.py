# ============================================================
# tools.py (FINAL - BANK STYLE LOGIC)
# ============================================================

import pandas as pd

LGD = 0.45  # Loss Given Default (industry standard assumption)

# ============================================================
# CORE MODEL PREDICTION
# ============================================================

def predict_credit_risk(input_df, model, calibrator, scaler, feature_columns):

    # Ensure correct feature order
    input_df = input_df[feature_columns]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # -------------------------------
    # Step 1: Base Model Prediction
    # -------------------------------
    raw_pd = model.predict_proba(input_scaled)[:, 1][0]

    # -------------------------------
    # Step 2: Calibration
    # -------------------------------
    calibrated_pd = calibrator.predict([raw_pd])[0]

    # Safety clamp (VERY IMPORTANT)
    calibrated_pd = min(max(calibrated_pd, 0.01), 0.95)

    # -------------------------------
    # Step 3: Derived Metrics
    # -------------------------------
    limit_bal = input_df["LIMIT_BAL"].iloc[0]

    expected_loss = calibrated_pd * limit_bal * LGD
    credit_score = int(300 + (1 - calibrated_pd) * 600)

    # -------------------------------
    # Step 4: Decision Logic (Aligned with UI)
    # -------------------------------
    if calibrated_pd < 0.10:
        risk_bucket = "Low Risk"
        decision = "Approve"
    elif calibrated_pd < 0.25:
        risk_bucket = "Medium Risk"
        decision = "Manual Review"
    else:
        risk_bucket = "High Risk"
        decision = "Reject"

    return {
        "pd": calibrated_pd,
        "credit_score": credit_score,
        "expected_loss": expected_loss,
        "risk_bucket": risk_bucket,
        "decision": decision
    }


# ============================================================
# RECOMMENDATIONS (BANK STYLE)
# ============================================================

def generate_recommendations(input_df, result):

    recs = []

    utilization = input_df["BILL_AMT1"].iloc[0] / max(input_df["LIMIT_BAL"].iloc[0], 1)
    repayment_ratio = input_df["PAY_AMT1"].iloc[0] / max(input_df["BILL_AMT1"].iloc[0], 1)

    # High risk actions
    if result["pd"] >= 0.25:
        recs.append("Reduce outstanding balances before applying for new credit.")
        recs.append("Ensure timely repayments across consecutive billing cycles.")
        recs.append("Avoid multiple credit enquiries within a short period.")

    # Repayment behavior
    if input_df["PAY_0"].iloc[0] > 0:
        recs.append("Recent repayment delay detected. Clearing dues promptly will reduce risk.")

    # Utilization
    if utilization > 0.70:
        recs.append("Credit utilisation is high (>70%). Reducing it below 50% can improve score.")

    # Payment strength
    if repayment_ratio < 0.10:
        recs.append("Low repayment relative to bill. Increasing monthly payments improves profile.")

    # Positive reinforcement
    if not recs:
        recs.append("Maintain current financial discipline to preserve strong credit profile.")

    return recs


# ============================================================
# DECISION SUMMARY (CIBIL STYLE)
# ============================================================

def generate_decision_summary(result, recommendations):

    summary = f"""
CREDIT RISK ASSESSMENT REPORT

Probability of Default (PD): {result['pd']:.2%}
AI Credit Score: {result['credit_score']}
Risk Category: {result['risk_bucket']}
Expected Loss (₹): {result['expected_loss']:,.0f}
Decision: {result['decision']}

INTERPRETATION:
The model evaluates repayment behaviour, utilisation, and financial patterns 
to estimate the probability of default. Lower PD indicates stronger creditworthiness.

RECOMMENDED ACTIONS:
"""

    for i, rec in enumerate(recommendations, 1):
        summary += f"\n{i}. {rec}"

    return summary


# ============================================================
# REASONING ENGINE (EXPLAINABLE AI TEXT)
# ============================================================

def reasoning_engine(input_df):

    reasons = []

    utilization = input_df["BILL_AMT1"].iloc[0] / max(input_df["LIMIT_BAL"].iloc[0], 1)
    repayment_ratio = input_df["PAY_AMT1"].iloc[0] / max(input_df["BILL_AMT1"].iloc[0], 1)

    if input_df["PAY_0"].iloc[0] > 0:
        reasons.append("Recent repayment delay observed, indicating increased default risk.")

    if utilization > 0.70:
        reasons.append("High credit utilisation ratio is associated with elevated risk.")

    if repayment_ratio < 0.10:
        reasons.append("Low repayment relative to outstanding balance indicates financial stress.")

    if input_df["AGE"].iloc[0] < 25:
        reasons.append("Limited credit history due to lower age may impact risk evaluation.")

    if not reasons:
        reasons.append("No major adverse credit behaviour detected. Profile appears stable.")

    return reasons
