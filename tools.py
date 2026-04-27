# tools.py

import pandas as pd

LGD = 0.45

def predict_credit_risk(input_df, model, calibrator, scaler, feature_columns):
    input_df = input_df[feature_columns]
    input_scaled = scaler.transform(input_df)

    raw_pd = model.predict_proba(input_scaled)[:, 1][0]
    calibrated_pd = calibrator.predict([raw_pd])[0]

    limit_bal = input_df["LIMIT_BAL"].iloc[0]
    expected_loss = calibrated_pd * limit_bal * LGD
    credit_score = int(300 + (1 - calibrated_pd) * 600)

    if calibrated_pd < 0.20:
        risk_bucket = "Low Risk"
        decision = "Approve"
    elif calibrated_pd < 0.40:
        risk_bucket = "Medium Risk"
        decision = "Manual Review"
    else:
        risk_bucket = "High Risk"
        decision = "Reject / Require Further Review"

    return {
        "pd": calibrated_pd,
        "credit_score": credit_score,
        "expected_loss": expected_loss,
        "risk_bucket": risk_bucket,
        "decision": decision
    }


def generate_recommendations(input_df, result):
    recs = []

    if result["pd"] >= 0.40:
        recs.append("Reduce outstanding bill amounts before applying for additional credit.")
        recs.append("Improve repayment consistency over the next few billing cycles.")
        recs.append("Avoid delayed payments, especially recent repayment delays.")

    if input_df["PAY_0"].iloc[0] > 0:
        recs.append("Recent repayment delay is a major risk factor. Clearing overdue payments can reduce risk.")

    if input_df["BILL_AMT1"].iloc[0] > 0.7 * input_df["LIMIT_BAL"].iloc[0]:
        recs.append("Credit utilization appears high. Reducing utilization below 50% may improve the score.")

    if input_df["PAY_AMT1"].iloc[0] < 0.1 * input_df["BILL_AMT1"].iloc[0]:
        recs.append("Recent repayment amount is low relative to bill amount. Increasing monthly repayment can improve risk profile.")

    if not recs:
        recs.append("Maintain current repayment behavior and avoid increasing credit utilization.")

    return recs


def generate_decision_summary(result, recommendations):
    summary = f"""
Credit Risk Decision Summary

Predicted Probability of Default: {result['pd']:.2%}
AI Credit Score: {result['credit_score']}
Risk Category: {result['risk_bucket']}
Expected Loss: ₹{result['expected_loss']:,.2f}
Recommended Decision: {result['decision']}

Recommended Actions:
"""

    for i, rec in enumerate(recommendations, start=1):
        summary += f"\n{i}. {rec}"

    return summary

def reasoning_engine(input_df):

    reasons = []

    if input_df["PAY_0"].iloc[0] > 0:
        reasons.append("Recent repayment delay increases default risk.")

    if input_df["BILL_AMT1"].iloc[0] > 0.7 * input_df["LIMIT_BAL"].iloc[0]:
        reasons.append("High credit utilization contributes to higher risk.")

    if input_df["PAY_AMT1"].iloc[0] < 0.1 * input_df["BILL_AMT1"].iloc[0]:
        reasons.append("Low repayment relative to bill amount increases risk.")

    if input_df["AGE"].iloc[0] < 25:
        reasons.append("Lower age group may have limited credit history.")

    if not reasons:
        reasons.append("No major risk factors detected. Financial behavior appears stable.")

    return reasons
