import numpy as np
import pandas as pd

LGD_DEFAULT = 0.45
BASEL_CAR_THRESHOLD = 10.5


def rupee(x):
    return f"Rs. {float(x):,.0f}"


def safe_text(x):
    return str(x).encode("latin-1", "replace").decode("latin-1")


def calculate_emi(principal, annual_rate, tenure_years):
    monthly_rate = annual_rate / 12 / 100
    months = tenure_years * 12

    if principal <= 0 or months <= 0:
        return 0

    if monthly_rate == 0:
        return principal / months

    return (principal * monthly_rate * (1 + monthly_rate) ** months) / (
        ((1 + monthly_rate) ** months) - 1
    )


def calculate_foir(existing_emi, new_emi, monthly_income):
    if monthly_income <= 0:
        return 100
    return ((existing_emi + new_emi) / monthly_income) * 100


def calculate_utilisation(outstanding, limit):
    return (outstanding / max(limit, 1)) * 100


def get_score_band(score):
    if score >= 750:
        return "Excellent"
    if score >= 700:
        return "Good"
    if score >= 650:
        return "Fair"
    if score >= 600:
        return "Poor"
    return "Very Poor"


def get_risk_band(pd_value):
    if pd_value < 0.10:
        return "Low Risk"
    if pd_value < 0.25:
        return "Medium Risk"
    return "High Risk"


def derive_dpd_from_history(payment_df):
    if payment_df is None or payment_df.empty:
        return 0

    statuses = payment_df["Payment Status"].astype(str).tolist()

    if "Missed" in statuses:
        return 90
    if "Delay" in statuses:
        return 30
    return 0


def classify_stage(pd_value, days_past_due):
    if days_past_due >= 90:
        return "Stage 3 - Default"
    if days_past_due >= 30 or pd_value >= 0.20:
        return "Stage 2 - Significant Increase in Credit Risk"
    return "Stage 1 - Performing"


def compute_ecl(pd_value, lgd, ead, stage):
    if "Stage 1" in stage:
        return pd_value * lgd * ead * 0.30
    if "Stage 2" in stage:
        return pd_value * lgd * ead * 0.70
    return pd_value * lgd * ead


def get_risk_weight(pd_value):
    if pd_value < 0.02:
        return 0.20
    if pd_value < 0.05:
        return 0.50
    if pd_value < 0.10:
        return 0.75
    if pd_value < 0.20:
        return 1.00
    return 1.50


def compute_rwa(ead, pd_value):
    risk_weight = get_risk_weight(pd_value)
    return ead * risk_weight, risk_weight


def compute_car(capital, rwa):
    if rwa <= 0:
        return 0
    return (capital / rwa) * 100


def apply_financial_adjustment(pd_value, savings, mutual_funds, stocks, property_status, vehicle_status):
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

    return min(max(adjusted_pd, 0.01), 0.95)


def final_decision_engine(pd_value, qsvc_signal, foir, utilisation, stage, car):
    if "Stage 3" in stage:
        return "Reject", "Stage 3 default classification"

    if foir > 55:
        return "Reject", "High FOIR indicates weak repayment capacity"

    if utilisation > 80:
        return "Reject", "Credit utilisation is very high"

    if car < BASEL_CAR_THRESHOLD:
        return "Manual Review", "Capital adequacy is below simplified Basel threshold"

    if pd_value > 0.40:
        return "Reject", "High probability of default"

    if qsvc_signal == 1 and pd_value >= 0.20:
        return "Reject", "QSVC high-risk signal with elevated PD"

    if pd_value > 0.20 or foir > 40 or "Stage 2" in stage:
        return "Manual Review", "Moderate credit risk or repayment burden"

    if qsvc_signal == 1:
        return "Manual Review", "QSVC risk signal requires analyst review"

    return "Approve", "Meets risk, affordability, and capital criteria"


def simulate_credit_losses(pd_value, lgd, ead, n_simulations=10000, confidence_level=0.99):
    pd_value = min(max(pd_value, 0.0001), 0.9999)

    losses = np.random.binomial(
        n=1,
        p=pd_value,
        size=n_simulations
    ) * lgd * ead

    expected_loss = np.mean(losses)
    var_loss = np.percentile(losses, confidence_level * 100)
    economic_capital = var_loss - expected_loss

    return {
        "losses": losses,
        "expected_loss": expected_loss,
        "var_loss": var_loss,
        "economic_capital": economic_capital,
        "confidence_level": confidence_level
    }


def apply_stress_scenario(pd_value, lgd, ead, scenario):
    scenario_map = {
        "Base Case": (1.00, 1.00, 1.00, "Normal economic conditions"),
        "Mild Stress": (1.25, 1.05, 1.00, "Mild income pressure and repayment stress"),
        "Recession": (1.75, 1.15, 1.05, "Recession with higher default pressure"),
        "Severe Recession": (2.50, 1.30, 1.10, "Severe macroeconomic shock"),
        "High Interest Rate Shock": (1.50, 1.10, 1.00, "Interest rate rise increases EMI stress")
    }

    pd_mult, lgd_mult, ead_mult, desc = scenario_map[scenario]

    stressed_pd = min(pd_value * pd_mult, 0.95)
    stressed_lgd = min(lgd * lgd_mult, 0.95)
    stressed_ead = ead * ead_mult
    stressed_ecl = stressed_pd * stressed_lgd * stressed_ead

    return {
        "Scenario": scenario,
        "Description": desc,
        "Stressed PD": stressed_pd,
        "Stressed LGD": stressed_lgd,
        "Stressed EAD": stressed_ead,
        "Stressed ECL": stressed_ecl
    }


def build_model_input(data, feature_columns):
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


def run_credit_assessment(data, model, calibrator, scaler, feature_columns, qsvc_model=None):
    input_df = build_model_input(data, feature_columns)
    scaled = scaler.transform(input_df)

    lgb_probability = model.predict_proba(scaled)[:, 1][0]
    base_pd = calibrator.predict([lgb_probability])[0]
    base_pd = min(max(base_pd, 0.01), 0.95)

    try:
        qsvc_signal = qsvc_model.predict(scaled)[0] if qsvc_model is not None else (1 if base_pd > 0.25 else 0)
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

    ead = data["REQUESTED_LOAN_AMOUNT"]
    stage = classify_stage(adjusted_pd, data["DPD"])
    ecl = compute_ecl(adjusted_pd, LGD_DEFAULT, ead, stage)
    rwa, risk_weight = compute_rwa(ead, adjusted_pd)
    car = compute_car(data["CAPITAL"], rwa)

    decision, decision_reason = final_decision_engine(
        adjusted_pd,
        qsvc_signal,
        foir,
        utilisation,
        stage,
        car
    )

    score = int(300 + (1 - adjusted_pd) * 600)

    eligible_loan_multiplier = data["MONTHLY_INCOME"] * 15
    max_new_emi_40_rule = max((data["MONTHLY_INCOME"] * 0.40) - data["EXISTING_EMI"], 0)

    var_result = simulate_credit_losses(adjusted_pd, LGD_DEFAULT, ead, 10000, 0.99)

    return {
        "input_df": input_df,
        "scaled": scaled,
        "lgb_probability": lgb_probability,
        "base_pd": base_pd,
        "adjusted_pd": adjusted_pd,
        "qsvc_signal": qsvc_signal,
        "score": score,
        "risk_band": get_risk_band(adjusted_pd),
        "score_band": get_score_band(score),
        "decision": decision,
        "decision_reason": decision_reason,
        "new_emi": new_emi,
        "foir": foir,
        "utilisation": utilisation,
        "eligible_loan_multiplier": eligible_loan_multiplier,
        "max_new_emi_40_rule": max_new_emi_40_rule,
        "stage": stage,
        "ecl": ecl,
        "ead": ead,
        "lgd": LGD_DEFAULT,
        "rwa": rwa,
        "risk_weight": risk_weight,
        "car": car,
        "var_99": var_result["var_loss"],
        "economic_capital": var_result["economic_capital"]
    }


def generate_reasons(data, result):
    reasons = []

    if data["PAY_0"] > 0:
        reasons.append("Recent repayment delay detected.")

    if result["utilisation"] > 70:
        reasons.append("High credit utilisation above 70%.")

    if result["foir"] > 40:
        reasons.append("FOIR is above the preferred range.")

    if "Stage 2" in result["stage"]:
        reasons.append("Exposure is classified as Stage 2 due to increased credit risk.")

    if "Stage 3" in result["stage"]:
        reasons.append("Exposure is classified as Stage 3 due to default-level delinquency.")

    if result["car"] < BASEL_CAR_THRESHOLD:
        reasons.append("Capital adequacy is below simplified Basel threshold.")

    if not reasons:
        reasons.append("No major adverse credit risk signal detected.")

    return reasons


def generate_recommendations(data, result):
    recs = []

    if result["utilisation"] > 50:
        recs.append("Reduce outstanding balance to keep utilisation below 50%.")

    if result["foir"] > 40:
        recs.append("Reduce existing EMI burden or request a smaller loan amount.")

    if data["PAY_0"] > 0:
        recs.append("Maintain timely repayment for the next billing cycles.")

    if result["car"] < BASEL_CAR_THRESHOLD:
        recs.append("Increase capital allocation or reduce exposure.")

    if not recs:
        recs.append("Maintain current repayment behaviour and avoid increasing utilisation.")

    return recs
