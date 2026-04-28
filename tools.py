import numpy as np
import pandas as pd
from math import log, sqrt
from statistics import NormalDist

LGD_DEFAULT = 0.45
BASEL_CAR_THRESHOLD = 10.5
MATURITY_YEARS_DEFAULT = 2.5


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


def generate_pd_term_structure(pd_12m, years=5):
    rows = []
    pd_12m = min(max(pd_12m, 0.0001), 0.95)

    cumulative_pd = 0

    for year in range(1, years + 1):
        marginal_pd = min(pd_12m * (1 + 0.15 * (year - 1)), 0.95)
        cumulative_pd = 1 - ((1 - cumulative_pd) * (1 - marginal_pd))

        rows.append({
            "Year": year,
            "Marginal PD": marginal_pd,
            "Cumulative PD": cumulative_pd
        })

    return pd.DataFrame(rows)


def compute_ecl(pd_value, lgd, ead, stage, pd_term_df=None):
    if "Stage 1" in stage:
        # 12-month ECL
        return pd_value * lgd * ead, "12-month ECL"

    if "Stage 2" in stage:
        # Lifetime ECL using cumulative 5-year PD
        if pd_term_df is None or pd_term_df.empty:
            lifetime_pd = min(pd_value * 3, 0.95)
        else:
            lifetime_pd = float(pd_term_df["Cumulative PD"].iloc[-1])

        return lifetime_pd * lgd * ead, "Lifetime ECL"

    # Stage 3 default exposure
    return pd_value * lgd * ead, "Lifetime ECL - Default"


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


def basel_irb_capital(pd_value, lgd, ead, maturity_years=MATURITY_YEARS_DEFAULT):
    """
    Simplified Basel IRB corporate-style capital approximation.
    Used for academic demonstration, not regulatory reporting.
    """

    pd_value = min(max(pd_value, 0.0003), 0.999)
    lgd = min(max(lgd, 0.01), 0.95)

    n = NormalDist()

    # Basel-style asset correlation approximation
    correlation = (
        0.12 * (1 - np.exp(-50 * pd_value)) / (1 - np.exp(-50))
        + 0.24 * (1 - (1 - np.exp(-50 * pd_value)) / (1 - np.exp(-50)))
    )

    maturity_adjustment_b = (0.11852 - 0.05478 * log(pd_value)) ** 2
    maturity_adjustment = (1 + (maturity_years - 2.5) * maturity_adjustment_b) / (
        1 - 1.5 * maturity_adjustment_b
    )

    capital_requirement = lgd * (
        n.cdf(
            (1 / sqrt(1 - correlation)) * n.inv_cdf(pd_value)
            + sqrt(correlation / (1 - correlation)) * n.inv_cdf(0.999)
        )
    ) - (pd_value * lgd)

    capital_requirement = max(capital_requirement * maturity_adjustment, 0)
    irb_capital = capital_requirement * ead
    irb_rwa = irb_capital * 12.5

    return {
        "IRB Capital Requirement %": capital_requirement,
        "IRB Capital Amount": irb_capital,
        "IRB RWA": irb_rwa,
        "Asset Correlation": correlation,
        "Maturity Adjustment": maturity_adjustment
    }


def transition_matrix(stage):
    matrix = pd.DataFrame({
        "To Stage 1": [0.88, 0.18, 0.02],
        "To Stage 2": [0.10, 0.68, 0.13],
        "To Stage 3": [0.02, 0.14, 0.85]
    }, index=["From Stage 1", "From Stage 2", "From Stage 3"])

    if "Stage 1" in stage:
        current = "From Stage 1"
    elif "Stage 2" in stage:
        current = "From Stage 2"
    else:
        current = "From Stage 3"

    return matrix, current


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

    pd_term_df = generate_pd_term_structure(adjusted_pd, years=5)
    lifetime_pd = float(pd_term_df["Cumulative PD"].iloc[-1])

    ecl, ecl_type = compute_ecl(adjusted_pd, LGD_DEFAULT, ead, stage, pd_term_df)

    standardised_rwa, risk_weight = compute_rwa(ead, adjusted_pd)
    standardised_car = compute_car(data["CAPITAL"], standardised_rwa)

    irb = basel_irb_capital(
        adjusted_pd,
        LGD_DEFAULT,
        ead,
        maturity_years=MATURITY_YEARS_DEFAULT
    )

    irb_car = compute_car(data["CAPITAL"], irb["IRB RWA"])

    decision, decision_reason = final_decision_engine(
        adjusted_pd,
        qsvc_signal,
        foir,
        utilisation,
        stage,
        standardised_car
    )

    score = int(300 + (1 - adjusted_pd) * 600)

    eligible_loan_multiplier = data["MONTHLY_INCOME"] * 15
    max_new_emi_40_rule = max((data["MONTHLY_INCOME"] * 0.40) - data["EXISTING_EMI"], 0)

    var_result = simulate_credit_losses(adjusted_pd, LGD_DEFAULT, ead, 10000, 0.99)

    matrix, current_stage_row = transition_matrix(stage)

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
        "pd_term_df": pd_term_df,
        "lifetime_pd": lifetime_pd,
        "ecl": ecl,
        "ecl_type": ecl_type,
        "ead": ead,
        "lgd": LGD_DEFAULT,
        "standardised_rwa": standardised_rwa,
        "risk_weight": risk_weight,
        "standardised_car": standardised_car,
        "irb_capital_requirement_pct": irb["IRB Capital Requirement %"],
        "irb_capital_amount": irb["IRB Capital Amount"],
        "irb_rwa": irb["IRB RWA"],
        "irb_car": irb_car,
        "asset_correlation": irb["Asset Correlation"],
        "maturity_adjustment": irb["Maturity Adjustment"],
        "var_99": var_result["var_loss"],
        "economic_capital": var_result["economic_capital"],
        "transition_matrix": matrix,
        "current_stage_row": current_stage_row
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
        reasons.append("Exposure is classified as Stage 2 due to significant increase in credit risk.")

    if "Stage 3" in result["stage"]:
        reasons.append("Exposure is classified as Stage 3 due to default-level delinquency.")

    if result["standardised_car"] < BASEL_CAR_THRESHOLD:
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

    if result["standardised_car"] < BASEL_CAR_THRESHOLD:
        recs.append("Increase capital allocation or reduce exposure.")

    if "Stage 2" in result["stage"]:
        recs.append("Monitor exposure due to significant increase in credit risk.")

    if "Stage 3" in result["stage"]:
        recs.append("Classify exposure for recovery action and default management.")

    if not recs:
        recs.append("Maintain current repayment behaviour and avoid increasing utilisation.")

    return recs
