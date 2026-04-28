from tools import (
    predict_credit_risk,
    generate_recommendations,
    generate_decision_summary,
    reasoning_engine
)


class CreditRiskAgent:

    def __init__(self, model, calibrator, scaler, feature_columns, qsvc_model=None):
        self.model = model
        self.calibrator = calibrator
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.qsvc_model = qsvc_model  # Optional (for 3-layer system)

    def run(self, input_df):

        # ============================================================
        # STEP 1: BASE MODEL (LightGBM + Calibration)
        # ============================================================

        result = predict_credit_risk(
            input_df,
            self.model,
            self.calibrator,
            self.scaler,
            self.feature_columns
        )

        # ============================================================
        # STEP 2: QSVC (Optional Layer - Risk Signal)
        # ============================================================

        qsvc_pred = None

        if self.qsvc_model is not None:
            try:
                input_scaled = self.scaler.transform(input_df)
                qsvc_pred = self.qsvc_model.predict(input_scaled)[0]
            except Exception:
                # fallback if kernel fails (very common in cloud)
                if result["pd"] > 0.25:
                    qsvc_pred = 1
                else:
                    qsvc_pred = 0

        # ============================================================
        # STEP 3: REASONING + RECOMMENDATIONS
        # ============================================================

        reasons = reasoning_engine(input_df)
        recommendations = generate_recommendations(input_df, result)
        summary = generate_decision_summary(result, recommendations)

        # ============================================================
        # STEP 4: FINAL OUTPUT STRUCTURE
        # ============================================================

        return {
            "result": result,                  # PD, score, decision etc
            "qsvc_signal": qsvc_pred,          # Quantum risk signal
            "reasons": reasons,               # Explainability (rules)
            "recommendations": recommendations,
            "summary": summary
        }
