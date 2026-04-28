from tools import run_credit_assessment, generate_reasons, generate_recommendations


class CreditRiskAgent:
    def __init__(self, model, calibrator, scaler, feature_columns, qsvc_model=None):
        self.model = model
        self.calibrator = calibrator
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.qsvc_model = qsvc_model

    def run(self, data):
        result = run_credit_assessment(
            data=data,
            model=self.model,
            calibrator=self.calibrator,
            scaler=self.scaler,
            feature_columns=self.feature_columns,
            qsvc_model=self.qsvc_model
        )

        reasons = generate_reasons(data, result)
        recommendations = generate_recommendations(data, result)

        return {
            "result": result,
            "reasons": reasons,
            "recommendations": recommendations
        }
