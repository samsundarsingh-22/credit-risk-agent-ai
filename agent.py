from tools import (
    predict_credit_risk,
    generate_recommendations,
    generate_decision_summary,
    reasoning_engine
)

class CreditRiskAgent:

    def __init__(self, model, calibrator, scaler, feature_columns):
        self.model = model
        self.calibrator = calibrator
        self.scaler = scaler
        self.feature_columns = feature_columns

    def run(self, input_df):

        result = predict_credit_risk(
            input_df,
            self.model,
            self.calibrator,
            self.scaler,
            self.feature_columns
        )

        reasons = reasoning_engine(input_df)
        recommendations = generate_recommendations(input_df, result)
        summary = generate_decision_summary(result, recommendations)

        return {
            "result": result,
            "reasons": reasons,
            "recommendations": recommendations,
            "summary": summary
        }
