import joblib
import os
import pandas as pd
from collections import Counter

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression_heart_disease_model.joblib",
    "Random Forest": "random_forest_heart_disease_model.joblib",
    "Gradient Boosting": "gradient_boosting_heart_disease_model.joblib",
    "XGBoost": "xgboost_heart_disease_model.joblib"
}

# Load all models once
MODELS = {
    name: joblib.load(os.path.join(MODEL_DIR, file))
    for name, file in MODEL_FILES.items()
}


def ensemble_predict(input_df: pd.DataFrame):
    predictions = {}
    probabilities = {}

    for name, model in MODELS.items():
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]  # Probability of Heart Disease
        predictions[name] = "Heart Disease" if pred == 1 else "No Heart Disease"
        probabilities[name] = round(prob * 100, 2)

    # Count votes
    vote_counts = Counter(predictions.values())
    top_vote = vote_counts.most_common(1)[0]

    # Check for tie (e.g., 2 vs 2)
    if len(vote_counts) > 1 and list(vote_counts.values()).count(top_vote[1]) > 1:
        # Find model with highest confidence
        most_confident_model = max(probabilities, key=probabilities.get)
        suggested_label = predictions[most_confident_model]
        details = (
            f"⚠️ Conflicting predictions — results inconclusive.\n\n"
            f"Most confident model: `{most_confident_model}` suggests **{suggested_label}** "
            f"with {probabilities[most_confident_model]}% confidence.\n\n"
            f"Please consult a medical professional."
        )
        return {
            "label": "Conflicting",
            "details": details,
            "avg_confidence": probabilities[most_confident_model],  # Return highest confidence here
            "raw": {"predictions": predictions, "probabilities": probabilities}
        }

    # Majority result
    final_label = top_vote[0]
    agreeing_models = [name for name, lbl in predictions.items() if lbl == final_label]
    avg_conf = round(
        sum(probabilities[name] for name in agreeing_models) / len(agreeing_models), 2
    )
    details = (
        f"✅ Majority of models ({len(agreeing_models)} out of 4) predict: **{final_label}**\n\n"
        f"📊 Average confidence among agreeing models: **{avg_conf}%**"
    )

    return {
        "label": final_label,
        "details": details,
        "avg_confidence": avg_conf,  # This is the key your app expects
        "raw": {"predictions": predictions, "probabilities": probabilities}
    }
