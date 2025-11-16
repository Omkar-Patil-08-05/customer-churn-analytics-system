import json
import joblib
import pandas as pd
import os


# -----------------------------------------
# Load best model name from saved report
# -----------------------------------------
def get_best_model_name(path="reports/model_performance.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find metrics file at {path}")

    with open(path, "r") as f:
        metrics = json.load(f)

    best = max(metrics, key=lambda m: metrics[m]["f1_1"])
    return best


# -----------------------------------------
# Load trained model
# -----------------------------------------
def load_model(model_name):
    model_path = f"models/{model_name}.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing: {model_path}")

    return joblib.load(model_path)


# -----------------------------------------
# Prediction function
# -----------------------------------------
def predict_churn(input_dict: dict):

    df = pd.DataFrame([input_dict])

    # pick model
    best = get_best_model_name()
    model = load_model(best)

    # model includes preprocessing pipeline → safe!
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "best_model": best,
        "prediction": int(pred),
        "churn_probability": float(prob)
    }


# -----------------------------------------
# Manual test
# -----------------------------------------
if __name__ == "__main__":
    sample = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 1397.50
    }

    print(predict_churn(sample))
