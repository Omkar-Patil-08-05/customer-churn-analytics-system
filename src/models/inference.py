import json
import joblib
import pandas as pd
import os

# -----------------------------------------
# 1. Load best model name from metrics JSON
# -----------------------------------------
def get_best_model_name(report_path="reports/model_performance.json"):
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Performance report not found at {report_path}")

    with open(report_path, "r") as f:
        metrics = json.load(f)

    best_model = max(metrics, key=lambda m: metrics[m]["f1_1"])
    return best_model


# -----------------------------------------
# 2. Load the actual ML model
# -----------------------------------------
def load_model(model_name):
    model_path_map = {
        "logistic_regression": "models/logistic_regression.pkl",
        "random_forest": "models/random_forest.pkl",
        "xgboost": "models/xgboost.pkl"
    }

    model_path = model_path_map.get(model_name)

    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found for {model_name}")

    return joblib.load(model_path)


# -----------------------------------------
# 3. Prediction function
# -----------------------------------------
def predict_churn(input_dict: dict):

    # Convert JSON → DataFrame
    df = pd.DataFrame([input_dict])

    # No preprocessing here — model already contains preprocessing pipeline
    df_final = df.copy()

    # Load best model name
    best_model_name = get_best_model_name()

    # Load the model
    model = load_model(best_model_name)

    # Make prediction
    pred = model.predict(df_final)[0]
    prob = model.predict_proba(df_final)[0][1]

    return {
        "best_model": best_model_name,
        "prediction": int(pred),
        "churn_probability": float(prob)
    }


# -----------------------------------------
# 4. Manual test
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
