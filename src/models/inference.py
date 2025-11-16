import json
import joblib
import pandas as pd
import os

# Normalization mappings
CATEGORICAL_COLUMNS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def clean_input(df: pd.DataFrame):
    """
    Ensures:
    - All categorical columns become lowercase, stripped strings
    - All numeric columns become proper float/int
    """
    df = df.copy()

    # Clean categorical
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)   # force to string
                     .str.strip()
                     .str.lower()
            )

    # Clean numeric
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_best_model_name(report_path="reports/model_performance.json"):
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Performance report not found at {report_path}")

    with open(report_path, "r") as f:
        metrics = json.load(f)

    best_model = max(metrics, key=lambda m: metrics[m]["f1_1"])
    return best_model


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


def predict_churn(input_dict: dict):

    df = pd.DataFrame([input_dict])

    # CLEAN INPUT BEFORE PREDICTING
    df = clean_input(df)

    print("\n===== CLEANED INPUT DATA =====")
    print(df)
    print("======================================")

    best_model_name = get_best_model_name()
    model = load_model(best_model_name)

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "best_model": best_model_name,
        "prediction": int(pred),
        "churn_probability": float(prob)
    }


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
