import json
import joblib
import pandas as pd
import os


def get_best_model_name(path="reports/model_performance.json"):
    with open(path, "r") as f:
        metrics = json.load(f)
    return max(metrics, key=lambda m: metrics[m]["f1_1"])


def load_model(model_name):
    return joblib.load(f"models/{model_name}.pkl")


# -----------------------------
# FIX: Normalization function
# -----------------------------
def normalize_input(input_dict):
    """
    Ensures API input matches training format.
    Converts categories, strips spaces, fixes types.
    """

    df = pd.DataFrame([input_dict])

    # Convert all string fields to same format training used
    str_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod"
    ]

    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Ensure numeric fields are numeric
    numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace NaN with something consistent
    df.fillna("None", inplace=True)

    return df


# -----------------------------
# Prediction function
# -----------------------------
def predict_churn(input_dict):

    df = normalize_input(input_dict)

    best_model = get_best_model_name()
    model = load_model(best_model)

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "best_model": best_model,
        "prediction": int(pred),
        "churn_probability": float(prob)
    }
