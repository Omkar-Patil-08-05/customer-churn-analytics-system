from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import traceback
import pandas as pd

from src.models.inference import predict_churn

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using best ML model.",
    version="1.0"
)

# --------------------------
# CORS SETTINGS
# --------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# INPUT SCHEMA
# --------------------------
class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# --------------------------
# HEALTH CHECK
# --------------------------
@app.get("/")
def health_check():
    return {
        "status": "alive",
        "message": "Churn API running successfully",
        "docs": "/docs",
    }


# --------------------------
# PREDICTION ENDPOINT
# --------------------------
@app.post("/predict")
def predict_churn_api(customer: CustomerInput):
    try:
        input_dict = customer.dict()

        # Convert all categorical fields to string (Avoid np.isnan crash)
        for k, v in input_dict.items():
            if isinstance(v, str):
                input_dict[k] = v.strip()

        # Ensure DataFrame columns match the training schema exactly
        df = pd.DataFrame([input_dict])

        print("\n===== INFERENCE REQUEST RECEIVED =====")
        print(df)
        print("======================================\n")

        result = predict_churn(input_dict)

        return {
            "model_used": result["best_model"],
            "prediction": result["prediction"],
            "churn_probability": result["churn_probability"],
            "input_received": input_dict
        }

    except Exception as e:
        print("\n🔥🔥🔥 ERROR IN /predict 🔥🔥🔥")
        print("Error:", e)
        traceback.print_exc()
        print("--------------------------------------------------\n")

        raise HTTPException(status_code=500, detail=str(e))
