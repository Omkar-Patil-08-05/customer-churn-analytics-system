from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from src.models.inference import predict_churn

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using best ML model.",
    version="1.0"
)

# CORS (Allow frontend website to call the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # later we restrict this after deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Pydantic Input Schema
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
# Health Check Endpoint
# --------------------------
@app.get("/")
def health_check():
    return {"status": "alive", "message": "Churn API running successfully"}


# --------------------------
# Prediction Endpoint
# --------------------------
@app.post("/predict")
def predict_churn_api(customer: CustomerInput):
    result = predict_churn(customer.dict())
    return {
        "model_used": result["best_model"],
        "prediction": result["prediction"],
        "churn_probability": result["churn_probability"]
    }
