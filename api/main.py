from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Initialize app
app = FastAPI(title="Heart Disease Prediction API")

# Load model at startup
with open("artifacts/final_model.pkl", "rb") as f:
    model = pickle.load(f)

# Input schema
class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Health check
@app.get("/")
def health():
    return {"status": "API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: HeartInput):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4)
    }
