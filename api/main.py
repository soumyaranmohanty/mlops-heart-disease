from fastapi import FastAPI, Request
from pydantic import BaseModel
import pickle
import pandas as pd
import logging
import time

REQUEST_COUNT = 0
PREDICTION_COUNT = 0
START_TIME = time.time()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


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


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    response = await call_next(request)
    return response


# Health check
@app.get("/")
def health():
    logger.info("Health check endpoint called")
    return {"status": "API is running"}


# Prediction endpoint
@app.post("/predict")
def predict(data: HeartInput):
    global PREDICTION_COUNT
    PREDICTION_COUNT += 1

    logger.info(f"Prediction request received: {data.dict()}")

    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4)
    }

@app.get("/metrics")
def metrics():
    uptime = round(time.time() - START_TIME, 2)
    return {
        "uptime_seconds": uptime,
        "total_requests": REQUEST_COUNT,
        "total_predictions": PREDICTION_COUNT
    }


