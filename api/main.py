# FastAPI application

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API"}

@app.post("/predict")
def predict(data):
    """Make prediction endpoint"""
    pass
