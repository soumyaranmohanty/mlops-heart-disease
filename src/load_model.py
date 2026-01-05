import mlflow.sklearn

RUN_ID = "9c3d75e5e4d345fd8c949312f97a04e1"

model_uri = f"/Users/soumya/Documents/mlops/mlops-heart-disease/mlruns/1/models/m-7d88a058609745c69bbfaa9df8c79f14/artifacts"
model = mlflow.sklearn.load_model(model_uri)

print("Model loaded successfully")


import pickle

with open("artifacts/final_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Final model saved to artifacts/final_model.pkl")



