import pickle

with open("artifacts/final_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully")
