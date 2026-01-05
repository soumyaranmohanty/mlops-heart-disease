import pickle

with open("artifacts/final_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)


import pandas as pd

sample_input = pd.DataFrame([{
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 0,
    "thal": 3
}])



prediction = loaded_model.predict(sample_input)
probability = loaded_model.predict_proba(sample_input)

print("Prediction:", prediction)
print("Probability:", probability)