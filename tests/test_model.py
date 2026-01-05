import pickle
import pandas as pd

def test_model_prediction():
    with open("artifacts/final_model.pkl", "rb") as f:
        model = pickle.load(f)

    sample = pd.DataFrame([{
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

    prediction = model.predict(sample)
    probability = model.predict_proba(sample)

    assert prediction.shape == (1,)
    assert probability.shape == (1, 2)
