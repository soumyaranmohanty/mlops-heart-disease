import pandas as pd

def test_data_schema():
    df = pd.read_csv("data/processed/heart_disease_cleaned.csv")

    expected_columns = {
        "age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal","target"
    }

    assert set(df.columns) == expected_columns


def test_no_missing_values():
    df = pd.read_csv("data/processed/heart_disease_cleaned.csv")
    assert df.isnull().sum().sum() == 0
