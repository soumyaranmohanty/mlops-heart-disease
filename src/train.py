import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/processed/heart_disease_cleaned.csv")

X = df.drop("target", axis=1)
y = df["target"]


binary_features = ["sex", "fbs", "exang"]
ordinal_features = ["cp", "restecg", "slope", "thal"]
discrete_numeric = ["ca"]

continuous_features = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), continuous_features),
        ("cat", "passthrough",
         binary_features + ordinal_features + discrete_numeric)
    ]
)

mlflow.end_run()

mlflow.set_experiment("Heart Disease Classification")


with mlflow.start_run(run_name="Logistic_Regression"):
    
    model_lr = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])


scoring = ["accuracy", "precision", "recall", "roc_auc"]

cv_results_lr = cross_validate(
    model_lr, X, y, cv=5, scoring=scoring
)

metrics_lr = {
    "accuracy": np.mean(cv_results_lr["test_accuracy"]),
    "precision": np.mean(cv_results_lr["test_precision"]),
    "recall": np.mean(cv_results_lr["test_recall"]),
    "roc_auc": np.mean(cv_results_lr["test_roc_auc"])
}


mlflow.log_param("model_type", "LogisticRegression")
mlflow.log_param("max_iter", 1000)

for k, v in metrics_lr.items():
    mlflow.log_metric(k, v)


model_lr.fit(X, y)
mlflow.sklearn.log_model(model_lr, "logistic_regression_model")
mlflow.end_run()

with mlflow.start_run(run_name="Random_Forest"):
    
    model_rf= Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ])


cv_results_rf = cross_validate(
    model_rf, X, y, cv=5, scoring=scoring
)

metrics_rf = {
    "accuracy": np.mean(cv_results_rf["test_accuracy"]),
    "precision": np.mean(cv_results_rf["test_precision"]),
    "recall": np.mean(cv_results_rf["test_recall"]),
    "roc_auc": np.mean(cv_results_rf["test_roc_auc"])
}

mlflow.log_param("model_type", "RandomForest")
mlflow.log_param("n_estimators", 200)

for k, v in metrics_rf.items():
    mlflow.log_metric(k, v)

model_rf.fit(X, y)
mlflow.sklearn.log_model(model_rf, "Random_Forest_model")
mlflow.end_run()


