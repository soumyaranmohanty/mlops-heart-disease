# Heart Disease Prediction Pipeline (MLOps)

This repository implements an end-to-end **Machine Learning & MLOps pipeline** for predicting heart disease using the **UCI Heart Disease (Cleveland) dataset**.  
It covers **data preprocessing, model training, experiment tracking, CI/CD, API deployment, containerization, Kubernetes deployment, and monitoring**.

---

## Dataset

- **Source:** UCI Heart Disease Dataset (processed.cleveland.data)
- **Features:** Clinical and demographic attributes
- **Target Variable:**
  - `0` → No heart disease
  - `1` → Presence of heart disease

---

## Task 1: Data Acquisition & Exploratory Data Analysis (EDA)

### Data Acquisition
- Dataset sourced from the UCI repository
- Cleaned and processed dataset stored at:
```
data/processed/heart_disease_cleaned.csv
```

### Data Cleaning
- Removal of missing values
- Proper encoding of categorical variables
- Binary target conversion

### Exploratory Data Analysis
- Feature distribution analysis
- Correlation heatmaps
- Class balance inspection

**Key Observations:**
- Mild class imbalance
- Features such as `thalach`, `oldpeak`, `exang`, and `cp` show strong association with the target
- Continuous features require scaling

EDA plots are saved under:
```
screenshots/
```

---

## Task 2: Data Preprocessing

Preprocessing is implemented using **scikit-learn Pipelines and ColumnTransformer**.

### Preprocessing Steps
- Continuous features scaled using `StandardScaler`
- Binary, ordinal, and discrete features passed through directly
- Preprocessing bundled inside the model pipeline

---

## Task 3: Model Training & Experiment Tracking

### Training Script
```
src/train.py
```

### Models Trained
- Logistic Regression
- Random Forest

### Experiment Tracking
- Implemented using **MLflow**
- Parameters, metrics, and model artifacts logged for each run

### Final Model Artifact
```
artifacts/final_model.pkl
```

---

## Task 4: Model Evaluation

Evaluation performed using cross-validation.

### Metrics
- Accuracy
- Precision
- Recall
- ROC-AUC

---

## Task 5: Reproducibility & Testing

Run tests using:
```
pytest
```

---

## Task 6: Model Serving & API (FastAPI)

### Endpoints
- `/` → Health check
- `/predict` → Prediction
- `/metrics` → Monitoring metrics

---

## Task 7: Containerization & Kubernetes Deployment

### Docker
```
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

### Kubernetes
```
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Access API:
```
http://localhost:30007
```

---

## Task 8: CI/CD Pipeline

- CI implemented using GitHub Actions
- CD implemented using a self-hosted runner
- Automated testing, training, image build, and deployment

---

## Task 9: Monitoring & Logging

- Logs accessible via:
```
kubectl logs <pod-name>
```
- Metrics exposed via `/metrics`

---

## Author
Soumya Ranjan
