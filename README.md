# MLOps Heart Disease Prediction

A production-ready machine learning project for heart disease prediction with MLOps best practices.

## Project Structure

```
mlops-heart-disease/
├── data/                    # Data files
│   ├── raw/                 # Raw data
│   ├── processed/           # Processed data
│   └── download_data.py     # Data download script
├── notebooks/               # Jupyter notebooks
│   ├── 01_eda.ipynb        # Exploratory data analysis
│   └── 02_model_training.ipynb  # Model training
├── src/                     # Source code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── api/                     # FastAPI application
│   ├── main.py
│   └── schema.py
├── tests/                   # Unit tests
│   ├── test_preprocessing.py
│   └── test_model.py
├── mlruns/                  # MLflow logs
├── .github/workflows/       # GitHub Actions CI/CD
├── k8s/                     # Kubernetes configurations
├── screenshots/             # Project screenshots
├── Dockerfile               # Docker configuration
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download data:
   ```bash
   python data/download_data.py
   ```

## Training

Run the model training:
```bash
python src/train.py
```

## API

Start the FastAPI server:
```bash
uvicorn api.main:app --reload
```

## Testing

Run tests:
```bash
pytest
```

## Docker

Build and run with Docker:
```bash
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

## Kubernetes

Deploy to Kubernetes:
```bash
kubectl apply -f k8s/deployment.yaml
```
