"""
FastAPI application for serving fraud detection model.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import numpy as np

# Define request schema
class Transaction(BaseModel):
    features: list

app = FastAPI()

# Load artifacts
model = joblib.load(os.getenv('MODEL_PATH', 'artifacts/model.joblib'))
preprocessor = joblib.load(os.getenv('PREPROCESSOR_PATH', 'artifacts/preprocessor.joblib'))

@app.post("/predict")
def predict(tx: Transaction):
    """Predict fraud probability for a transaction."""
    X = np.array(tx.features).reshape(1, -1)
    X_proc = preprocessor.transform(X)
    pred = model.predict(X_proc)
    # IsolationForest returns -1 for anomaly
    prob = float(pred[0] == -1)
    return {"fraud": bool(prob), "fraud_score": prob}

@app.get("/health")
def health():
    return {"status": "ok"}
