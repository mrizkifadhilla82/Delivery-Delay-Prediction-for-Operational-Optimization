import os
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import warnings
from datetime import datetime

# Disable Keras warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# TENSORFLOW & SCIKERAS DESERIALIZATION SETUP
# ─────────────────────────────────────────────
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier # noqa: F401

def create_model(optimizer="adam", dropout_rate=0.2, input_shape=None):
    m = Sequential([
        Dense(64, activation="relu", input_shape=(input_shape,) if input_shape else (None,)),
        Dropout(dropout_rate),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    m.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return m

# Resolve paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "delivery_delay_model.pkl"
LOGS_DIR = BASE_DIR / "logs"
LOGS_FILE = LOGS_DIR / "prediction_logs.csv"

import sys
sys.modules['__main__'].create_model = create_model

# Load the pipeline model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Make sure to run your notebook first.")

model = joblib.load(MODEL_PATH)

# Ensure logs directory exists
LOGS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# FASTAPI APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Delivery Delay Serving API",
    description="MLOps API endpoint for predicting delivery delays using a TensorFlow Neural Network model.",
    version="1.0.0"
)

# Pydantic input schema
class DeliveryInput(BaseModel):
    delivery_partner: str
    package_type: str
    vehicle_type: str
    delivery_mode: str
    region: str
    weather_condition: str
    distance_km: float
    package_weight_kg: float
    delivery_cost: float

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": True
    }

@app.post("/predict")
def predict(payload: DeliveryInput):
    try:
        # Convert inputs to DataFrame matching pipeline features
        input_data = pd.DataFrame([{
            "delivery_partner": payload.delivery_partner,
            "package_type": payload.package_type,
            "vehicle_type": payload.vehicle_type,
            "delivery_mode": payload.delivery_mode,
            "region": payload.region,
            "weather_condition": payload.weather_condition,
            "distance_km": payload.distance_km,
            "package_weight_kg": payload.package_weight_kg,
            "delivery_cost": payload.delivery_cost
        }])
        
        # Inference
        prob_raw = model.predict_proba(input_data)
        
        # Parse probabilities
        if hasattr(prob_raw, "shape") and len(prob_raw.shape) > 1 and prob_raw.shape[1] > 1:
            prob = float(prob_raw[0, 1])
        elif hasattr(prob_raw, "shape") and len(prob_raw.shape) > 1:
            prob = float(prob_raw[0, 0])
        else:
            prob = float(prob_raw[0])
            
        is_delayed = prob >= 0.5
        prediction_label = "Delayed" if is_delayed else "On Time"
        
        # MLOps Monitoring: Log the request and prediction
        log_data = {
            "timestamp": datetime.now().isoformat(),
            **payload.model_dump(),
            "predicted_probability": prob,
            "prediction": prediction_label
        }
        
        log_df = pd.DataFrame([log_data])
        
        # Append to prediction logs
        if not LOGS_FILE.exists():
            log_df.to_csv(LOGS_FILE, index=False)
        else:
            log_df.to_csv(LOGS_FILE, mode='a', header=False, index=False)
            
        return {
            "prediction": prediction_label,
            "probability": prob,
            "logged": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
