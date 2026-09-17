import logging
import os

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal


# -------------------------
# Logging configuration
# -------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)


# -------------------------
# Application
# -------------------------

app = FastAPI()


# -------------------------
# Load model
# -------------------------

model = joblib.load("models/churn_model.joblib")


# -------------------------
# Configuration
# -------------------------

def get_churn_threshold():
    raw_value = os.getenv("CHURN_THRESHOLD", "0.5")

    try:
        threshold = float(raw_value)
    except ValueError:
        raise ValueError(
            f"CHURN_THRESHOLD must be a number, got: {raw_value}"
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"CHURN_THRESHOLD must be between 0 and 1, got: {threshold}"
        )

    return threshold


CHURN_THRESHOLD = get_churn_threshold()
SERVICE_VERSION = "1.0.0"
MODEL_VERSION = "1.0"

logger.info(
    "Application initialized with churn_threshold=%s",
    CHURN_THRESHOLD,
)


# -------------------------
# Request schema
# -------------------------

class CustomerInput(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0)
    support_calls: int = Field(ge=0)
    contract_type: Literal["Monthly", "Annual"]


# -------------------------
# Health endpoint
# -------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.get("/ready")
def ready():
    if model is None:
        return {
            "status": "not_ready"
        }

    return {
        "status": "ready"
    }

@app.get("/info")
def info():
    return {
        "service": "customer-intelligence",
        "service_version": SERVICE_VERSION,
        "model_version": MODEL_VERSION,
        "churn_threshold": CHURN_THRESHOLD,
    }

# -------------------------
# Prediction endpoint
# -------------------------

@app.post("/predict")
def predict(customer: CustomerInput):
    customer_df = pd.DataFrame(
        [
            customer.model_dump()
        ]
    )

    churn_probability = model.predict_proba(customer_df)[0][1]

    prediction = int(
        churn_probability >= CHURN_THRESHOLD
    )

    logger.info(
        "Prediction generated prediction=%s probability=%.3f threshold=%.2f",
        prediction,
        churn_probability,
        CHURN_THRESHOLD,
    )

    return {
        "prediction": prediction,
        "churn_probability": round(
            float(churn_probability),
            3
        )
    }