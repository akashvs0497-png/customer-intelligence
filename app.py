import logging
import os
import time

from contextlib import asynccontextmanager

from httpx import request
import joblib
import pandas as pd

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field
from typing import Literal

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


# -------------------------
# Logging configuration
# -------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)

# -------------------------
# Monitoring metrics
# -------------------------

CHURN_REQUESTS = Counter(
    "churn_prediction_requests_total",
    "Total number of churn prediction requests",
)

CHURN_PREDICTIONS = Counter(
    "churn_predictions_total",
    "Total churn predictions by result",
    ["prediction"],
)

SENTIMENT_REQUESTS = Counter(
    "sentiment_requests_total",
    "Total number of sentiment requests",
)

SENTIMENT_PREDICTIONS = Counter(
    "sentiment_predictions_total",
    "Total sentiment predictions by result",
    ["sentiment"],
)

INFERENCE_LATENCY = Histogram(
    "model_inference_seconds",
    "Model inference latency in seconds",
    ["model"],
)


# -------------------------
# Model lifecycle
# -------------------------

model = None
sentiment_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    global sentiment_model

    logger.info("Loading ML models")

    try:
        model = joblib.load("models/churn_model.joblib")
        sentiment_model = joblib.load(
            "models/sentiment_model.joblib"
        )

        logger.info("ML models loaded successfully")

    except Exception:
        logger.exception("Failed to load ML models")
        raise

    yield

    model = None
    sentiment_model = None

    logger.info("Application shutdown complete")


# -------------------------
# Application
# -------------------------

app = FastAPI(lifespan=lifespan)


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

class SentimentInput(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=2000,
    )


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
    if model is None or sentiment_model is None:
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

@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )

# -------------------------
# Prediction endpoint
# -------------------------

@app.post("/predict")
def predict(customer: CustomerInput):
    CHURN_REQUESTS.inc()

    customer_df = pd.DataFrame(
        [
            customer.model_dump()
        ]
    )

    start_time = time.perf_counter()

    churn_probability = model.predict_proba(
        customer_df
    )[0][1]

    INFERENCE_LATENCY.labels(
        model="churn"
    ).observe(
        time.perf_counter() - start_time
    )

    prediction = int(
        churn_probability >= CHURN_THRESHOLD
    )

    CHURN_PREDICTIONS.labels(
        prediction=str(prediction)
    ).inc()

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

@app.post("/sentiment")
def analyze_sentiment(request: SentimentInput):
    SENTIMENT_REQUESTS.inc()

    start_time = time.perf_counter()

    prediction = sentiment_model.predict(
        [request.text]
    )[0]

    SENTIMENT_PREDICTIONS.labels(
        sentiment=str(prediction)
    ).inc()

    probabilities = sentiment_model.predict_proba(
        [request.text]
    )[0]

    INFERENCE_LATENCY.labels(
        model="sentiment"
    ).observe(
        time.perf_counter() - start_time
    )

    class_probabilities = dict(
        zip(
            sentiment_model.classes_,
            probabilities,
        )
    )

    confidence = class_probabilities[prediction]

    logger.info(
        "Sentiment prediction generated sentiment=%s confidence=%.3f",
        prediction,
        confidence,
    )

    return {
        "sentiment": str(prediction),
        "confidence": round(
            float(confidence),
            3,
        ),
    }