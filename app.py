import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal


app = FastAPI()


# Load the trained model once when the application starts
model = joblib.load("models/churn_model.joblib")


# Define the expected customer input
class CustomerInput(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0)
    support_calls: int = Field(ge=0)
    contract_type: Literal["Monthly", "Annual"]


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(customer: CustomerInput):

    customer_df = pd.DataFrame(
        [
            customer.model_dump()
        ]
    )

    prediction = model.predict(customer_df)[0]

    churn_probability = model.predict_proba(customer_df)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": round(
            float(churn_probability),
            3
        )
    }