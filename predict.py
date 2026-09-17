import joblib
import pandas as pd


model = joblib.load("models/churn_model.joblib")


customer = pd.DataFrame(
    [
        {
            "tenure_months": 6,
            "monthly_charges": 95.0,
            "support_calls": 7,
            "contract_type": "Monthly",
        }
    ]
)


prediction = model.predict(customer)[0]

churn_probability = model.predict_proba(customer)[0][1]


print("Prediction:", prediction)
print("Churn probability:", round(churn_probability, 3))