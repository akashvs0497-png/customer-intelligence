import numpy as np
import pandas as pd


np.random.seed(42)

number_of_customers = 1000

data = {
    "customer_id": [f"CUST{i:04d}" for i in range(1, number_of_customers + 1)],
    "tenure_months": np.random.randint(1, 73, number_of_customers),
    "monthly_charges": np.random.uniform(20, 120, number_of_customers).round(2),
    "support_calls": np.random.randint(0, 10, number_of_customers),
    "contract_type": np.random.choice(
        ["Monthly", "Annual"],
        number_of_customers
    )
}

df = pd.DataFrame(data)

churn_probability = (
    0.15
    + (df["tenure_months"] < 12) * 0.25
    + (df["monthly_charges"] > 80) * 0.20
    + (df["support_calls"] >= 5) * 0.20
    + (df["contract_type"] == "Monthly") * 0.15
)

churn_probability = churn_probability.clip(0, 0.95)

df["churn"] = np.random.binomial(
    1,
    churn_probability
)

df.to_csv("data/customers.csv", index=False)

print("Dataset created successfully")
print("Rows:", len(df))
print("Churned:", df["churn"].sum())
print("Stayed:", (df["churn"] == 0).sum())