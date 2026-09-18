import pandas as pd

from scipy.stats import ks_2samp


# Reference data = what the model was trained on
reference = pd.read_csv("data/customers.csv")


# Simulate recent production traffic
current = reference.sample(
    n=300,
    random_state=42,
).copy()


# Simulate a change in customer behavior
current["monthly_charges"] = (
    current["monthly_charges"] * 1.25
)


def check_drift(feature):
    reference_values = reference[feature]
    current_values = current[feature]

    statistic, p_value = ks_2samp(
        reference_values,
        current_values,
    )

    print(f"\nFeature: {feature}")
    print(
        "Reference mean:",
        round(reference_values.mean(), 2),
    )
    print(
        "Current mean:",
        round(current_values.mean(), 2),
    )
    print(
        "KS statistic:",
        round(statistic, 3),
    )
    print(
        "p-value:",
        round(p_value, 5),
    )

    if p_value < 0.05:
        print("Drift detected")
    else:
        print("No significant drift detected")


check_drift("monthly_charges")
check_drift("tenure_months")