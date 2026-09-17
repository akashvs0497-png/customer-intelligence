import pandas as pd

import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# 1. Load customer data
df = pd.read_csv("data/customers.csv")


# 2. Separate features and target
X = df[
    [
        "tenure_months",
        "monthly_charges",
        "support_calls",
        "contract_type",
    ]
]

y = df["churn"]


# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# 4. Define categorical columns
categorical_features = [
    "contract_type",
]


# 5. Create preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "category",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        )
    ],
    remainder="passthrough",
)


# 6. Create Logistic Regression pipeline
logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)


# 7. Train Logistic Regression
logistic_model.fit(X_train, y_train)


# 8. Make Logistic Regression predictions
logistic_predictions = logistic_model.predict(X_test)

probabilities = logistic_model.predict_proba(X_test)[:, 1]

custom_predictions = (probabilities >= 0.4).astype(int)

print("\nFirst 10 churn probabilities:")

for probability in probabilities[:10]:
    print(round(probability, 3))

# 9. Evaluate Logistic Regression
print("===== Logistic Regression =====")

print(
    "Accuracy:",
    round(accuracy_score(y_test, logistic_predictions), 3),
)

print(
    "Precision:",
    round(precision_score(y_test, logistic_predictions), 3),
)

print(
    "Recall:",
    round(recall_score(y_test, logistic_predictions), 3),
)

print(
    "F1 Score:",
    round(f1_score(y_test, logistic_predictions), 3),
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, logistic_predictions))

print("\n===== Logistic Regression: Threshold 0.4 =====")

print(
    "Accuracy:",
    round(accuracy_score(y_test, custom_predictions), 3),
)

print(
    "Precision:",
    round(precision_score(y_test, custom_predictions), 3),
)

print(
    "Recall:",
    round(recall_score(y_test, custom_predictions), 3),
)

print(
    "F1 Score:",
    round(f1_score(y_test, custom_predictions), 3),
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, custom_predictions))


# 10. Create Random Forest pipeline
random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
        ),
    ]
)


# 11. Train Random Forest
random_forest_model.fit(X_train, y_train)


# 12. Make Random Forest predictions
rf_predictions = random_forest_model.predict(X_test)


# 13. Evaluate Random Forest
print("\n===== Random Forest =====")

print(
    "Accuracy:",
    round(accuracy_score(y_test, rf_predictions), 3),
)

print(
    "Precision:",
    round(precision_score(y_test, rf_predictions), 3),
)

print(
    "Recall:",
    round(recall_score(y_test, rf_predictions), 3),
)

print(
    "F1 Score:",
    round(f1_score(y_test, rf_predictions), 3),
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

joblib.dump(
    logistic_model,
    "models/churn_model.joblib"
)

print("\nModel saved successfully.")