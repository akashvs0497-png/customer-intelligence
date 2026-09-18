import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# 1. Load sentiment data
df = pd.read_csv("data/sentiment.csv")

X = df["text"]
y = df["sentiment"]


# 2. Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)


# 3. Build text classification pipeline
sentiment_model = Pipeline(
    steps=[
        (
            "vectorizer",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
            ),
        ),
        (
            "classifier",
            LogisticRegression(max_iter=1000),
        ),
    ]
)


# 4. Train
sentiment_model.fit(X_train, y_train)


# 5. Predict
predictions = sentiment_model.predict(X_test)

results = pd.DataFrame(
    {
        "text": X_test.values,
        "actual": y_test.values,
        "predicted": predictions,
    }
)

print("\n===== Sentiment Predictions =====")
print(results.to_string(index=False))


# 6. Evaluate
print("===== Sentiment Model =====")

print(
    "Accuracy:",
    round(accuracy_score(y_test, predictions), 3),
)

print(
    "Precision:",
    round(
        precision_score(
            y_test,
            predictions,
            pos_label="positive",
        ),
        3,
    ),
)

print(
    "Recall:",
    round(
        recall_score(
            y_test,
            predictions,
            pos_label="positive",
        ),
        3,
    ),
)

print(
    "F1 Score:",
    round(
        f1_score(
            y_test,
            predictions,
            pos_label="positive",
        ),
        3,
    ),
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


# 7. Save complete pipeline
joblib.dump(
    sentiment_model,
    "models/sentiment_model.joblib",
)

print("\nSentiment model saved successfully.")