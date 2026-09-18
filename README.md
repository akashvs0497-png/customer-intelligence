# Customer Intelligence Service

Small production-style ML service for customer churn prediction and sentiment analysis.

## Features

- Customer churn prediction
- Customer sentiment classification
- Logistic Regression, Random Forest, and XGBoost experiments
- Hyperparameter tuning with GridSearchCV
- FastAPI inference API
- Pydantic input validation
- Configurable churn threshold
- Prometheus metrics
- Data drift detection example
- Docker containerization
- GitHub Actions CI/CD
- GHCR container registry
- Cloud deployment on Render

## Architecture

```text
Customer Request
      |
      v
   FastAPI
   /     \
  /       \
Churn    Sentiment
Model     Model
  \       /
   \     /
   Metrics
      |
      v
  /metrics
      |
      v
 Prometheus
      |
      v
   Grafana