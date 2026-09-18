import pytest

from fastapi.testclient import TestClient
from app import app, get_churn_threshold


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }

def test_ready(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}

def test_info(client):
    response = client.get("/info")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "customer-intelligence"
    assert data["service_version"] == "1.0.0"
    assert data["model_version"] == "1.0"
    assert 0 <= data["churn_threshold"] <= 1

def test_predict_valid_customer(client):
    customer = {
        "tenure_months": 6,
        "monthly_charges": 95.0,
        "support_calls": 7,
        "contract_type": "Monthly",
    }

    response = client.post(
        "/predict",
        json=customer,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "churn_probability" in data

    assert data["prediction"] in [0, 1]

    assert 0 <= data["churn_probability"] <= 1

def test_predict_invalid_customer(client):
    customer = {
        "tenure_months": -10,
        "monthly_charges": 95.0,
        "support_calls": 7,
        "contract_type": "Gold",
    }

    response = client.post(
        "/predict",
        json=customer,
    )

    assert response.status_code == 422

def test_sentiment_valid_text(client):
    payload = {
        "text": "The support team was excellent and very helpful"
    }

    response = client.post(
        "/sentiment",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["sentiment"] in ["positive", "negative"]
    assert 0 <= data["confidence"] <= 1


def test_sentiment_empty_text(client):
    payload = {
        "text": ""
    }

    response = client.post(
        "/sentiment",
        json=payload,
    )

    assert response.status_code == 422

def test_churn_threshold_valid(monkeypatch):
    monkeypatch.setenv("CHURN_THRESHOLD", "0.4")

    assert get_churn_threshold() == 0.4


def test_churn_threshold_not_number(monkeypatch):
    monkeypatch.setenv("CHURN_THRESHOLD", "hello")

    with pytest.raises(
        ValueError,
        match="CHURN_THRESHOLD must be a number",
    ):
        get_churn_threshold()


def test_churn_threshold_out_of_range(monkeypatch):
    monkeypatch.setenv("CHURN_THRESHOLD", "1.5")

    with pytest.raises(
        ValueError,
        match="CHURN_THRESHOLD must be between 0 and 1",
    ):
        get_churn_threshold()