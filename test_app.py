import pytest

from fastapi.testclient import TestClient
from app import app, get_churn_threshold


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }

def test_ready():
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}

def test_info():
    response = client.get("/info")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "customer-intelligence"
    assert data["service_version"] == "1.0.0"
    assert data["model_version"] == "1.0"
    assert 0 <= data["churn_threshold"] <= 1

def test_predict_valid_customer():
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

def test_predict_invalid_customer():
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