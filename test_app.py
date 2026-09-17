from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }

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