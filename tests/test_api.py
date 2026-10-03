"""
test_api.py
-----------
Tests for the HouseIQ API.
"""

from fastapi.testclient import TestClient
import pytest
from app.main import app



@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "HouseIQ API is running."
    assert app.title == "HouseIQ"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()
    assert response.json()["model_loaded"] is True


def test_predict(client):
    payload = {
        "BHK": 3,
        "Size_in_SqFt": 1500,
        "Price_per_SqFt": 8000,
        "Year_Built": 2015,
        "Parking_Space": 1,
        "location": "Mumbai"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_price_in_lakhs" in response.json()
    assert "input_summary" not in response.json()


def test_future_forecast_is_summarized_in_decades(client):
    payload = {
        "BHK": 3,
        "Size_in_SqFt": 1200,
        "Price_per_SqFt": 5000,
        "Year_Built": 2020,
        "Parking_Space": 1,
        "location": "Mumbai",
        "years": 50,
        "inflation_rate": 0.06,
        "interest_rate": 0.085,
        "gdp_growth_rate": 0.07,
        "migration_rate": 0.02,
    }

    response = client.post("/predict/future", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "input_summary" not in body
    assert len(body["forecast"]) == 5
    assert [period["period"] for period in body["forecast"]] == [
        "Years 1–10",
        "Years 11–20",
        "Years 21–30",
        "Years 31–40",
        "Years 41–50",
    ]
    assert body["forecast"][0]["previous_price_in_lakhs"] == body["current_price_in_lakhs"]
    assert body["forecast"][-1]["ending_price_in_lakhs"] > body["forecast"][0]["ending_price_in_lakhs"]
    assert body["current_price_display"].startswith("₹")
    assert "crore" in body["current_price_display"]
