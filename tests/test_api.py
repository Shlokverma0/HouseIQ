"""API and business-rule tests for HouseIQ."""
from copy import deepcopy
from datetime import date
import asyncio
import json
import re
import shutil
import subprocess

import httpx
import pytest

from app.main import app
from app.schemas.house import FuturePredictionRequest
from app.services.prediction_service import PredictionService
from app.repositories.model_repository import model_repository


class SyncASGIClient:
    """Small sync wrapper around httpx ASGITransport (Starlette TestClient is incompatible with installed httpx)."""
    def _request(self, method, path, **kwargs):
        async def send():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                return await client.request(method, path, **kwargs)
        return asyncio.run(send())

    def get(self, path, **kwargs):
        return self._request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self._request("POST", path, **kwargs)


@pytest.fixture(scope="module")
def client():
    model_repository.load()
    yield SyncASGIClient()


@pytest.fixture
def base_payload():
    return {
        "BHK": 3,
        "Size_in_SqFt": 1500,
        "Year_Built": 2015,
        "Parking_Space": 1,
        "location": "Mumbai",
        "rate_per_sqft": 37930,
    }


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "HouseIQ API is running."
    assert app.title == "HouseIQ"


def test_dashboard_explains_reference_and_demo_ml_estimates(client):
    response = client.get("/app")
    assert response.status_code == 200
    assert "Rate per sq ft (editable)" in response.text
    assert 'step="any"' in response.text
    assert 'rel="noopener noreferrer"' in response.text
    assert "Reset to reference rate" in response.text
    assert "(enter rate)" in response.text
    assert "Range unavailable: source gave only an average." in response.text
    assert "Enter a rate manually to create the main estimate" in response.text
    assert 'citySelect.addEventListener("change",syncRateForCity)' in response.text
    assert 'rateInput.value=String(entry.typical)' in response.text
    assert 'max="__CURRENT_YEAR__"' not in response.text
    assert f'max="{date.today().year}"' in response.text
    assert "Demo ML model estimate (synthetic data, comparison only)" in response.text
    assert "BHK" in response.text and "does not change the reference-rate estimate" in response.text
    assert '$("resetRate").hidden=true' in response.text
    assert 'Estimate (your rate)' in response.text
    assert 'Estimate uses the rate you entered' in response.text
    assert 'Synthetic data, not comparable to real prices' in response.text
    assert 'Adjusted rate ${lakhsPerSqFt(pointRate)}' in response.text
    assert 'requestRevision!==inputRevision' in response.text
    assert 'el.addEventListener("change"' in response.text


def test_health_and_metadata(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True

    metadata = client.get("/metadata")
    assert metadata.status_code == 200
    assert metadata.json()["dataset_is_synthetic"] is True
    assert metadata.json()["bhk_limits"]["3"]["min_size_sqft"] == 500


def test_city_rate_configuration_is_served_locally(client):
    response = client.get("/city-rates")
    assert response.status_code == 200
    cities = response.json()["cities"]
    assert cities["Mumbai"]["typical"] == 37930
    assert cities["Mumbai"]["source_url"].startswith("https://www.magicbricks.com/")
    assert cities["Patna"]["status"] == "not_set"
    assert cities["Pune"]["typical"] == 12853
    assert cities["Pune"]["low"] == 9200 and cities["Pune"]["high"] == 13200
    assert cities["Hyderabad"]["typical"] == 9120
    for city in ("New Delhi", "Chennai", "Kolkata", "Ahmedabad"):
        assert cities[city]["status"] == "average_only"
        assert "low" not in cities[city] and "high" not in cities[city]
    assert cities["Kolkata"]["source_date"] == "Oct-Dec 2025"


def test_prediction_uses_city_reference_rate_and_keeps_ml_comparison_separate(client, base_payload):
    response = client.post("/predict", json=base_payload)
    assert response.status_code == 200
    body = response.json()
    expected_total_lakhs = round(body["reference_price_per_sqft"] * 1500 / 100000, 2)
    assert body["reference_status"] == "set"
    assert body["reference_price_in_lakhs"] == expected_total_lakhs
    assert body["locality_spread_low_in_lakhs"] <= body["reference_price_in_lakhs"]
    assert body["reference_price_in_lakhs"] <= body["locality_spread_high_in_lakhs"]
    assert body["demo_ml_price_in_lakhs"] > 0
    assert body["demo_ml_price_per_sqft"] > 0
    assert body["reference_source_url"].startswith("https://www.magicbricks.com/")
    assert body["reference_accessed_date"] == "2026-10-09"
    assert body["reference_rate_low_per_sqft"] <= body["reference_rate_typical_per_sqft"]
    assert body["reference_rate_typical_per_sqft"] <= body["reference_rate_high_per_sqft"]
    assert body["reference_rate_used_per_sqft"] == 37930


def test_edited_rate_drives_main_estimate_and_warning_is_non_blocking(client, base_payload):
    payload = {**base_payload, "rate_per_sqft": 1000}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["reference_rate_used_per_sqft"] == 1000
    assert body["reference_price_in_lakhs"] < 20
    assert any("far outside" in warning for warning in body["warnings"])


def test_reference_rate_threshold_has_no_warning_inside_configured_limits(client, base_payload):
    for rate in (21700 * 0.5, 40600 * 1.5):
        response = client.post("/predict", json={**base_payload, "rate_per_sqft": rate})
        assert response.status_code == 200
        assert not any("far outside" in warning for warning in response.json()["warnings"])


@pytest.mark.parametrize("field,value", [
    ("Size_in_SqFt", 499), ("Size_in_SqFt", 5001),
    ("rate_per_sqft", 999), ("rate_per_sqft", 150001),
    ("Year_Built", 1899), ("inflation_rate", -0.001), ("inflation_rate", 0.151),
    ("interest_rate", 0.009), ("interest_rate", 0.201),
    ("gdp_growth_rate", -0.051), ("gdp_growth_rate", 0.151),
    ("migration_rate", -0.001), ("migration_rate", 0.101),
])
def test_api_rejects_out_of_range_values(client, base_payload, field, value):
    payload = {**base_payload, field: value}
    if field in {"inflation_rate", "interest_rate", "gdp_growth_rate", "migration_rate"}:
        payload["years"] = 5
    assert client.post("/predict/future" if "years" in payload else "/predict", json=payload).status_code == 422


@pytest.mark.parametrize("field,value", [("Size_in_SqFt", 1500.5), ("Year_Built", 2020.5)])
def test_api_requires_integer_size_and_year(client, base_payload, field, value):
    assert client.post("/predict", json={**base_payload, field: value}).status_code == 422


@pytest.mark.parametrize("field", ["Size_in_SqFt", "Year_Built", "rate_per_sqft"])
def test_api_rejects_empty_values(client, base_payload, field):
    assert client.post("/predict", json={**base_payload, field: ""}).status_code == 422


@pytest.mark.parametrize("field,value", [
    ("Size_in_SqFt", float("nan")), ("Year_Built", float("inf")),
    ("rate_per_sqft", float("nan")),
])
def test_api_rejects_raw_non_finite_values(client, base_payload, field, value):
    raw_json = json.dumps({**base_payload, field: value}, allow_nan=True)
    response = client.post(
        "/predict",
        content=raw_json,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]


def test_frontend_pune_city_change_autofills_rate_source_and_date(client):
    rates = client.get("/city-rates").json()["cities"]["Pune"]
    html = client.get("/app").text
    assert rates["typical"] == 12853
    assert rates["source"] == "MagicBricks, Flats for Sale in Pune; Jan-Mar 2026 average-rate table and published flat-rate range"
    assert rates["source_date"] == "Jan-Mar 2026"
    assert 'citySelect.addEventListener("change",syncRateForCity)' in html
    assert 'rateInput.value=String(entry.typical)' in html
    assert 'Reference ₹${Number(entry.typical).toLocaleString("en-IN")}/sq ft · ${entry.source} · ${entry.source_date}' in html
    assert 'cityRates=data.cities||{};ratesReady=true' in html
    assert 'entry.low!=null&&entry.high!=null?' in html
    assert 'source spread ₹${payload.reference_rate_low_per_sqft}–₹${payload.reference_rate_high_per_sqft}' in html
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the frontend autofill function")
    function = re.search(r"function syncRateForCity\(\)\{[\s\S]*?\}(?=\s*function showError)", html)
    assert function, "Could not find the city rate autofill function"
    harness = r'''
const vm = require("node:vm");
const config = JSON.parse(process.argv[2]);
const elements = {
  liveRateLink: {}, resetRate: {}, rateHint: {},
  rate_per_sqft: {value: "", disabled: false, placeholder: ""},
  location: {value: "Pune"}
};
const context = {
  $: id => elements[id], citySelect: elements.location,
  cityRates: {Pune: config}, rateInput: elements.rate_per_sqft,
  hasCityRate: e => Boolean(e && ["set", "average_only"].includes(e.status) && Number.isFinite(Number(e.typical))),
  validateForm: () => {}, clearResult: () => {}, rateEdited: true, inputRevision: 0
};
vm.runInNewContext(process.argv[1] + "\nsyncRateForCity()", context);
process.stdout.write(JSON.stringify({rate: elements.rate_per_sqft.value, hint: elements.rateHint.textContent}));
'''
    completed = subprocess.run(
        [node, "-e", harness, function.group(0), json.dumps(rates)],
        check=True, capture_output=True, text=True,
    )
    autofill = json.loads(completed.stdout)
    assert autofill["rate"] == "12853"
    assert rates["source"] in autofill["hint"]
    assert rates["source_date"] in autofill["hint"]


def test_unsupported_city_manual_rate_creates_estimate_and_keeps_demo_ml(client, base_payload):
    response = client.post("/predict", json={**base_payload, "location": "Patna", "rate_per_sqft": 37930})
    assert response.status_code == 200
    body = response.json()
    assert body["reference_status"] == "manual"
    assert body["reference_price_in_lakhs"] > 0
    assert body["locality_spread_low_in_lakhs"] is None
    assert body["locality_spread_high_in_lakhs"] is None
    assert body["demo_ml_price_in_lakhs"] > 0
    assert not any("far outside" in warning for warning in body["warnings"])


def test_patna_current_and_future_routes_return_friendly_not_set_status(client, base_payload):
    patna = {**base_payload, "location": "Patna"}
    patna.pop("rate_per_sqft")
    current = client.post("/predict", json=patna)
    assert current.status_code == 200
    current_body = current.json()
    assert current_body["reference_status"] == "not_set"
    assert current_body["reference_price_in_lakhs"] is None
    assert "not set for Patna" in current_body["reference_message"]
    assert current_body["demo_ml_price_in_lakhs"] > 0

    future = client.post("/predict/future", json={
        **patna, "years": 5, "inflation_rate": 0.06,
        "interest_rate": 0.085, "gdp_growth_rate": 0.07, "migration_rate": 0.02,
    })
    assert future.status_code == 200
    future_body = future.json()
    assert future_body["reference_status"] == "not_set"
    assert future_body["forecast"] == []
    assert "not set for Patna" in future_body["reference_message"]
    assert future_body["demo_ml_price_in_lakhs"] > 0


def test_training_range_validation_rejects_unsupported_inputs(client, base_payload):
    invalid_bhk = deepcopy(base_payload)
    invalid_bhk["BHK"] = 6
    assert client.post("/predict", json=invalid_bhk).status_code == 422

    invalid_size = deepcopy(base_payload)
    invalid_size["Size_in_SqFt"] = 100
    assert client.post("/predict", json=invalid_size).status_code == 422


def test_old_year_is_allowed_with_rough_extrapolation_warning(client, base_payload):
    payload = deepcopy(base_payload)
    payload["Year_Built"] = 1908
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert any("rough extrapolation" in warning for warning in response.json()["warnings"])

    future_year = deepcopy(base_payload)
    future_year["Year_Built"] = date.today().year + 1
    response = client.post("/predict", json=future_year)
    assert response.status_code == 422


def test_future_forecast_uses_reference_spread_and_assumptions(client, base_payload):
    payload = {
        **base_payload,
        "years": 50,
        "inflation_rate": 0.06,
        "interest_rate": 0.085,
        "gdp_growth_rate": 0.07,
        "migration_rate": 0.02,
    }
    response = client.post("/predict/future", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["forecast"]) == 5
    assert [period["period"] for period in body["forecast"]] == [
        "Years 1–10",
        "Years 11–20",
        "Years 21–30",
        "Years 31–40",
        "Years 41–50",
    ]
    assert body["forecast_is_city_specific"] is False
    assert 0.02 <= body["effective_growth_rate"] <= 0.12
    assert body["reference_status"] == "set"
    assert body["current_locality_spread_low_in_lakhs"] <= body["current_reference_price_in_lakhs"]
    assert body["current_reference_price_in_lakhs"] <= body["current_locality_spread_high_in_lakhs"]
    assert body["demo_ml_price_in_lakhs"] > 0
    assert body["forecast"][0]["ending_price_high_in_lakhs"] > body["forecast"][0]["ending_price_low_in_lakhs"]
    assert body["forecast"][-1]["ending_price_high_in_lakhs"] > body["forecast"][0]["ending_price_high_in_lakhs"]


def test_long_run_growth_moves_toward_inflation_and_rate_is_capped():
    service = PredictionService()
    short_request = FuturePredictionRequest(
        BHK=3, Size_in_SqFt=1500, Year_Built=2015, Parking_Space=1,
        location="Mumbai", years=5, inflation_rate=0.03, interest_rate=0.01,
        gdp_growth_rate=0.15, migration_rate=0.10,
    )
    long_request = short_request.model_copy(update={"years": 50})
    short_rates = service._annual_growth_rates(short_request)
    long_rates = service._annual_growth_rates(long_request)
    assert all(0.02 <= rate <= 0.12 for rate in long_rates)
    assert abs(long_rates[-1] - 0.03) < abs(short_rates[-1] - 0.03)


def test_current_value_does_not_depend_on_forecast_horizon(client, base_payload):
    five_years = {
        **base_payload, "years": 5, "inflation_rate": 0.06,
        "interest_rate": 0.085, "gdp_growth_rate": 0.07, "migration_rate": 0.02,
    }
    fifty_years = {**five_years, "years": 50}
    five = client.post("/predict/future", json=five_years).json()
    fifty = client.post("/predict/future", json=fifty_years).json()
    assert five["current_reference_price_in_lakhs"] == fifty["current_reference_price_in_lakhs"]
    assert five["effective_growth_rate"] != fifty["effective_growth_rate"]
