"""Fast unit checks for strict validation and reference estimate arithmetic."""
from datetime import date
import json
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.house import HouseFeatures, FuturePredictionRequest
from app.services import prediction_service as prediction_service_module
from app.services.prediction_service import PredictionService
from app.main import city_rates, dashboard


def features(**overrides):
    data = {
        "BHK": 3,
        "Size_in_SqFt": 1500,
        "Year_Built": 2015,
        "Parking_Space": 1,
        "location": "Mumbai",
        "rate_per_sqft": 37930,
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize("field,value", [
    ("Size_in_SqFt", 500.5), ("Year_Built", 2015.5),
    ("Size_in_SqFt", ""), ("Year_Built", ""), ("rate_per_sqft", ""),
    ("Size_in_SqFt", float("nan")), ("Year_Built", float("inf")),
    ("rate_per_sqft", float("nan")),
])
def test_house_schema_rejects_non_integer_empty_and_non_finite(field, value):
    with pytest.raises(ValidationError):
        HouseFeatures.model_validate(features(**{field: value}))


def test_year_upper_bound_tracks_calendar_year():
    assert HouseFeatures.model_validate(features(Year_Built=date.today().year)).Year_Built == date.today().year
    with pytest.raises(ValidationError):
        HouseFeatures.model_validate(features(Year_Built=date.today().year + 1))


def test_reference_estimate_uses_edit_and_scales_locality_range():
    service = PredictionService()
    request = HouseFeatures.model_validate(features(rate_per_sqft=10_000))
    result = service._reference_values(request)
    expected = round(10_000 * service._adjustment_factor(request) * 1500 / 100_000, 2)
    assert result["reference_price_in_lakhs"] == expected
    assert result["reference_rate_used_per_sqft"] == 10_000
    assert result["locality_spread_low_in_lakhs"] < expected < result["locality_spread_high_in_lakhs"]
    assert any("far outside" in warning for warning in result["_rate_warnings"])


def test_unconfigured_city_has_no_reference_or_rate_warning():
    request = HouseFeatures.model_validate(features(location="Patna", rate_per_sqft=None))
    result = PredictionService()._reference_values(request)
    assert result["reference_status"] == "not_set"
    assert "_rate_warnings" not in result


def test_unconfigured_city_still_returns_demo_ml_comparison():
    request = HouseFeatures.model_validate(features(location="Patna", rate_per_sqft=None))
    result = PredictionService().predict(request)
    assert result.reference_status == "not_set"
    assert result.reference_price_in_lakhs is None
    assert result.demo_ml_price_in_lakhs > 0


def test_not_set_city_with_manual_rate_creates_main_estimate():
    request = HouseFeatures.model_validate(features(location="Patna", rate_per_sqft=7000))
    result = PredictionService().predict(request)
    assert result.reference_status == "manual"
    assert result.reference_price_in_lakhs > 0
    assert result.locality_spread_low_in_lakhs is None
    assert result.locality_spread_high_in_lakhs is None
    assert result.demo_ml_price_in_lakhs > 0
    assert not any("far outside" in warning for warning in result.warnings)


@pytest.mark.parametrize("city,typical,period", [
    ("New Delhi", 20333, "Jan-Mar 2026"),
    ("Chennai", 10357, "Jan-Mar 2026"),
    ("Kolkata", 8632, "Oct-Dec 2025"),
    ("Ahmedabad", 6725, "Apr-Jun 2026"),
])
def test_average_only_city_has_no_range_or_rate_warning(city, typical, period):
    request = HouseFeatures.model_validate(features(location=city, rate_per_sqft=typical))
    result = PredictionService().predict(request)
    assert result.reference_status == "average_only"
    assert result.reference_rate_used_per_sqft == typical
    assert result.reference_rate_low_per_sqft is None
    assert result.reference_rate_high_per_sqft is None
    assert result.locality_spread_low_in_lakhs is None
    assert result.locality_spread_high_in_lakhs is None
    assert result.reference_source_date == period
    assert "Range unavailable: source gave only an average." in result.reference_message
    assert not any("far outside" in warning for warning in result.warnings)


def test_configured_city_rate_edit_changes_main_value():
    service = PredictionService()
    low_rate = service.predict(HouseFeatures.model_validate(features(rate_per_sqft=20_000)))
    high_rate = service.predict(HouseFeatures.model_validate(features(rate_per_sqft=30_000)))
    assert high_rate.reference_price_in_lakhs > low_rate.reference_price_in_lakhs
    assert low_rate.demo_ml_price_in_lakhs == high_rate.demo_ml_price_in_lakhs


def test_dashboard_has_city_autofill_reset_and_dynamic_year_limit():
    html = dashboard().body.decode("utf-8")
    assert 'citySelect.addEventListener("change",syncRateForCity)' in html
    assert "rateInput.value=String(entry.typical)" in html
    assert "Reset to reference rate" in html
    assert f'max="{date.today().year}"' in html
    assert "rel=\"noopener noreferrer\"" in html


def test_city_rates_handler_returns_approved_config_entries():
    cities = city_rates()["cities"]
    assert cities["Pune"]["typical"] == 12853
    assert cities["Hyderabad"]["low"] == 8200
    assert cities["Kolkata"]["status"] == "average_only"
    assert "low" not in cities["Kolkata"]
    assert cities["Kolkata"]["source_date"] == "Oct-Dec 2025"
    assert cities["Patna"]["status"] == "not_set"


def test_city_rate_config_reloads_when_file_changes(monkeypatch):
    rate_file = Path(__file__).resolve().parents[1] / ".pytest_cache" / f"city-rates-{uuid4().hex}.json"
    rate_file.write_text(json.dumps({"cities": {"Pune": {"typical": 12853}}}), encoding="utf-8")
    monkeypatch.setattr(prediction_service_module, "RATE_CONFIG", rate_file)
    prediction_service_module._read_city_rates_config.cache_clear()
    try:
        assert prediction_service_module.city_rates_config()["cities"]["Pune"]["typical"] == 12853
        rate_file.write_text(json.dumps({"cities": {"Pune": {"typical": 12999, "source_date": "new"}}}), encoding="utf-8")
        updated = prediction_service_module.city_rates_config()["cities"]["Pune"]
        assert updated["typical"] == 12999
        assert updated["source_date"] == "new"
    finally:
        prediction_service_module._read_city_rates_config.cache_clear()
        rate_file.unlink(missing_ok=True)


def test_average_only_city_forecast_has_point_values_but_no_fake_range():
    request = FuturePredictionRequest.model_validate({
        **features(location="Kolkata", rate_per_sqft=8632),
        "years": 5,
        "inflation_rate": 0.06,
        "interest_rate": 0.085,
        "gdp_growth_rate": 0.07,
        "migration_rate": 0.02,
    })
    result = PredictionService().predict_future(request)
    assert result.reference_status == "average_only"
    assert result.current_reference_price_in_lakhs > 0
    assert result.forecast
    assert result.forecast[0].ending_price_low_in_lakhs is None
    assert result.forecast[0].ending_price_high_in_lakhs is None


@pytest.mark.parametrize("field,value", [
    ("inflation_rate", float("nan")), ("interest_rate", float("inf")),
    ("gdp_growth_rate", -0.051), ("migration_rate", 0.101),
])
def test_forecast_schema_rejects_invalid_assumptions(field, value):
    data = {**features(), "years": 5, field: value}
    with pytest.raises(ValidationError):
        FuturePredictionRequest.model_validate(data)
