"""Reference-rate estimate plus an explicitly secondary synthetic ML comparison."""
import json
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.repositories.model_repository import model_repository
from app.schemas.house import (
    ForecastPeriod,
    FuturePredictionRequest,
    FuturePredictionResponse,
    HouseFeatures,
    PredictionResponse,
)
from app.utils.logger import logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RATE_CONFIG = PROJECT_ROOT / "data" / "city_reference_rates.json"


@lru_cache(maxsize=4)
def _read_city_rates_config(modified_ns: int, file_size: int) -> dict:
    # Cache by file metadata so a config edit takes effect without restarting Uvicorn.
    return json.loads(RATE_CONFIG.read_text(encoding="utf-8"))


def city_rates_config() -> dict:
    stat = RATE_CONFIG.stat()
    return _read_city_rates_config(stat.st_mtime_ns, stat.st_size)


class PredictionService:
    @staticmethod
    def _prepare_features(features: HouseFeatures) -> pd.DataFrame:
        data = {
            "BHK": float(features.BHK),
            "Size_in_SqFt": float(features.Size_in_SqFt),
            "Year_Built": float(features.Year_Built),
            "Parking_Space": float(features.Parking_Space),
            "City": features.location.value,
        }
        frame = pd.get_dummies(
            pd.DataFrame([data]), columns=["City"], prefix="loc", dtype=float
        )
        return frame.reindex(columns=model_repository.columns, fill_value=0.0)

    def _ensure_model(self):
        if not model_repository.loaded:
            model_repository.load()

    def _predict_demo_ml_psf(self, features: HouseFeatures) -> float:
        self._ensure_model()
        frame = self._prepare_features(features)
        return max(0.0, float(model_repository.model.predict(frame)[0]))

    @staticmethod
    def _year_adjustment(year_built: int) -> float:
        # Transparent illustrative rule: ±0.25% per year from 2020, capped at ±10%.
        return min(0.10, max(-0.10, (year_built - 2020) * 0.0025))

    @classmethod
    def _adjustment_factor(cls, features: HouseFeatures) -> float:
        parking_factor = 1.02 if features.Parking_Space else 1.0
        return (1.0 + cls._year_adjustment(features.Year_Built)) * parking_factor

    @staticmethod
    def _year_warnings(year_built: int) -> list[str]:
        minimum = int(model_repository.metadata.get("year_min", 1990))
        maximum = int(model_repository.metadata.get("year_max", 2023))
        if year_built < minimum or year_built > maximum:
            return [
                f"Year built is outside the demo model training range ({minimum}–{maximum}); the demo ML comparison is a rough extrapolation."
            ]
        return []

    @staticmethod
    def _city_rate(city: str) -> dict:
        return city_rates_config().get("cities", {}).get(city, {"status": "not_set"})

    def _reference_values(self, features: HouseFeatures) -> dict:
        rate = self._city_rate(features.location.value)
        configured_rate = rate.get("status") in {"set", "average_only"} and rate.get("typical") is not None
        has_manual_rate = features.rate_per_sqft is not None
        if not configured_rate and not has_manual_rate:
            return {
                "reference_status": "not_set",
                "reference_message": (
                    f"A verified MagicBricks reference rate is not set for {features.location.value} yet. "
                    "Enter a rate per sq ft to create a manual estimate, or use the demo ML comparison shown separately."
                ),
            }

        used_rate = float(features.rate_per_sqft if has_manual_rate else rate["typical"])
        factor = self._adjustment_factor(features)
        area_factor = features.Size_in_SqFt / 100_000
        typical_rate = float(rate["typical"]) if configured_rate else used_rate
        scale = used_rate / typical_rate
        low_rate = rate.get("low") if configured_rate else None
        high_rate = rate.get("high") if configured_rate else None
        has_spread = low_rate is not None and high_rate is not None
        warnings = []
        if has_spread and (used_rate < float(low_rate) * 0.5 or used_rate > float(high_rate) * 1.5):
            warnings.append(
                f"Entered rate ₹{used_rate:,.0f}/sq ft is far outside the configured city reference spread."
            )
        if configured_rate and not has_spread:
            reference_message = "Range unavailable: source gave only an average."
            status = "average_only"
        elif configured_rate:
            reference_message = None
            status = "set"
        else:
            reference_message = f"Using your entered rate; no verified city reference is configured for {features.location.value}."
            status = "manual"
        return {
            "reference_status": status,
            "reference_message": reference_message,
            "reference_rate_used_per_sqft": used_rate,
            "reference_price_per_sqft": round(used_rate * factor, 2),
            "reference_price_in_lakhs": round(used_rate * factor * area_factor, 2),
            "locality_spread_low_in_lakhs": round(float(low_rate) * scale * factor * area_factor, 2) if has_spread else None,
            "locality_spread_high_in_lakhs": round(float(high_rate) * scale * factor * area_factor, 2) if has_spread else None,
            "reference_rate_low_per_sqft": float(low_rate) if has_spread else None,
            "reference_rate_typical_per_sqft": typical_rate if configured_rate else None,
            "reference_rate_high_per_sqft": float(high_rate) if has_spread else None,
            "reference_source": rate.get("source") if configured_rate else None,
            "reference_source_url": rate.get("source_url") if configured_rate else None,
            "reference_source_date": rate.get("source_date") if configured_rate else None,
            "reference_accessed_date": rate.get("accessed_date") if configured_rate else None,
            "_rate_warnings": warnings,
        }

    def predict(self, features: HouseFeatures) -> PredictionResponse:
        reference = self._reference_values(features)
        rate_warnings = reference.pop("_rate_warnings", [])
        ml_psf = self._predict_demo_ml_psf(features)
        ml_total = ml_psf * features.Size_in_SqFt / 100_000
        logger.info(
            "Estimate request: city=%s, BHK=%s, size=%s, reference_status=%s, demo_ml_psf=%.2f",
            features.location.value,
            features.BHK,
            features.Size_in_SqFt,
            reference["reference_status"],
            ml_psf,
        )
        return PredictionResponse(
            **reference,
            demo_ml_price_per_sqft=round(ml_psf, 2),
            demo_ml_price_in_lakhs=round(ml_total, 2),
            warnings=self._year_warnings(features.Year_Built) + rate_warnings,
        )

    @staticmethod
    def _clamp_growth(rate: float) -> float:
        return min(0.12, max(0.02, rate))

    def _annual_growth_rates(self, request: FuturePredictionRequest) -> list[float]:
        inflation_target = self._clamp_growth(request.inflation_rate)
        short_rate = self._clamp_growth(
            request.inflation_rate
            + 0.25 * request.gdp_growth_rate
            + 0.10 * request.migration_rate
            - 0.20 * request.interest_rate
        )
        return [
            self._clamp_growth(
                inflation_target
                + (short_rate - inflation_target) * math.exp(-(year - 1) / 15.0)
            )
            for year in range(1, request.years + 1)
        ]

    def predict_future(self, request: FuturePredictionRequest) -> FuturePredictionResponse:
        current = self.predict(request)
        reference_set = current.reference_price_in_lakhs is not None
        if not reference_set:
            return FuturePredictionResponse(
                reference_status=current.reference_status,
                reference_message=current.reference_message,
                demo_ml_price_per_sqft=current.demo_ml_price_per_sqft,
                demo_ml_price_in_lakhs=current.demo_ml_price_in_lakhs,
                reference_rate_used_per_sqft=None,
                forecast=[],
                forecast_is_city_specific=False,
                warnings=current.warnings,
            )

        current_price = current.reference_price_in_lakhs
        current_low = current.locality_spread_low_in_lakhs
        current_high = current.locality_spread_high_in_lakhs
        rates = self._annual_growth_rates(request)
        point_prices = {0: current_price}
        range_available = current_low is not None and current_high is not None
        low_prices = {0: current_low} if range_available else {}
        high_prices = {0: current_high} if range_available else {}
        for year, rate in enumerate(rates, start=1):
            factor = 1 + rate
            point_prices[year] = point_prices[year - 1] * factor
            if range_available:
                low_prices[year] = low_prices[year - 1] * factor
                high_prices[year] = high_prices[year - 1] * factor

        forecast = []
        for start_year in range(1, request.years + 1, 10):
            end_year = min(start_year + 9, request.years)
            previous_year = start_year - 1
            forecast.append(
                ForecastPeriod(
                    period=f"Years {start_year}–{end_year}",
                    start_year=start_year,
                    end_year=end_year,
                    previous_price_in_lakhs=round(point_prices[previous_year], 2),
                    ending_price_in_lakhs=round(point_prices[end_year], 2),
                    previous_price_low_in_lakhs=round(low_prices[previous_year], 2) if range_available else None,
                    previous_price_high_in_lakhs=round(high_prices[previous_year], 2) if range_available else None,
                    ending_price_low_in_lakhs=round(low_prices[end_year], 2) if range_available else None,
                    ending_price_high_in_lakhs=round(high_prices[end_year], 2) if range_available else None,
                )
            )

        average_growth = (point_prices[request.years] / current_price) ** (1 / request.years) - 1
        logger.info(
            "Reference forecast: city=%s, years=%s, average_growth=%.4f",
            request.location.value,
            request.years,
            average_growth,
        )
        return FuturePredictionResponse(
            reference_status=current.reference_status,
            reference_message=current.reference_message,
            current_reference_price_per_sqft=current.reference_price_per_sqft,
            reference_rate_used_per_sqft=current.reference_rate_used_per_sqft,
            current_reference_price_in_lakhs=current.reference_price_in_lakhs,
            current_locality_spread_low_in_lakhs=current.locality_spread_low_in_lakhs,
            current_locality_spread_high_in_lakhs=current.locality_spread_high_in_lakhs,
            reference_rate_low_per_sqft=current.reference_rate_low_per_sqft,
            reference_rate_typical_per_sqft=current.reference_rate_typical_per_sqft,
            reference_rate_high_per_sqft=current.reference_rate_high_per_sqft,
            reference_source=current.reference_source,
            reference_source_url=current.reference_source_url,
            reference_source_date=current.reference_source_date,
            reference_accessed_date=current.reference_accessed_date,
            demo_ml_price_per_sqft=current.demo_ml_price_per_sqft,
            demo_ml_price_in_lakhs=current.demo_ml_price_in_lakhs,
            effective_growth_rate=round(average_growth, 4),
            forecast=forecast,
            forecast_is_city_specific=False,
            warnings=current.warnings,
        )


prediction_service = PredictionService()
