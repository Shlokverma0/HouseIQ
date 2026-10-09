"""
Pydantic schemas for HouseIQ inputs and outputs.
"""
import json
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, FiniteFloat, StrictInt, field_validator, model_validator
from enum import Enum


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_METADATA = PROJECT_ROOT / "models" / "house_metadata.json"


@lru_cache(maxsize=1)
def training_limits():
    try:
        return json.loads(MODEL_METADATA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


class LocationEnum(str, Enum):
    ahmedabad = "Ahmedabad"
    amritsar = "Amritsar"
    bangalore = "Bangalore"
    bhopal = "Bhopal"
    bhubaneswar = "Bhubaneswar"
    bilaspur = "Bilaspur"
    chennai = "Chennai"
    coimbatore = "Coimbatore"
    cuttack = "Cuttack"
    dehradun = "Dehradun"
    durgapur = "Durgapur"
    dwarka = "Dwarka"
    faridabad = "Faridabad"
    gaya = "Gaya"
    gurgaon = "Gurgaon"
    guwahati = "Guwahati"
    haridwar = "Haridwar"
    hyderabad = "Hyderabad"
    indore = "Indore"
    jaipur = "Jaipur"
    jamshedpur = "Jamshedpur"
    jodhpur = "Jodhpur"
    kochi = "Kochi"
    kolkata = "Kolkata"
    lucknow = "Lucknow"
    ludhiana = "Ludhiana"
    mangalore = "Mangalore"
    mumbai = "Mumbai"
    mysore = "Mysore"
    nagpur = "Nagpur"
    new_delhi = "New Delhi"
    noida = "Noida"
    patna = "Patna"
    pune = "Pune"
    raipur = "Raipur"
    ranchi = "Ranchi"
    silchar = "Silchar"
    surat = "Surat"
    trivandrum = "Trivandrum"
    vijayawada = "Vijayawada"
    visakhapatnam = "Vishakhapatnam"
    warangal = "Warangal"


class HouseFeatures(BaseModel):
    BHK: int = Field(..., ge=1, le=5)
    Size_in_SqFt: StrictInt = Field(..., ge=500, le=5000)
    Year_Built: StrictInt = Field(..., ge=1900)
    Parking_Space: int = Field(..., ge=0, le=1)
    location: LocationEnum = Field(..., description="City name")
    rate_per_sqft: Optional[FiniteFloat] = Field(
        None, ge=1_000, le=150_000,
        description="Editable city reference rate in INR per square foot. Ignored when that city has no configured reference rate.",
    )

    @field_validator("location", mode="before")
    @classmethod
    def normalize_location(cls, value):
        if isinstance(value, str):
            synonyms = {
                "Delhi": "New Delhi",
                "Bombay": "Mumbai",
                "Bengaluru": "Bangalore",
                "Madras": "Chennai",
                "Calcutta": "Kolkata",
                "Gurugram": "Gurgaon",
            }
            cleaned = value.strip()
            return synonyms.get(cleaned, cleaned)
        return value

    @field_validator("Year_Built")
    @classmethod
    def year_must_not_be_in_the_future(cls, value):
        if value > date.today().year:
            raise ValueError(f"Year built cannot be later than the current year ({date.today().year}).")
        return value

    @model_validator(mode="after")
    def validate_training_size_for_bhk(self):
        limits = training_limits().get("bhk_limits", {})
        bhk_limits = limits.get(str(self.BHK))
        if not bhk_limits:
            raise ValueError(f"No training records are available for {self.BHK} BHK.")
        min_size = bhk_limits["min_size_sqft"]
        max_size = bhk_limits["max_size_sqft"]
        if not min_size <= self.Size_in_SqFt <= max_size:
            raise ValueError(
                f"For {self.BHK} BHK, supported size is {min_size}–{max_size} sq ft."
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "BHK": 3,
                "Size_in_SqFt": 1500,
                "Year_Built": 2015,
                "Parking_Space": 1,
                "location": "Mumbai",
                "rate_per_sqft": 37930,
            }
        }
    }


class PredictionResponse(BaseModel):
    reference_status: str
    reference_message: Optional[str] = None
    reference_price_per_sqft: Optional[float] = None
    reference_rate_used_per_sqft: Optional[float] = None
    reference_price_in_lakhs: Optional[float] = None
    locality_spread_low_in_lakhs: Optional[float] = None
    locality_spread_high_in_lakhs: Optional[float] = None
    reference_rate_low_per_sqft: Optional[float] = None
    reference_rate_typical_per_sqft: Optional[float] = None
    reference_rate_high_per_sqft: Optional[float] = None
    reference_source: Optional[str] = None
    reference_source_url: Optional[str] = None
    reference_source_date: Optional[str] = None
    reference_accessed_date: Optional[str] = None
    demo_ml_price_per_sqft: float
    demo_ml_price_in_lakhs: float
    warnings: List[str] = Field(default_factory=list)
    currency: str = "INR"
    disclaimer: str = (
        "The main estimate uses sourced reference rates where set. The separate ML comparison uses synthetic Kaggle data and is not a market valuation."
    )


class FuturePredictionRequest(HouseFeatures):
    """Forecasts an assumption-based scenario; it is not city-specific."""
    years: StrictInt = Field(..., gt=0, le=50)
    inflation_rate: FiniteFloat = Field(0.06, ge=0, le=0.15)
    interest_rate: FiniteFloat = Field(0.085, ge=0.01, le=0.20)
    gdp_growth_rate: FiniteFloat = Field(0.07, ge=-0.05, le=0.15)
    migration_rate: FiniteFloat = Field(0.02, ge=0, le=0.10)


class ForecastPeriod(BaseModel):
    period: str
    start_year: int
    end_year: int
    previous_price_in_lakhs: float
    ending_price_in_lakhs: float
    previous_price_low_in_lakhs: Optional[float] = None
    previous_price_high_in_lakhs: Optional[float] = None
    ending_price_low_in_lakhs: Optional[float] = None
    ending_price_high_in_lakhs: Optional[float] = None


class FuturePredictionResponse(BaseModel):
    reference_status: str
    reference_message: Optional[str] = None
    current_reference_price_per_sqft: Optional[float] = None
    reference_rate_used_per_sqft: Optional[float] = None
    current_reference_price_in_lakhs: Optional[float] = None
    current_locality_spread_low_in_lakhs: Optional[float] = None
    current_locality_spread_high_in_lakhs: Optional[float] = None
    reference_rate_low_per_sqft: Optional[float] = None
    reference_rate_typical_per_sqft: Optional[float] = None
    reference_rate_high_per_sqft: Optional[float] = None
    reference_source: Optional[str] = None
    reference_source_url: Optional[str] = None
    reference_source_date: Optional[str] = None
    reference_accessed_date: Optional[str] = None
    demo_ml_price_per_sqft: float
    demo_ml_price_in_lakhs: float
    effective_growth_rate: Optional[float] = None
    forecast: List[ForecastPeriod]
    forecast_is_city_specific: bool = False
    warnings: List[str] = Field(default_factory=list)
    currency: str = "INR"
    disclaimer: str = (
        "The forecast compounds sourced reference estimates and locality spreads using user assumptions; the separate ML comparison uses synthetic Kaggle data."
    )
