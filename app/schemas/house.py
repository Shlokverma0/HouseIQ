"""
house.py
--------
Layer 2: Pydantic schemas for input validation and response formatting.
"""
from enum import Enum
from typing import List
from pydantic import BaseModel, Field, field_validator


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
    BHK: int = Field(..., ge=1, le=10)
    Size_in_SqFt: float = Field(..., ge=100, le=20000)
    Price_per_SqFt: float = Field(..., ge=500, le=50000)
    Year_Built: int = Field(..., ge=1900, le=2026)
    Parking_Space: int = Field(..., ge=0, le=1)
    location: LocationEnum = Field(..., description="City name")

    @field_validator("location", mode="before")
    @classmethod
    def normalize_location(cls, v):
        if isinstance(v, str):
            synonyms = {
                "Delhi": "New Delhi",
                "Bombay": "Mumbai",
                "Bengaluru": "Bangalore",
                "Madras": "Chennai",
                "Calcutta": "Kolkata",
                "Gurugram": "Gurgaon",
            }
            return synonyms.get(v.strip(), v.strip())
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "BHK": 3,
                "Size_in_SqFt": 1500,
                "Price_per_SqFt": 5000,
                "Year_Built": 2015,
                "Parking_Space": 1,
                "location": "New Delhi",
            }
        }
    }


class PredictionResponse(BaseModel):
    input_summary: dict
    predicted_price_in_lakhs: float
    currency: str = "INR"
    disclaimer: str = "This is an ML-based estimate, not financial advice."


# ============================================================
# FUTURE PRICE PREDICTION (Category 1: Economic Factors)
# ============================================================

class FuturePredictionRequest(HouseFeatures):
    """Extends HouseFeatures with economic factors for future forecasting."""
    years: int = Field(..., gt=0, le=50, description="Number of years to forecast (1-50)")
    inflation_rate: float = Field(0.06, gt=0, le=0.5, description="Annual inflation rate (e.g., 0.06 for 6%)")
    interest_rate: float = Field(0.085, gt=0, le=0.5, description="Home loan interest rate (e.g., 0.085 for 8.5%)")
    gdp_growth_rate: float = Field(0.07, gt=0, le=0.2, description="Annual GDP growth rate (e.g., 0.07 for 7%)")
    migration_rate: float = Field(0.02, ge=0, le=0.2, description="Annual migration rate (e.g., 0.02 for 2%)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "BHK": 3,
                "Size_in_SqFt": 1500,
                "Price_per_SqFt": 5000,
                "Year_Built": 2015,
                "Parking_Space": 1,
                "location": "Mumbai",
                "years": 10,
                "inflation_rate": 0.06,
                "interest_rate": 0.085,
                "gdp_growth_rate": 0.07,
                "migration_rate": 0.02,
            }
        }
    }


class YearlyPrice(BaseModel):
    year: int
    price_in_lakhs: float


class FuturePredictionResponse(BaseModel):
    input_summary: dict
    current_price_in_lakhs: float
    effective_growth_rate: float
    forecast: List[YearlyPrice]
    currency: str = "INR"
    disclaimer: str = "This is an ML-based estimate, not financial advice."