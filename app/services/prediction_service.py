import pandas as pd
from app.schemas.house import (
    HouseFeatures,
    FuturePredictionRequest,
    FuturePredictionResponse,
    YearlyPrice,
)
from app.repositories.model_repository import model_repository
from app.utils.logger import logger


class PredictionService:

    def _prepare_features(self, features: HouseFeatures) -> pd.DataFrame:
        data = features.model_dump()
        location = data.pop("location")

        if hasattr(location, "value"):
            location = location.value
        location = location.strip().title()

        synonyms = {
            "Delhi": "New Delhi",
            "Bombay": "Mumbai",
            "Bengaluru": "Bangalore",
            "Madras": "Chennai",
            "Calcutta": "Kolkata",
            "Gurugram": "Gurgaon",
        }
        location = synonyms.get(location, location)

        for col in model_repository.columns:
            if col.startswith("loc_"):
                data[col] = 0

        loc_col = f"loc_{location}"
        if loc_col in data:
            data[loc_col] = 1
        elif "loc_Other" in data:
            data["loc_Other"] = 1

        df = pd.DataFrame([data])
        df = df[model_repository.columns]
        return df

    def predict(self, features: HouseFeatures) -> float:
        if not model_repository.loaded:
            model_repository.load()

        logger.info(f"Current prediction requested: location={features.location}, BHK={features.BHK}")
        df = self._prepare_features(features)
        prediction = model_repository.model.predict(df)[0]
        result = round(float(prediction), 2)
        logger.info(f"Prediction result: {result} lakhs")
        return result

    def predict_future(self, request: FuturePredictionRequest) -> FuturePredictionResponse:
        if not model_repository.loaded:
            model_repository.load()

        logger.info(f"Future prediction requested: location={request.location}, years={request.years}")

        # Current price
        current_price = self.predict(request)

        # Effective growth rate
        effective_rate = (
            request.inflation_rate
            + request.gdp_growth_rate
            + request.migration_rate
            - (request.interest_rate * 0.5)
        )

        # Forecast
        forecast = []
        for year in range(1, request.years + 1):
            future_price = current_price * ((1 + effective_rate) ** year)
            forecast.append(YearlyPrice(year=year, price_in_lakhs=round(future_price, 2)))

        logger.info(f"Forecast generated for {request.years} years, effective rate={effective_rate:.4f}")

        # Input summary banane ke liye
        input_summary = {
            "location": str(request.location.value) if hasattr(request.location, "value") else str(request.location),
            "BHK": request.BHK,
            "Size_in_SqFt": request.Size_in_SqFt,
            "Price_per_SqFt": request.Price_per_SqFt,
            "Year_Built": request.Year_Built,
            "Parking_Space": request.Parking_Space,
            "years": request.years,
            "inflation_rate": request.inflation_rate,
            "interest_rate": request.interest_rate,
            "gdp_growth_rate": request.gdp_growth_rate,
            "migration_rate": request.migration_rate,
        }

        return FuturePredictionResponse(
            input_summary=input_summary,
            current_price_in_lakhs=current_price,
            effective_growth_rate=round(effective_rate, 4),
            forecast=forecast,
            currency="INR",
        )


prediction_service = PredictionService()