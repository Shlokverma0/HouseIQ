import pandas as pd
from app.schemas.house import (
    HouseFeatures,
    FuturePredictionRequest,
    FuturePredictionResponse,
    ForecastPeriod,
)
from app.repositories.model_repository import model_repository
from app.utils.logger import logger


class PredictionService:

    @staticmethod
    def _format_price(price_in_lakhs: float) -> str:
        if price_in_lakhs >= 100:
            return f"₹{price_in_lakhs / 100:,.2f} crore"
        return f"₹{price_in_lakhs:,.2f} lakh"

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

        # Calculate yearly prices, then return compact 10-year ranges.
        yearly_prices = {}
        for year in range(1, request.years + 1):
            future_price = current_price * ((1 + effective_rate) ** year)
            yearly_prices[year] = round(future_price, 2)

        forecast = []
        for start_year in range(1, request.years + 1, 10):
            end_year = min(start_year + 9, request.years)
            previous_price = current_price if start_year == 1 else yearly_prices[start_year - 1]
            ending_price = yearly_prices[end_year]
            forecast.append(
                ForecastPeriod(
                    period=f"Years {start_year}–{end_year}",
                    start_year=start_year,
                    end_year=end_year,
                    previous_price_in_lakhs=previous_price,
                    ending_price_in_lakhs=ending_price,
                    previous_price_display=self._format_price(previous_price),
                    ending_price_display=self._format_price(ending_price),
                )
            )

        logger.info(f"Forecast generated for {request.years} years, effective rate={effective_rate:.4f}")

        return FuturePredictionResponse(
            current_price_in_lakhs=current_price,
            current_price_display=self._format_price(current_price),
            effective_growth_rate=round(effective_rate, 4),
            forecast=forecast,
            currency="INR",
        )


prediction_service = PredictionService()
