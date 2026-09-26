from app.schemas.house import (
    HouseFeatures,
    PredictionResponse,
    FuturePredictionRequest,
    FuturePredictionResponse,
)
from app.services.prediction_service import PredictionService
from app.utils.logger import logger


class PredictionController:
    def __init__(self):
        self.service = PredictionService()

    def predict_price(self, features: HouseFeatures) -> PredictionResponse:
        price = self.service.predict(features)
        input_summary = {
            "location": str(features.location.value) if hasattr(features.location, "value") else str(features.location),
            "BHK": features.BHK,
            "Size_in_SqFt": features.Size_in_SqFt,
            "Price_per_SqFt": features.Price_per_SqFt,
            "Year_Built": features.Year_Built,
            "Parking_Space": features.Parking_Space,
        }
        return PredictionResponse(
            input_summary=input_summary,
            predicted_price_in_lakhs=round(price, 2),
            currency="INR",
        )

    def predict_future_price(self, request: FuturePredictionRequest) -> FuturePredictionResponse:
        return self.service.predict_future(request)