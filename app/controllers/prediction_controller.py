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
        return PredictionResponse(
            predicted_price_in_lakhs=round(price, 2),
            currency="INR",
        )

    def predict_future_price(self, request: FuturePredictionRequest) -> FuturePredictionResponse:
        return self.service.predict_future(request)
