from app.schemas.house import (
    FuturePredictionRequest,
    FuturePredictionResponse,
    HouseFeatures,
    PredictionResponse,
)
from app.services.prediction_service import PredictionService


class PredictionController:
    def __init__(self):
        self.service = PredictionService()

    def predict_price(self, features: HouseFeatures) -> PredictionResponse:
        return self.service.predict(features)

    def predict_future_price(self, request: FuturePredictionRequest) -> FuturePredictionResponse:
        return self.service.predict_future(request)
