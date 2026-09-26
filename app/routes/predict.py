from fastapi import APIRouter, Request
from app.schemas.house import (
    HouseFeatures,
    PredictionResponse,
    FuturePredictionRequest,
    FuturePredictionResponse,
)
from app.controllers.prediction_controller import PredictionController
from app.utils.limiter import limiter  # Alag file se import karo

router = APIRouter()
controller = PredictionController()


@router.post("/predict", response_model=PredictionResponse)
@limiter.limit("30/minute")
def predict_price(request: Request, features: HouseFeatures):
    """Predict current house price based on input features."""
    return controller.predict_price(features)


@router.post("/predict/future", response_model=FuturePredictionResponse)
@limiter.limit("10/minute")
def predict_future_price(request: Request, request_body: FuturePredictionRequest):
    """Predict future house prices for N years."""
    return controller.predict_future_price(request_body)