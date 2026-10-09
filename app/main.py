"""
main.py
-------
App entrypoint. Loads model at startup, sets up middleware, and mounts routers.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import math

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import time
from datetime import date
import json

from app.routes.predict import router as predict_router
from app.repositories.model_repository import model_repository
from app.utils.logger import logger
from app.utils.limiter import limiter

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"


def _json_safe_validation_details(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe_validation_details(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_validation_details(item) for item in value]
    return value

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up HouseIQ...")
    model_repository.load()
    logger.info("API ready to serve requests.")
    yield


app = FastAPI(
    title="HouseIQ",
    version="1.0.0",
    description="HouseIQ estimates values from sourced city reference rates where available, returns a synthetic-data ML comparison, and provides assumption-based future scenarios.",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(request: Request, exc: RequestValidationError):
    details = _json_safe_validation_details(exc.errors())
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(details)})

# Rate limiter setup
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response


@app.get("/", tags=["Root"])
def root():
    return {"message": "HouseIQ API is running."}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok" if model_repository.loaded else "unavailable",
        "model_loaded": model_repository.loaded,
    }


@app.get("/metadata", tags=["Model"])
def metadata():
    if not model_repository.loaded:
        model_repository.load()
    info = model_repository.metadata
    return {
        "bhk_min": info["bhk_min"],
        "bhk_max": info["bhk_max"],
        "size_min_sqft": info["size_min_sqft"],
        "size_max_sqft": info["size_max_sqft"],
        "bhk_limits": info["bhk_limits"],
        "year_min": info["year_min"],
        "year_max": info["year_max"],
        "dataset_is_synthetic": True,
        "interval_coverage": info.get("calibration", {}).get("coverage"),
        "interval_test_coverage": info.get("calibration", {}).get("held_out_coverage"),
    }


@app.get("/app", include_in_schema=False)
def dashboard():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html.replace("__CURRENT_YEAR__", str(date.today().year)))


@app.get("/city-rates", tags=["Reference rates"])
def city_rates():
    return json.loads((BASE_DIR / "data" / "city_reference_rates.json").read_text(encoding="utf-8"))


app.include_router(predict_router, tags=["Prediction"])
