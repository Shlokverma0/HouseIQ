"""
main.py
-------
App entrypoint. Loads model at startup, sets up middleware, and mounts routers.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import time

from app.routes.predict import router as predict_router
from app.repositories.model_repository import model_repository
from app.utils.logger import logger
from app.utils.limiter import limiter  # Alag file se import karo

app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
    description="Predict house price (in Lakhs INR) from property features.",
)

# Rate limiter setup
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response


@app.on_event("startup")
def load_model():
    logger.info("Starting up House Price Prediction API...")
    model_repository.load()
    logger.info("API ready to serve requests.")


@app.get("/", tags=["Root"])
def root():
    return {"message": "House Price Prediction API is running."}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": model_repository.loaded}


app.include_router(predict_router, tags=["Prediction"])