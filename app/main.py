"""
main.py
-------
App entrypoint. Loads model at startup, sets up middleware, and mounts routers.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import time

from app.routes.predict import router as predict_router
from app.repositories.model_repository import model_repository
from app.utils.logger import logger
from app.utils.limiter import limiter  # Alag file se import karo

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up HouseIQ...")
    model_repository.load()
    logger.info("API ready to serve requests.")
    yield


app = FastAPI(
    title="HouseIQ",
    version="1.0.0",
    description="HouseIQ predicts current house prices and summarizes future price forecasts for Indian cities.",
    lifespan=lifespan,
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


@app.get("/", tags=["Root"])
def root():
    return {"message": "HouseIQ API is running."}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": model_repository.loaded}


app.include_router(predict_router, tags=["Prediction"])
