"""
app/main.py
-----------
FastAPI application entry point.
Exposes:  GET  /health
          POST /predict
          POST /train
          GET  /predictions  (history)
"""

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Make sure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas import (
    HealthResponse, PredictRequest, PredictResponse,
    TrainRequest, TrainResponse,
)
from database.db import get_db, init_db, log_prediction
from ml.model import models_exist, predict, train_models

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

APP_VERSION = "1.0.0"


# ── Startup / Shutdown ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise DB and auto-train models if not already present."""
    logger.info("Starting Transport Demand Prediction API …")
    init_db()
    if not models_exist():
        logger.info("No models found — running initial training …")
        try:
            train_models()
            logger.info("Initial training complete ✓")
        except FileNotFoundError as exc:
            logger.warning(f"Dataset not found, skipping auto-train: {exc}")
    yield
    logger.info("Shutting down …")


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Transport Demand Prediction API",
    description=(
        "Predicts transport demand (trips/hour) based on date, time, "
        "location, and weather using Linear Regression & Random Forest."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

# Allow the React frontend (localhost:3000 / 5173) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Quick liveness check — returns model readiness status."""
    return HealthResponse(
        status="ok",
        models_ready=models_exist(),
        version=APP_VERSION,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_demand(
    body: PredictRequest,
    db: Session = Depends(get_db),
):
    """
    Predict transport demand for a given date, time, location, and weather.

    - **date**: YYYY-MM-DD
    - **hour**: 0–23
    - **location**: downtown | suburb | airport | university | shopping_mall
    - **weather**: sunny | cloudy | rainy | snowy | windy
    - **model_type**: random_forest (default) | linear_regression
    """
    if not models_exist():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not trained yet. Call POST /train first.",
        )

    # Parse date into derived features
    try:
        dt         = datetime.strptime(body.date, "%Y-%m-%d")
        day_of_week = dt.weekday()          # 0=Mon … 6=Sun
        month       = dt.month
        is_weekend  = int(dt.weekday() >= 5)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        result = predict(
            hour          = body.hour,
            day_of_week   = day_of_week,
            month         = month,
            is_weekend     = is_weekend,
            temperature_c = body.temperature_c or 20.0,
            location      = body.location.value,
            weather       = body.weather.value,
            model_type    = body.model_type.value,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs.")

    # Persist to DB (non-blocking)
    input_data = body.model_dump()
    log_prediction(db, input_data, result)

    return PredictResponse(
        predicted_demand    = result["predicted_demand"],
        confidence_interval = result["confidence_interval"],
        model_used          = result["model_used"],
        unit                = result["unit"],
        input_summary       = {
            "date":       body.date,
            "hour":       body.hour,
            "location":   body.location.value,
            "weather":    body.weather.value,
            "is_weekend": bool(is_weekend),
        },
    )


@app.post("/train", response_model=TrainResponse, tags=["Model"])
def retrain_models(body: TrainRequest = TrainRequest()):
    """
    Trigger a full model retrain from the CSV dataset.
    Returns evaluation metrics for both models.
    """
    try:
        metrics = train_models(
            test_size    = body.test_size,
            random_state = body.random_state,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Training error")
        raise HTTPException(status_code=500, detail="Training failed. Check server logs.")

    return TrainResponse(
        message="Models trained and saved successfully ✓",
        metrics=metrics,
    )


@app.get("/predictions", tags=["History"])
def get_predictions(limit: int = 20, db: Session = Depends(get_db)):
    """Returns recent prediction history (latest first)."""
    from database.db import PredictionLog
    rows = (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id":               r.id,
            "created_at":       r.created_at.isoformat(),
            "date":             r.input_date,
            "hour":             r.input_hour,
            "location":         r.input_location,
            "weather":          r.input_weather,
            "predicted_demand": r.predicted_demand,
            "model_used":       r.model_used,
        }
        for r in rows
    ]
