"""
ml/model.py
-----------
Trains, evaluates, saves, and loads transport demand prediction models.
Supports Linear Regression and Random Forest (ensemble) with a clean API.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data" / "transport_demand.csv"
MODEL_DIR  = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

RF_MODEL_PATH = MODEL_DIR / "random_forest.pkl"
LR_MODEL_PATH = MODEL_DIR / "linear_regression.pkl"

# ── Feature definitions ────────────────────────────────────────────────────
NUMERIC_FEATURES     = ["hour", "day_of_week", "month", "is_weekend", "temperature_c"]
CATEGORICAL_FEATURES = ["location", "weather"]
TARGET               = "demand"

# ── Preprocessor ──────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """Creates a sklearn ColumnTransformer for feature preprocessing."""
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


# ── Training ───────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Loads and validates the training CSV."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Run `python data/generate_data.py` first."
        )
    df = pd.read_csv(DATA_PATH)
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Derives additional features from the raw dataframe."""
    df = df.copy()
    # Cyclical encoding for hour and month to handle wrap-around
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    return df


def train_models(
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Trains Linear Regression and Random Forest models.
    Returns a metrics dict for both models.
    """
    logger.info("Loading dataset …")
    df = load_data()
    df = feature_engineering(df)

    # Extended numeric features after engineering
    extended_numeric = NUMERIC_FEATURES + ["hour_sin", "hour_cos", "month_sin", "month_cos"]
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), extended_numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])

    X = df[extended_numeric + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    results = {}

    # ── Linear Regression ──────────────────────────────────────────────────
    logger.info("Training Linear Regression …")
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor",    LinearRegression()),
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_metrics = _evaluate(lr_pipeline, X_test, y_test, "Linear Regression")
    joblib.dump({"pipeline": lr_pipeline, "features": extended_numeric + CATEGORICAL_FEATURES},
                LR_MODEL_PATH)
    results["linear_regression"] = lr_metrics

    # ── Random Forest ──────────────────────────────────────────────────────
    logger.info("Training Random Forest …")
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        )),
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_metrics = _evaluate(rf_pipeline, X_test, y_test, "Random Forest")
    joblib.dump({"pipeline": rf_pipeline, "features": extended_numeric + CATEGORICAL_FEATURES},
                RF_MODEL_PATH)
    results["random_forest"] = rf_metrics

    logger.info("Models saved ✓")
    return results


def _evaluate(pipeline: Pipeline, X_test, y_test, name: str) -> dict:
    """Computes regression metrics for a fitted pipeline."""
    y_pred = pipeline.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    logger.info(f"[{name}] MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}")
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4), "model": name}


# ── Prediction ─────────────────────────────────────────────────────────────

def _load_pipeline(path: Path) -> dict:
    """Loads a saved model dict from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Call POST /train first."
        )
    return joblib.load(path)


def predict(
    hour: int,
    day_of_week: int,
    month: int,
    is_weekend: int,
    temperature_c: float,
    location: str,
    weather: str,
    model_type: str = "random_forest",
) -> dict:
    """
    Runs demand prediction using the selected model.
    Returns predicted demand (trips) plus confidence interval estimate.
    """
    path = RF_MODEL_PATH if model_type == "random_forest" else LR_MODEL_PATH
    artifact = _load_pipeline(path)
    pipeline = artifact["pipeline"]

    # Build input DataFrame with feature engineering
    input_df = pd.DataFrame([{
        "hour":          hour,
        "day_of_week":   day_of_week,
        "month":         month,
        "is_weekend":    is_weekend,
        "temperature_c": temperature_c,
        "location":      location,
        "weather":       weather,
    }])
    input_df = feature_engineering(input_df)

    raw_pred = pipeline.predict(input_df)[0]
    predicted = max(0, round(float(raw_pred)))

    # Approximate confidence interval (±10% for RF, ±15% for LR)
    margin = 0.10 if model_type == "random_forest" else 0.15
    ci_low  = max(0, round(predicted * (1 - margin)))
    ci_high = round(predicted * (1 + margin))

    return {
        "predicted_demand": predicted,
        "confidence_interval": {"low": ci_low, "high": ci_high},
        "model_used": model_type,
        "unit": "estimated trips per hour",
    }


def models_exist() -> bool:
    """Returns True if both model files are present on disk."""
    return RF_MODEL_PATH.exists() and LR_MODEL_PATH.exists()
