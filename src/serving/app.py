"""FastAPI Serving Application (Phase 10 / Interactive Demo).

Provides REST APIs and an interactive web interface for real-time inference
using the 3-tier LightGBM models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.features.build_features import (
    FEATURE_COLUMNS,
    TIERS,
    assign_tier,
)
from src.models.tier_model import TierModel

logger = logging.getLogger("eia.serving")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "features.parquet"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Energy Demand Forecasting MLOps",
    description="Next-day regional electricity demand forecasting across US balancing authorities.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory model cache
loaded_models: dict[str, TierModel] = {}
features_cache: pd.DataFrame | None = None


def get_models() -> dict[str, TierModel]:
    global loaded_models
    if not loaded_models and MODELS_DIR.exists():
        for tier in ["Large", "Medium", "Small"]:
            m_path = MODELS_DIR / f"model_tier_{tier.lower()}.joblib"
            if m_path.exists():
                try:
                    loaded_models[tier] = TierModel.load(m_path)
                    logger.info("loaded %s tier model from %s", tier, m_path)
                except Exception as e:
                    logger.warning("could not load %s model: %s", tier, e)
    return loaded_models


def get_features_dataset() -> pd.DataFrame | None:
    global features_cache
    if features_cache is None and DATA_PROCESSED.exists():
        try:
            features_cache = pd.read_parquet(DATA_PROCESSED)
            logger.info("cached features dataset: %d rows", len(features_cache))
        except Exception as e:
            logger.warning("could not load features parquet: %s", e)
    return features_cache


class PredictRequest(BaseModel):
    respondent: str = Field(..., json_schema_extra={"example": "PJM"}, description="Balancing authority code")
    target_date: str | None = Field(None, json_schema_extra={"example": "2025-06-01"}, description="Target date (YYYY-MM-DD)")


class PredictResponse(BaseModel):
    respondent: str
    tier: str
    forecast_date: str
    target_date: str
    predicted_demand_mwh: float
    naive_persistence_mwh: float | None = None
    eia_day_ahead_mwh: float | None = None
    actual_demand_mwh: float | None = None
    model_abs_error_mwh: float | None = None
    model_error_pct: float | None = None
    eia_error_pct: float | None = None
    beats_naive: bool | None = None
    beats_eia: bool | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    models = get_models()
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "features_available": DATA_PROCESSED.exists(),
    }


@app.post("/admin/reload-models")
def reload_models() -> dict[str, Any]:
    """Drop the in-memory model cache so the next request loads from disk.

    Called by the retraining pipeline after a promotion so newly trained
    artifacts take effect without restarting the server.
    """
    global loaded_models, features_cache
    loaded_models = {}
    features_cache = None
    models = get_models()
    return {"status": "reloaded", "models_loaded": list(models.keys())}


@app.get("/api/regions")
def get_regions() -> dict[str, Any]:
    """Return balancing authorities grouped by tier."""
    return {
        "tiers": TIERS,
        "total_regions": sum(len(v) for v in TIERS.values()),
    }


@app.get("/api/history")
def get_history(
    respondent: str = Query(..., description="Balancing authority code"),
    limit: int = Query(60, description="Number of recent days to return"),
) -> dict[str, Any]:
    """Return historical actual demand and benchmarks for charting."""
    df = get_features_dataset()
    if df is None or df.empty:
        raise HTTPException(status_code=503, detail="Feature store not generated yet")

    sub = df[df["respondent"] == respondent.upper()].sort_values("target_date").tail(limit)
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"No data for respondent {respondent}")

    records = []
    for _, row in sub.iterrows():
        records.append({
            "target_date": str(pd.to_datetime(row["target_date"]).date()),
            "actual_demand": round(float(row["target"]), 1) if not pd.isna(row["target"]) else None,
            "eia_forecast": round(float(row["benchmark_eia_forecast"]), 1) if not pd.isna(row.get("benchmark_eia_forecast")) else None,
            "naive_persistence": round(float(row["naive_persistence"]), 1) if not pd.isna(row.get("naive_persistence")) else None,
        })

    return {
        "respondent": respondent.upper(),
        "tier": assign_tier(respondent.upper()),
        "count": len(records),
        "history": records,
    }


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    resp = req.respondent.upper()
    tier = assign_tier(resp)
    models = get_models()

    if tier not in models:
        raise HTTPException(
            status_code=503,
            detail=f"Model for tier '{tier}' is not loaded. Train models first via scripts/train_and_evaluate.py",
        )

    model = models[tier]
    df = get_features_dataset()
    if df is None or df.empty:
        raise HTTPException(status_code=503, detail="Feature store not available")

    sub = df[df["respondent"] == resp].sort_values("target_date")
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"No features available for {resp}")

    # Select row matching date or default to latest
    if req.target_date:
        target_ts = pd.Timestamp(req.target_date)
        match = sub[sub["target_date"] == target_ts]
        if match.empty:
            raise HTTPException(status_code=404, detail=f"Date {req.target_date} not found for {resp}")
        row = match.iloc[0:1]
    else:
        row = sub.iloc[-1:]

    # Predict with TierModel
    pred = float(model.predict(row)[0])

    actual = float(row["target"].iloc[0]) if not pd.isna(row["target"].iloc[0]) else None
    naive = float(row["naive_persistence"].iloc[0]) if not pd.isna(row["naive_persistence"].iloc[0]) else None
    eia = float(row["benchmark_eia_forecast"].iloc[0]) if not pd.isna(row["benchmark_eia_forecast"].iloc[0]) else None

    abs_err = abs(actual - pred) if actual is not None else None
    model_err_pct = (abs_err / actual * 100.0) if (actual and abs_err is not None) else None
    eia_err_pct = (abs(actual - eia) / actual * 100.0) if (actual and eia is not None) else None
    naive_err_pct = (abs(actual - naive) / actual * 100.0) if (actual and naive is not None) else None

    beats_naive = (abs_err < abs(actual - naive)) if (abs_err is not None and actual and naive) else None
    beats_eia = (abs_err < abs(actual - eia)) if (abs_err is not None and actual and eia) else None

    return PredictResponse(
        respondent=resp,
        tier=tier,
        forecast_date=str(pd.to_datetime(row["forecast_date"].iloc[0]).date()),
        target_date=str(pd.to_datetime(row["target_date"].iloc[0]).date()),
        predicted_demand_mwh=round(pred, 1),
        naive_persistence_mwh=round(naive, 1) if naive is not None else None,
        eia_day_ahead_mwh=round(eia, 1) if eia is not None else None,
        actual_demand_mwh=round(actual, 1) if actual is not None else None,
        model_abs_error_mwh=round(abs_err, 1) if abs_err is not None else None,
        model_error_pct=round(model_err_pct, 2) if model_err_pct is not None else None,
        eia_error_pct=round(eia_err_pct, 2) if eia_err_pct is not None else None,
        beats_naive=beats_naive,
        beats_eia=beats_eia,
    )


@app.get("/")
def serve_ui():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>UI not installed yet</h1>")
    return FileResponse(index_file)
