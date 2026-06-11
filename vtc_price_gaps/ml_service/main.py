"""
ML Service — FastAPI servant les modèles LightGBM.
Expose /predict (un point) et /predict/batch (liste de points).
Chargement paresseux du dernier modèle disponible par ville.
"""
from __future__ import annotations
import logging
import math
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
app = FastAPI(title="VTC ML Service", version="1.0")

_models: dict = {}
MODEL_PATH = Path("/app/models")

FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "is_rush_am", "is_rush_pm", "is_night", "minutes_since_midnight",
    "distance_km", "bearing_sin", "bearing_cos", "delta_lat", "delta_lon",
    "h3r8_price_median_7d", "h3r8_price_std_7d", "h3r8_trip_count_7d",
    "h3r8_duration_med_7d", "h3r8_price_median_30d", "h3r8_price_p10_30d",
    "h3r8_neighbor_mean", "h3r8_price_vs_neighbors", "h3r8_is_minimum",
]


# ─── Chargement du modèle ─────────────────────────────────────────

def load_model(city_id: str) -> dict | None:
    """Charge la dernière version disponible pour une ville."""
    city_dir = MODEL_PATH / city_id
    if not city_dir.exists():
        return None
    versions = sorted(
        [v for v in city_dir.iterdir() if v.is_dir()],
        reverse=True,
    )
    for v in versions:
        artefact = v / "artifacts.pkl"
        if artefact.exists():
            with open(artefact, "rb") as f:
                artifacts = pickle.load(f)
            logger.info(f"[ML] Modèle chargé: {city_id}/{v.name} "
                        f"(test_mae={artifacts.get('test_mae','?')}€)")
            return artifacts
    return None


def get_model(city_id: str) -> dict:
    if city_id not in _models or _models[city_id] is None:
        m = load_model(city_id)
        if m is None:
            raise HTTPException(
                status_code=503,
                detail=f"Aucun modèle disponible pour la ville '{city_id}'. "
                       f"Lancez d'abord l'agent M3 pour entraîner un modèle."
            )
        _models[city_id] = m
    return _models[city_id]


# ─── Construction des features ────────────────────────────────────

def build_features(data: dict) -> list[float]:
    """Construit le vecteur de features depuis un dict de requête."""
    lat1 = data["pickup_lat"]
    lon1 = data["pickup_lon"]
    lat2 = data["dropoff_lat"]
    lon2 = data["dropoff_lon"]
    hour = data.get("hour", 12)
    dow  = data.get("dow", 0)
    month = data.get("month", 6)
    is_weekend = data.get("is_weekend", dow >= 5)

    # Haversine
    R = 6371.0
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    dist_km = 2 * R * math.asin(math.sqrt(max(0, a)))

    # Bearing
    dx = math.sin(dlon) * math.cos(lat2r)
    dy = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    bearing = math.atan2(dx, dy)

    return [
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * dow / 7),
        math.cos(2 * math.pi * dow / 7),
        math.sin(2 * math.pi * month / 12),
        math.cos(2 * math.pi * month / 12),
        float(is_weekend),
        float(7 <= hour < 9),
        float(17 <= hour < 20),
        float(hour < 6 or hour >= 23),
        hour * 60,
        dist_km,
        math.sin(bearing),
        math.cos(bearing),
        lat2 - lat1,
        lon2 - lon1,
        float(data.get("h3r8_price_median_7d") or 0),
        float(data.get("h3r8_price_std_7d") or 0),
        float(data.get("h3r8_trip_count_7d") or 0),
        float(data.get("h3r8_duration_med_7d") or 0),
        float(data.get("h3r8_price_median_30d") or 0),
        float(data.get("h3r8_price_p10_30d") or 0),
        float(data.get("h3r8_neighbor_mean") or 0),
        float(data.get("h3r8_price_vs_neighbors") or 0),
        float(bool(data.get("h3r8_is_minimum") or False)),
    ]


# ─── Schemas Pydantic ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    city_id: str
    pickup_lat: float
    pickup_lon: float
    dropoff_lat: float
    dropoff_lon: float
    hour: int = 12
    dow: int = 0
    is_weekend: bool = False
    month: int = 6
    h3r8_price_median_7d: Optional[float] = 0
    h3r8_price_std_7d: Optional[float] = 0
    h3r8_trip_count_7d: Optional[int] = 0
    h3r8_duration_med_7d: Optional[float] = 0
    h3r8_price_median_30d: Optional[float] = 0
    h3r8_price_p10_30d: Optional[float] = 0
    h3r8_neighbor_mean: Optional[float] = 0
    h3r8_price_vs_neighbors: Optional[float] = 0
    h3r8_is_minimum: Optional[bool] = False


class PointIn(BaseModel):
    cell_id: str
    lat: float
    lon: float


class BatchRequest(BaseModel):
    city_id: str
    hour: int
    dow: int
    is_weekend: bool
    dropoff_lat: float
    dropoff_lon: float
    month: int = 6
    points: List[PointIn]


# ─── Endpoints ───────────────────────────────────────────────────

@app.post("/predict")
def predict(req: PredictRequest):
    """Prédit le prix pour un unique point de pickup."""
    m = get_model(req.city_id)
    feats = [build_features(req.model_dump())]
    p_med = float(m["model"].predict(feats)[0])
    p_q10 = float(m["model_q10"].predict(feats)[0])
    p_q90 = float(m["model_q90"].predict(feats)[0])
    return {
        "price_mean": max(0.0, p_med),
        "price_q10":  max(0.0, p_q10),
        "price_q90":  max(0.0, p_q90),
        "price_std":  (p_q90 - p_q10) / 2,
    }


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    """Prédit le prix pour une liste de points de pickup (même destination)."""
    m = get_model(req.city_id)

    all_feats = []
    for pt in req.points:
        data = {
            "pickup_lat":  pt.lat,
            "pickup_lon":  pt.lon,
            "dropoff_lat": req.dropoff_lat,
            "dropoff_lon": req.dropoff_lon,
            "hour":        req.hour,
            "dow":         req.dow,
            "is_weekend":  req.is_weekend,
            "month":       req.month,
        }
        all_feats.append(build_features(data))

    preds_med = m["model"].predict(all_feats)
    preds_q10 = m["model_q10"].predict(all_feats)
    preds_q90 = m["model_q90"].predict(all_feats)

    return {
        "predictions": [
            {
                "cell_id":    pt.cell_id,
                "price_mean": max(0.0, float(p)),
                "price_q10":  max(0.0, float(q10)),
                "price_q90":  max(0.0, float(q90)),
                "price_std":  (float(q90) - float(q10)) / 2,
            }
            for pt, p, q10, q90 in zip(req.points, preds_med, preds_q10, preds_q90)
        ]
    }


@app.post("/reload/{city_id}")
def reload_model(city_id: str):
    """Force le rechargement du dernier modèle pour une ville."""
    _models.pop(city_id, None)
    m = load_model(city_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"Aucun modèle pour {city_id}")
    _models[city_id] = m
    return {
        "status": "reloaded",
        "city_id": city_id,
        "version": m.get("model_version"),
        "test_mae": m.get("test_mae"),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(_models.keys()),
        "model_versions": {k: v.get("model_version") for k, v in _models.items()},
    }
