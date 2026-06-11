from __future__ import annotations
import math
import h3
from typing import List, Tuple
from shared.config import cfg

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def latlng_to_cells(lat, lon) -> dict:
    return {res: h3.latlng_to_cell(lat, lon, res) for res in cfg.h3_resolutions}

def generate_candidate_cells(lat, lon, max_walk_minutes=None, resolution=None):
    """Retourne liste de (cell_id, lat, lon, dist_km, walk_min)."""
    max_walk = max_walk_minutes or cfg.max_walk_minutes
    res = resolution or cfg.h3_main_res
    walk_radius_km = max_walk * cfg.walk_speed_m_per_min / 1000.0
    center_cell = h3.latlng_to_cell(lat, lon, res)
    candidates = []
    for ring_k in range(1, cfg.n_rings + 1):
        for cell in h3.grid_ring(center_cell, ring_k):
            clat, clon = h3.cell_to_latlng(cell)
            dist_km = haversine_km(lat, lon, clat, clon)
            if dist_km <= walk_radius_km:
                walk_min = (dist_km * 1000) / cfg.walk_speed_m_per_min
                candidates.append((cell, clat, clon, dist_km, walk_min))
    candidates.sort(key=lambda x: x[3])
    return candidates[:cfg.max_candidates]

def compute_temporal_features(ts) -> dict:
    hour = ts.hour + ts.minute / 60.0
    dow, month = ts.weekday(), ts.month
    return {
        "hour_sin": math.sin(2*math.pi*hour/24), "hour_cos": math.cos(2*math.pi*hour/24),
        "dow_sin":  math.sin(2*math.pi*dow/7),   "dow_cos":  math.cos(2*math.pi*dow/7),
        "month_sin":math.sin(2*math.pi*month/12),"month_cos":math.cos(2*math.pi*month/12),
        "is_weekend": dow >= 5, "is_rush_am": 7 <= ts.hour < 9,
        "is_rush_pm": 17 <= ts.hour < 20, "is_night": ts.hour < 6 or ts.hour >= 23,
        "minutes_since_midnight": ts.hour * 60 + ts.minute,
    }