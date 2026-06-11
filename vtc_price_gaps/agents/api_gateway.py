"""Agent M5 — API Gateway: retourne les 3 meilleurs trous de prix pour un utilisateur."""
from __future__ import annotations
import json, logging, math
from datetime import datetime, timezone
from typing import List, Optional
import httpx, h3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from shared.config import cfg
from shared.db import get_pg_pool, get_redis
from shared.geo_utils import haversine_km, bearing_deg, generate_candidate_cells

logger = logging.getLogger(__name__)
app = FastAPI(title="VTC Price Gaps API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class OpportunityRequest(BaseModel):
    pickup_lat: float; pickup_lon: float
    dropoff_lat: float; dropoff_lon: float
    city_id: str; max_walk_minutes: float = 6.0

class OpportunityResult(BaseModel):
    rank: int; candidate_lat: float; candidate_lon: float; h3_cell_id: str
    walk_distance_m: float; walk_time_min: float
    direction_cardinal: str; direction_deg: float
    price_current: float; price_candidate: float
    gain_eur: float; gain_pct: float
    confidence: float; stability: float; explanation: str

class OpportunityResponse(BaseModel):
    opportunities: List[OpportunityResult]
    current_price: float; city_id: str; computed_at: str; cache_hit: bool

def _to_cardinal(deg):
    return ["N","NE","E","SE","S","SO","O","NO"][round(deg/45) % 8]

def _explain(r):
    cardinal = _to_cardinal(r["direction"])
    stab = "souvent" if r["stability"] > 0.6 else "parfois"
    return (f"En marchant ~{int(r['walk_dist_m'])}m vers le {cardinal} "
            f"({r['walk_min']:.1f} min), économisez ~{r['gain_eur']:.2f}€. "
            f"Cet avantage est {stab} présent dans ce contexte.")

def _deduplicate(results, min_dist_m=150):
    kept = []
    for r in results:
        if not any(haversine_km(r["clat"],r["clon"],k["clat"],k["clon"])*1000 < min_dist_m for k in kept):
            kept.append(r)
    return kept

async def _fetch_h3_feats(pool, cell_id, city_id):
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT price_median AS h3r8_price_median_7d, price_std AS h3r8_price_std_7d,
                   trip_count AS h3r8_trip_count_7d, duration_median AS h3r8_duration_med_7d,
                   neighbor_price_mean AS h3r8_neighbor_mean,
                   price_vs_neighbors AS h3r8_price_vs_neighbors,
                   is_price_minimum AS h3r8_is_minimum
            FROM h3_price_stats WHERE h3_cell_id=$1 AND city_id=$2 AND window_days=7
            ORDER BY computed_at DESC LIMIT 1""", cell_id, city_id)
    return dict(row) if row else {}

async def _get_stability_fast(pool, src, tgt, city_id, hour, dow):
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT stability_score FROM price_opportunities
            WHERE h3_source_cell=$1 AND h3_target_cell=$2
              AND hour_bucket=$3 AND dow_bucket=$4
            ORDER BY valid_from DESC LIMIT 1""", src, tgt, hour, dow)
    return float(row["stability_score"]) if row else 0.3

@app.post("/opportunities", response_model=OpportunityResponse)
async def get_opportunities(req: OpportunityRequest):
    pool = await get_pg_pool(); redis = await get_redis()
    now = datetime.now(timezone.utc); hour, dow = now.hour, now.weekday()
    source_cell  = h3.latlng_to_cell(req.pickup_lat, req.pickup_lon, cfg.h3_main_res)
    dropoff_cell = h3.latlng_to_cell(req.dropoff_lat, req.dropoff_lon, cfg.h3_main_res)
    cache_key = f"opp:{req.city_id}:{source_cell}:{dropoff_cell}:h{hour}:d{dow}"
    cached = await redis.get(cache_key)
    if cached:
        data = json.loads(cached); data["cache_hit"] = True; return OpportunityResponse(**data)
    async with httpx.AsyncClient(base_url=cfg.ml_service_url, timeout=10.0) as client:
        h3_feats = await _fetch_h3_feats(pool, source_cell, req.city_id)
        baseline = await client.post("/predict", json={
            "city_id":req.city_id,"pickup_lat":req.pickup_lat,"pickup_lon":req.pickup_lon,
            "dropoff_lat":req.dropoff_lat,"dropoff_lon":req.dropoff_lon,
            "hour":hour,"dow":dow,"is_weekend":dow>=5,"month":now.month,**h3_feats})
        current_price = baseline.json()["price_mean"]
        candidates = generate_candidate_cells(req.pickup_lat, req.pickup_lon, req.max_walk_minutes)
        if not candidates:
            return OpportunityResponse(opportunities=[],current_price=round(current_price,2),
                city_id=req.city_id,computed_at=now.isoformat(),cache_hit=False)
        batch = await client.post("/predict/batch", json={
            "city_id":req.city_id,"hour":hour,"dow":dow,"is_weekend":dow>=5,
            "dropoff_lat":req.dropoff_lat,"dropoff_lon":req.dropoff_lon,
            "points":[{"cell_id":c[0],"lat":c[1],"lon":c[2]} for c in candidates]})
        predictions = batch.json()["predictions"]
    results = []
    for cand, pred in zip(candidates, predictions):
        cell_id, clat, clon, dist_km, walk_min = cand
        gain_eur = current_price - pred["price_mean"]
        if gain_eur < cfg.min_gain_eur: continue
        gain_pct = gain_eur / current_price if current_price > 0 else 0
        if gain_pct < cfg.min_gain_pct: continue
        direction = bearing_deg(req.pickup_lat, req.pickup_lon, clat, clon)
        stability = await _get_stability_fast(pool, source_cell, cell_id, req.city_id, hour, dow)
        confidence = max(0, 1 - pred["price_std"] / max(current_price, 1))
        score = gain_eur - cfg.score_alpha*walk_min - cfg.score_beta*pred["price_std"] + cfg.score_gamma*stability
        if score <= 0: continue
        results.append({"cell_id":cell_id,"clat":clat,"clon":clon,"walk_dist_m":dist_km*1000,
                        "walk_min":walk_min,"direction":direction,"gain_eur":gain_eur,
                        "gain_pct":gain_pct,"confidence":confidence,"stability":stability,
                        "score":score,"price_candidate":pred["price_mean"]})
    top3 = _deduplicate(sorted(results, key=lambda x:x["score"], reverse=True), 150)[:cfg.max_api_results]
    opportunities = [
        OpportunityResult(rank=i+1,candidate_lat=r["clat"],candidate_lon=r["clon"],
            h3_cell_id=r["cell_id"],walk_distance_m=round(r["walk_dist_m"]),
            walk_time_min=round(r["walk_min"],1),direction_cardinal=_to_cardinal(r["direction"]),
            direction_deg=round(r["direction"],1),price_current=round(current_price,2),
            price_candidate=round(r["price_candidate"],2),gain_eur=round(r["gain_eur"],2),
            gain_pct=round(r["gain_pct"]*100,1),confidence=round(r["confidence"],2),
            stability=round(r["stability"],2),explanation=_explain(r))
        for i,r in enumerate(top3)]
    response_data = {"opportunities":[o.model_dump() for o in opportunities],
                     "current_price":round(current_price,2),"city_id":req.city_id,
                     "computed_at":now.isoformat(),"cache_hit":False}
    await redis.setex(cache_key, cfg.opportunity_cache_ttl, json.dumps(response_data))
    return OpportunityResponse(**response_data)

@app.get("/health")
def health(): return {"status": "ok"}