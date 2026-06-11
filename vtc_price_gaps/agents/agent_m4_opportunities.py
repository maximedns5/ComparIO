"""Agent M4 — Détection des trous de prix et alimentation de price_opportunities."""
from __future__ import annotations
import asyncio, logging
from datetime import datetime, timezone
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
import httpx, h3, numpy as np
from shared.config import cfg
from shared.db import get_pg_pool
from shared.geo_utils import haversine_km, bearing_deg, generate_candidate_cells

logger = logging.getLogger(__name__)

class M4State(TypedDict):
    city_id: str; hour_bucket: int; dow_bucket: int; is_weekend: bool
    batch_size: int; opportunities_written: int; errors: list; status: str

def _score(gain_eur, walk_min, uncertainty, stability):
    return gain_eur - cfg.score_alpha*walk_min - cfg.score_beta*uncertainty + cfg.score_gamma*stability

async def node_fetch_active_cells(state: M4State) -> M4State:
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT DISTINCT tf.h3_pickup_r8 AS cell_id,
                   AVG(tc.pickup_lat) AS cell_lat, AVG(tc.pickup_lon) AS cell_lon
            FROM trip_features tf JOIN trips_clean tc ON tf.trip_id = tc.trip_id
            WHERE tc.city_id = $1 AND EXTRACT(HOUR FROM tc.pickup_ts) = $2
              AND EXTRACT(DOW FROM tc.pickup_ts) = $3
              AND tc.pickup_ts >= NOW() - INTERVAL '30 days'
              AND tf.h3_pickup_r8 IS NOT NULL
            GROUP BY tf.h3_pickup_r8 LIMIT $4
        """, state["city_id"], state["hour_bucket"], state["dow_bucket"], state["batch_size"])
    state["_active_cells"] = [dict(r) for r in rows]
    logger.info(f"[M4] {len(rows)} cellules actives")
    return state

async def _get_stability(pool, src, tgt, city_id, hour, dow):
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            WITH s AS (SELECT tc.price, DATE_TRUNC('day',tc.pickup_ts) AS day
                       FROM trip_features tf JOIN trips_clean tc ON tf.trip_id=tc.trip_id
                       WHERE tf.h3_pickup_r8=$1 AND tc.city_id=$3
                         AND EXTRACT(HOUR FROM tc.pickup_ts)=$4 AND EXTRACT(DOW FROM tc.pickup_ts)=$5),
                 t AS (SELECT tc.price, DATE_TRUNC('day',tc.pickup_ts) AS day
                       FROM trip_features tf JOIN trips_clean tc ON tf.trip_id=tc.trip_id
                       WHERE tf.h3_pickup_r8=$2 AND tc.city_id=$3
                         AND EXTRACT(HOUR FROM tc.pickup_ts)=$4 AND EXTRACT(DOW FROM tc.pickup_ts)=$5)
            SELECT AVG(CASE WHEN t.price < s.price THEN 1.0 ELSE 0.0 END) AS stability
            FROM s JOIN t USING (day)
        """, src, tgt, city_id, hour, dow)
    return float(row["stability"]) if row and row["stability"] else 0.0

async def node_compute_opportunities(state: M4State) -> M4State:
    active_cells = state.get("_active_cells", [])
    if not active_cells: state["opportunities_written"] = 0; return state
    pool = await get_pg_pool(); written = 0
    async with httpx.AsyncClient(base_url=cfg.ml_service_url, timeout=15.0) as client:
        for cell in active_cells:
            src_cell, src_lat, src_lon = cell["cell_id"], cell["cell_lat"], cell["cell_lon"]
            candidates = generate_candidate_cells(src_lat, src_lon)
            if not candidates: continue
            payload = {
                "city_id":state["city_id"],"hour":state["hour_bucket"],"dow":state["dow_bucket"],
                "is_weekend":state["is_weekend"],"dropoff_lat":0.0,"dropoff_lon":0.0,
                "points":[{"cell_id":src_cell,"lat":src_lat,"lon":src_lon}]
                       + [{"cell_id":c[0],"lat":c[1],"lon":c[2]} for c in candidates]}
            try:
                resp = await client.post("/predict/batch", json=payload)
                predictions = resp.json()["predictions"]
            except Exception as e:
                logger.warning(f"[M4] ML error {src_cell}: {e}"); continue
            src_price = predictions[0]["price_mean"]
            src_std   = predictions[0].get("price_std", 0)
            opps = []
            for cand, pred in zip(candidates, predictions[1:]):
                cell_id, clat, clon, dist_km, walk_min = cand
                gain_eur = src_price - pred["price_mean"]
                if gain_eur < cfg.min_gain_eur: continue
                gain_pct = gain_eur / src_price if src_price > 0 else 0
                if gain_pct < cfg.min_gain_pct: continue
                stability = await _get_stability(pool, src_cell, cell_id, state["city_id"],
                                                 state["hour_bucket"], state["dow_bucket"])
                uncertainty = (src_std + pred.get("price_std", 0)) / 2
                score = _score(gain_eur, walk_min, uncertainty, stability)
                if score <= 0: continue
                opps.append({"source":src_cell,"target":cell_id,"gain_eur":gain_eur,
                             "gain_pct":gain_pct,"walk_dist_m":dist_km*1000,"walk_min":walk_min,
                             "stability":stability,"confidence":max(0,1-uncertainty/max(src_price,1)),
                             "direction_deg":bearing_deg(src_lat,src_lon,clat,clon),"score":score})
            for opp in sorted(opps, key=lambda x:x["score"], reverse=True)[:5]:
                async with pool.acquire() as conn:
                    try:
                        await conn.execute("""
                            INSERT INTO price_opportunities (
                                h3_source_cell,h3_target_cell,resolution,city_id,
                                hour_bucket,dow_bucket,is_weekend,gain_median_eur,gain_pct,
                                walk_dist_m,stability_score,confidence,direction_deg,
                                valid_from,valid_until)
                            VALUES ($1,$2,8,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,NOW(),NOW()+INTERVAL '1 hour')
                            ON CONFLICT (h3_source_cell,h3_target_cell,hour_bucket,dow_bucket)
                            DO UPDATE SET gain_median_eur=EXCLUDED.gain_median_eur,
                                          stability_score=EXCLUDED.stability_score,
                                          confidence=EXCLUDED.confidence,valid_until=EXCLUDED.valid_until""",
                            opp["source"],opp["target"],state["city_id"],
                            state["hour_bucket"],state["dow_bucket"],state["is_weekend"],
                            opp["gain_eur"],opp["gain_pct"],opp["walk_dist_m"],
                            opp["stability"],opp["confidence"],opp["direction_deg"])
                        written += 1
                    except Exception as e:
                        logger.warning(f"[M4] insert error: {e}")
    state["opportunities_written"] = written
    logger.info(f"[M4] {written} opportunités écrites")
    return state

def build_m4_graph():
    g = StateGraph(M4State)
    g.add_node("fetch_cells", node_fetch_active_cells)
    g.add_node("compute_opps", node_compute_opportunities)
    g.set_entry_point("fetch_cells")
    g.add_edge("fetch_cells","compute_opps"); g.add_edge("compute_opps",END)
    return g.compile()

async def run_m4(city_id, hour=None, dow=None):
    now = datetime.now(timezone.utc)
    h = hour if hour is not None else now.hour
    d = dow  if dow  is not None else now.weekday()
    return await build_m4_graph().ainvoke({
        "city_id":city_id,"hour_bucket":h,"dow_bucket":d,"is_weekend":d>=5,
        "batch_size":500,"opportunities_written":0,"errors":[],"status":"running"})

if __name__ == "__main__":
    asyncio.run(run_m4("paris"))