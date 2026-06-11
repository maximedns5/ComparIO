"""Agent M1 — Ingestion & Nettoyage des données brutes."""
from __future__ import annotations
import asyncio, logging, math
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from shared.config import cfg
from shared.db import get_pg_pool
from shared.geo_utils import haversine_km

logger = logging.getLogger(__name__)

class M1State(TypedDict):
    city_id: str; source_table: str; batch_size: int
    last_processed_ts: str | None; rows_read: int
    rows_written: int; rows_rejected: int; errors: list; status: str

CITY_BOUNDS = {
    "paris":    {"lat": (48.6, 49.1), "lon": (2.1, 2.7),   "max_dist_km": 80},
    "london":   {"lat": (51.3, 51.7), "lon": (-0.5, 0.3),  "max_dist_km": 80},
    "new_york": {"lat": (40.4, 41.0), "lon": (-74.3, -73.6),"max_dist_km": 60},
    "_default": {"lat": (-90, 90),    "lon": (-180, 180),   "max_dist_km": 100},
}

def _validate_row(row, bounds):
    lat1, lon1 = row.get("pickup_lat"), row.get("pickup_lon")
    lat2, lon2 = row.get("dropoff_lat"), row.get("dropoff_lon")
    price = row.get("price")
    if any(v is None for v in [lat1, lon1, lat2, lon2, price]):
        return False, "null"
    lmin, lmax = bounds["lat"]; omin, omax = bounds["lon"]
    if not (lmin <= lat1 <= lmax and omin <= lon1 <= omax): return False, "pickup_bounds"
    if not (lmin <= lat2 <= lmax and omin <= lon2 <= omax): return False, "dropoff_bounds"
    dist = haversine_km(lat1, lon1, lat2, lon2)
    if dist < 0.1: return False, "too_short"
    if dist > bounds["max_dist_km"]: return False, "too_long"
    if price <= 0 or price > 500: return False, "price_range"
    dur = row.get("trip_duration_s")
    if dur and dur > 0:
        spd = (dist / dur) * 3600
        if spd < 1 or spd > 150: return False, f"speed_{spd:.0f}"
    return True, ""

async def node_fetch_batch(state: M1State) -> M1State:
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        where = f"WHERE pickup_ts > '{state['last_processed_ts']}'" if state["last_processed_ts"] else ""
        rows = await conn.fetch(
            f"SELECT * FROM {state['source_table']} {where} ORDER BY pickup_ts ASC LIMIT {state['batch_size']}"
        )
    state["_raw_rows"] = [dict(r) for r in rows]
    state["rows_read"] = len(rows)
    return state

async def node_validate_and_enrich(state: M1State) -> M1State:
    bounds = CITY_BOUNDS.get(state["city_id"], CITY_BOUNDS["_default"])
    clean, rejected = [], 0
    for row in state.get("_raw_rows", []):
        valid, _ = _validate_row(row, bounds)
        if not valid: rejected += 1; continue
        lat1, lon1, lat2, lon2 = row["pickup_lat"], row["pickup_lon"], row["dropoff_lat"], row["dropoff_lon"]
        clean.append({**row, "distance_km": round(haversine_km(lat1, lon1, lat2, lon2), 4),
                      "city_id": state["city_id"], "data_quality_score": 1.0,
                      "pickup_wkt": f"POINT({lon1} {lat1})", "dropoff_wkt": f"POINT({lon2} {lat2})"})
    state["_clean_rows"] = clean; state["rows_rejected"] = rejected
    return state

async def node_write_clean(state: M1State) -> M1State:
    rows = state.get("_clean_rows", [])
    if not rows: state["rows_written"] = 0; return state
    pool = await get_pg_pool(); written = 0
    async with pool.acquire() as conn:
        async with conn.transaction():
            for row in rows:
                try:
                    await conn.execute("""
                        INSERT INTO trips_clean (
                            trip_id,city_id,provider,pickup_lat,pickup_lon,
                            dropoff_lat,dropoff_lon,pickup_geom,dropoff_geom,
                            pickup_ts,trip_duration_s,distance_km,price,is_special_day,data_quality_score
                        ) VALUES ($1,$2,$3,$4,$5,$6,$7,
                            ST_GeomFromText($8,4326),ST_GeomFromText($9,4326),
                            $10,$11,$12,$13,$14,$15)
                        ON CONFLICT (trip_id) DO NOTHING""",
                        row["trip_id"], row["city_id"], row.get("provider","unknown"),
                        row["pickup_lat"], row["pickup_lon"], row["dropoff_lat"], row["dropoff_lon"],
                        row["pickup_wkt"], row["dropoff_wkt"], row["pickup_ts"],
                        row.get("trip_duration_s"), row["distance_km"], row["price"],
                        row.get("is_special_day", False), row["data_quality_score"])
                    written += 1
                except Exception as e:
                    logger.warning(f"[M1] insert error {row.get('trip_id')}: {e}")
    if rows:
        last = max(r["pickup_ts"] for r in rows)
        state["last_processed_ts"] = last.isoformat() if hasattr(last, "isoformat") else str(last)
    state["rows_written"] = written
    logger.info(f"[M1] {state['city_id']} — {written} écrits, {state['rows_rejected']} rejetés")
    return state

def node_decide_continue(state): return "fetch" if state["rows_read"] >= state["batch_size"] else END

def build_m1_graph():
    g = StateGraph(M1State)
    g.add_node("fetch", node_fetch_batch)
    g.add_node("validate", node_validate_and_enrich)
    g.add_node("write", node_write_clean)
    g.set_entry_point("fetch")
    g.add_edge("fetch", "validate"); g.add_edge("validate", "write")
    g.add_conditional_edges("write", node_decide_continue, {"fetch": "fetch", END: END})
    return g.compile()

async def run_m1(city_id, source_table, batch_size=1000):
    return await build_m1_graph().ainvoke({
        "city_id": city_id, "source_table": source_table, "batch_size": batch_size,
        "last_processed_ts": None, "rows_read": 0, "rows_written": 0,
        "rows_rejected": 0, "errors": [], "status": "running"})

if __name__ == "__main__":
    asyncio.run(run_m1("paris", "trips_raw_paris"))