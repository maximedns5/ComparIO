"""Agent M2 — Feature Store: features par trajet + stats H3 avec gradient voisins."""
from __future__ import annotations
import asyncio, logging, math
from datetime import datetime, timezone
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
import h3, numpy as np
from shared.config import cfg
from shared.db import get_pg_pool
from shared.geo_utils import haversine_km, bearing_deg, latlng_to_cells, compute_temporal_features

logger = logging.getLogger(__name__)

class M2State(TypedDict):
    city_id: str; batch_size: int; window_days: int
    last_trip_ts: str | None; features_written: int; h3_stats_written: int
    errors: list; status: str

async def node_compute_trip_features(state: M2State) -> M2State:
    pool = await get_pg_pool(); written = 0
    async with pool.acquire() as conn:
        where = f"AND t.pickup_ts > '{state['last_trip_ts']}'" if state["last_trip_ts"] else ""
        rows = await conn.fetch(f"""
            SELECT t.* FROM trips_clean t
            LEFT JOIN trip_features tf ON t.trip_id = tf.trip_id
            WHERE t.city_id = $1 AND tf.trip_id IS NULL {where}
            ORDER BY t.pickup_ts ASC LIMIT $2
        """, state["city_id"], state["batch_size"])
        for row in rows:
            row = dict(row); ts = row["pickup_ts"]
            lat1, lon1 = row["pickup_lat"], row["pickup_lon"]
            lat2, lon2 = row["dropoff_lat"], row["dropoff_lon"]
            tf = compute_temporal_features(ts)
            dist = row["distance_km"] or haversine_km(lat1, lon1, lat2, lon2)
            b = bearing_deg(lat1, lon1, lat2, lon2)
            pc = latlng_to_cells(lat1, lon1); dc = latlng_to_cells(lat2, lon2)
            try:
                await conn.execute("""
                    INSERT INTO trip_features (
                        trip_id, hour_sin,hour_cos,dow_sin,dow_cos,month_sin,month_cos,
                        is_weekend,is_rush_am,is_rush_pm,is_night,minutes_since_midnight,
                        distance_km,bearing_sin,bearing_cos,delta_lat,delta_lon,
                        h3_pickup_r6,h3_pickup_r7,h3_pickup_r8,h3_pickup_r9,
                        h3_dropoff_r7,h3_dropoff_r8, price
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,
                              $13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24)
                    ON CONFLICT (trip_id) DO NOTHING""",
                    row["trip_id"],
                    tf["hour_sin"],tf["hour_cos"],tf["dow_sin"],tf["dow_cos"],
                    tf["month_sin"],tf["month_cos"],tf["is_weekend"],
                    tf["is_rush_am"],tf["is_rush_pm"],tf["is_night"],
                    tf["minutes_since_midnight"], dist,
                    math.sin(math.radians(b)), math.cos(math.radians(b)),
                    lat2-lat1, lon2-lon1,
                    pc[6],pc[7],pc[8],pc[9], dc[7],dc[8], row["price"])
                written += 1
            except Exception as e:
                logger.warning(f"[M2] feature error {row.get('trip_id')}: {e}")
    state["features_written"] = written
    if rows:
        last = max(dict(r)["pickup_ts"] for r in rows)
        state["last_trip_ts"] = last.isoformat() if hasattr(last, "isoformat") else str(last)
    return state

async def node_compute_h3_stats(state: M2State) -> M2State:
    pool = await get_pg_pool(); window = state["window_days"]; written = 0
    async with pool.acquire() as conn:
        cells = await conn.fetch("""
            SELECT DISTINCT tf.h3_pickup_r8 AS cell_id
            FROM trip_features tf JOIN trips_clean tc ON tf.trip_id = tc.trip_id
            WHERE tc.city_id = $1 AND tc.pickup_ts >= NOW() - ($2 || ' days')::INTERVAL
              AND tf.h3_pickup_r8 IS NOT NULL
        """, state["city_id"], str(window))
        if not cells: state["h3_stats_written"] = 0; return state
        cell_stats = {}; now = datetime.now(timezone.utc)
        for cell_row in cells:
            cell_id = cell_row["cell_id"]
            stats = await conn.fetchrow("""
                SELECT
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tf.price) AS price_median,
                    AVG(tf.price) AS price_mean, STDDEV(tf.price) AS price_std,
                    PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY tf.price) AS price_p10,
                    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY tf.price) AS price_p90,
                    COUNT(*) AS trip_count,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tc.trip_duration_s) AS duration_median
                FROM trip_features tf JOIN trips_clean tc ON tf.trip_id = tc.trip_id
                WHERE tf.h3_pickup_r8 = $1 AND tc.city_id = $2
                  AND tc.pickup_ts >= NOW() - ($3 || ' days')::INTERVAL
            """, cell_id, state["city_id"], str(window))
            if stats and stats["trip_count"] and stats["trip_count"] >= cfg.min_trip_count:
                cell_stats[cell_id] = dict(stats)
        for cell_id, stats in cell_stats.items():
            neighbors = list(h3.grid_ring(cell_id, 1))
            neighbor_medians = [cell_stats[n]["price_median"] for n in neighbors
                                if n in cell_stats and cell_stats[n]["price_median"]]
            neighbor_mean = float(np.mean(neighbor_medians)) if neighbor_medians else None
            price_vs_neighbors = None; is_minimum = False
            if neighbor_mean and stats["price_median"]:
                price_vs_neighbors = stats["price_median"] - neighbor_mean
                neighbor_prices = [cell_stats.get(n,{}).get("price_median", float("inf"))
                                   for n in neighbors if n in cell_stats]
                is_minimum = bool(neighbor_prices and stats["price_median"] < min(neighbor_prices))
            try:
                await conn.execute("""
                    INSERT INTO h3_price_stats (
                        h3_cell_id,resolution,city_id,window_days,computed_at,
                        price_median,price_mean,price_std,price_p10,price_p90,
                        trip_count,duration_median,neighbor_price_mean,price_vs_neighbors,is_price_minimum
                    ) VALUES ($1,8,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                    ON CONFLICT (h3_cell_id,resolution,window_days,computed_at) DO UPDATE SET
                        price_median=EXCLUDED.price_median, trip_count=EXCLUDED.trip_count,
                        neighbor_price_mean=EXCLUDED.neighbor_price_mean,
                        price_vs_neighbors=EXCLUDED.price_vs_neighbors,
                        is_price_minimum=EXCLUDED.is_price_minimum""",
                    cell_id, state["city_id"], window, now,
                    stats.get("price_median"), stats.get("price_mean"), stats.get("price_std"),
                    stats.get("price_p10"), stats.get("price_p90"), stats.get("trip_count"),
                    stats.get("duration_median"), neighbor_mean, price_vs_neighbors, is_minimum)
                written += 1
            except Exception as e:
                logger.warning(f"[M2] h3_stats error {cell_id}: {e}")
    state["h3_stats_written"] = written
    logger.info(f"[M2] {state['city_id']} w={window}j — {written} cellules H3")
    return state

def node_check_done(state): return "features" if state["features_written"] >= state["batch_size"] else END

def build_m2_graph():
    g = StateGraph(M2State)
    g.add_node("features", node_compute_trip_features)
    g.add_node("h3_stats", node_compute_h3_stats)
    g.set_entry_point("features")
    g.add_conditional_edges("features", node_check_done, {"features": "features", END: "h3_stats"})
    g.add_edge("h3_stats", END)
    return g.compile()

async def run_m2(city_id, window_days=7, batch_size=2000):
    return await build_m2_graph().ainvoke({
        "city_id": city_id, "batch_size": batch_size, "window_days": window_days,
        "last_trip_ts": None, "features_written": 0, "h3_stats_written": 0,
        "errors": [], "status": "running"})

if __name__ == "__main__":
    asyncio.run(run_m2("paris"))