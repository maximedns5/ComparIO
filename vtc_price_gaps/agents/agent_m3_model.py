"""Agent M3 — Entraînement LightGBM MAE + quantiles avec split temporel strict."""
from __future__ import annotations
import asyncio, logging, pickle
from datetime import datetime, timezone
from pathlib import Path
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
import lightgbm as lgb, numpy as np, pandas as pd
from shared.config import cfg
from shared.db import get_pg_pool

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
    "is_weekend","is_rush_am","is_rush_pm","is_night","minutes_since_midnight",
    "distance_km","bearing_sin","bearing_cos","delta_lat","delta_lon",
    "h3r8_price_median_7d","h3r8_price_std_7d","h3r8_trip_count_7d",
    "h3r8_duration_med_7d","h3r8_price_median_30d","h3r8_price_p10_30d",
    "h3r8_neighbor_mean","h3r8_price_vs_neighbors","h3r8_is_minimum",
]

class M3State(TypedDict):
    city_id: str; model_version: str
    train_mae: float | None; val_mae: float | None; test_mae: float | None
    n_train: int; n_val: int; n_test: int
    model_path: str | None; errors: list; status: str

async def node_load_dataset(state: M3State) -> M3State:
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT tf.*, tc.pickup_ts,
                hs7.price_median   AS h3r8_price_median_7d,
                hs7.price_std      AS h3r8_price_std_7d,
                hs7.trip_count     AS h3r8_trip_count_7d,
                hs7.duration_median AS h3r8_duration_med_7d,
                hs7.neighbor_price_mean AS h3r8_neighbor_mean,
                hs7.price_vs_neighbors  AS h3r8_price_vs_neighbors,
                hs7.is_price_minimum    AS h3r8_is_minimum,
                hs30.price_median  AS h3r8_price_median_30d,
                hs30.price_p10     AS h3r8_price_p10_30d
            FROM trip_features tf
            JOIN trips_clean tc ON tf.trip_id = tc.trip_id
            LEFT JOIN LATERAL (
                SELECT * FROM h3_price_stats WHERE h3_cell_id = tf.h3_pickup_r8
                  AND city_id = tc.city_id AND window_days = 7 AND computed_at < tc.pickup_ts
                ORDER BY computed_at DESC LIMIT 1) hs7 ON true
            LEFT JOIN LATERAL (
                SELECT * FROM h3_price_stats WHERE h3_cell_id = tf.h3_pickup_r8
                  AND city_id = tc.city_id AND window_days = 30 AND computed_at < tc.pickup_ts
                ORDER BY computed_at DESC LIMIT 1) hs30 ON true
            WHERE tc.city_id = $1 ORDER BY tc.pickup_ts ASC
        """, state["city_id"])
    if not rows: state["errors"].append("No data"); state["status"] = "error"; return state
    df = pd.DataFrame([dict(r) for r in rows]).dropna(subset=["price"])
    total = len(df); n_test = max(1, int(total*0.1)); n_val = max(1, int(total*0.1))
    n_train = total - n_test - n_val
    state["_df_train"] = df.iloc[:n_train]; state["_df_val"] = df.iloc[n_train:n_train+n_val]
    state["_df_test"] = df.iloc[n_train+n_val:]
    state["n_train"], state["n_val"], state["n_test"] = n_train, n_val, n_test
    logger.info(f"[M3] Dataset: train={n_train}, val={n_val}, test={n_test}")
    return state

async def node_train_model(state: M3State) -> M3State:
    if state.get("status") == "error": return state
    def prep(df): return df[FEATURE_COLS].fillna(0), df["price"]
    X_train, y_train = prep(state["_df_train"]); X_val, y_val = prep(state["_df_val"])
    td = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    vd = lgb.Dataset(X_val, label=y_val, reference=td)
    base = {"num_leaves":63,"min_child_samples":30,"feature_fraction":0.7,
            "bagging_fraction":0.8,"bagging_freq":5,"learning_rate":0.05,"verbose":-1,"n_jobs":-1}
    cb = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
    state["_model"]     = lgb.train({**base,"objective":"regression_l1"},td,valid_sets=[vd],callbacks=cb)
    state["_model_q10"] = lgb.train({**base,"objective":"quantile","alpha":0.1},td,valid_sets=[vd],callbacks=cb)
    state["_model_q90"] = lgb.train({**base,"objective":"quantile","alpha":0.9},td,valid_sets=[vd],callbacks=cb)
    val_pred = state["_model"].predict(X_val)
    state["val_mae"] = float(np.mean(np.abs(val_pred - y_val.values)))
    logger.info(f"[M3] Val MAE: {state['val_mae']:.3f}€")
    return state

async def node_evaluate_and_save(state: M3State) -> M3State:
    if state.get("status") == "error": return state
    def prep(df): return df[FEATURE_COLS].fillna(0), df["price"]
    X_train, y_train = prep(state["_df_train"]); X_test, y_test = prep(state["_df_test"])
    state["train_mae"] = float(np.mean(np.abs(state["_model"].predict(X_train) - y_train.values)))
    state["test_mae"]  = float(np.mean(np.abs(state["_model"].predict(X_test)  - y_test.values)))
    logger.info(f"[M3] Train MAE: {state['train_mae']:.3f}€  Test MAE: {state['test_mae']:.3f}€")
    model_dir = Path(cfg.model_path) / state["city_id"] / state["model_version"]
    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "model":state["_model"],"model_q10":state["_model_q10"],"model_q90":state["_model_q90"],
        "feature_cols":FEATURE_COLS,"train_mae":state["train_mae"],"val_mae":state["val_mae"],
        "test_mae":state["test_mae"],"city_id":state["city_id"],
        "model_version":state["model_version"],"trained_at":datetime.now(timezone.utc).isoformat()}
    with open(model_dir/"artifacts.pkl","wb") as f: pickle.dump(artifacts, f)
    state["_model"].save_model(str(model_dir/"model.lgb"))
    state["_model_q10"].save_model(str(model_dir/"model_q10.lgb"))
    state["_model_q90"].save_model(str(model_dir/"model_q90.lgb"))
    state["model_path"] = str(model_dir/"artifacts.pkl"); state["status"] = "done"
    return state

def build_m3_graph():
    g = StateGraph(M3State)
    g.add_node("load", node_load_dataset); g.add_node("train", node_train_model)
    g.add_node("evaluate", node_evaluate_and_save)
    g.set_entry_point("load"); g.add_edge("load","train"); g.add_edge("train","evaluate"); g.add_edge("evaluate",END)
    return g.compile()

async def run_m3(city_id):
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return await build_m3_graph().ainvoke({
        "city_id":city_id,"model_version":version,"train_mae":None,"val_mae":None,
        "test_mae":None,"n_train":0,"n_val":0,"n_test":0,"model_path":None,"errors":[],"status":"running"})

if __name__ == "__main__":
    asyncio.run(run_m3("paris"))