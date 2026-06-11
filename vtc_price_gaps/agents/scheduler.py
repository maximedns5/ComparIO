"""
Scheduler — Orchestration LangGraph de tous les agents.
Un cycle complet (M1 → M2 → retrain si besoin → M4) toutes les heures.
M3 (entraînement) ne s'exécute qu'une fois par jour (cycle % 24 == 0).
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from typing import List
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from agent_m1_ingestion     import run_m1
from agent_m2_features      import run_m2
from agent_m3_model         import run_m3
from agent_m4_opportunities import run_m4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Ajouter vos villes ici ──────────────────────────────────────
CITIES_CONFIG = [
    {"city_id": "paris",  "source_table": "trips_raw_paris"},
    # {"city_id": "london", "source_table": "trips_raw_london"},
]


class PipelineState(TypedDict):
    cities: List[str]
    phase: str
    cycle: int
    errors: List[str]
    last_run: dict


async def node_ingest_all(state: PipelineState) -> PipelineState:
    logger.info(f"[Scheduler] Cycle {state['cycle']} — INGESTION")
    results = await asyncio.gather(
        *[run_m1(c["city_id"], c["source_table"]) for c in CITIES_CONFIG],
        return_exceptions=True,
    )
    for c, r in zip(CITIES_CONFIG, results):
        if isinstance(r, Exception):
            state["errors"].append(f"M1 {c['city_id']}: {r}")
            logger.error(f"[M1] Erreur {c['city_id']}: {r}")
        else:
            logger.info(f"[M1] {c['city_id']} — {r.get('rows_written', 0)} trajets écrits")
    return state


async def node_compute_features_all(state: PipelineState) -> PipelineState:
    logger.info(f"[Scheduler] Cycle {state['cycle']} — FEATURES")
    tasks = []
    for c in CITIES_CONFIG:
        tasks.append(run_m2(c["city_id"], window_days=7))
        tasks.append(run_m2(c["city_id"], window_days=30))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            state["errors"].append(f"M2: {r}")
            logger.error(f"[M2] Erreur: {r}")
    return state


async def node_retrain_if_needed(state: PipelineState) -> PipelineState:
    """Entraîne uniquement une fois par jour (cycle multiple de 24)."""
    if state["cycle"] % 24 != 0:
        return state
    logger.info(f"[Scheduler] Cycle {state['cycle']} — TRAINING (journalière)")
    for c in CITIES_CONFIG:
        try:
            result = await run_m3(c["city_id"])
            logger.info(
                f"[M3] {c['city_id']} — "
                f"Train MAE: {result.get('train_mae','N/A')}€  "
                f"Test MAE: {result.get('test_mae','N/A')}€"
            )
        except Exception as e:
            state["errors"].append(f"M3 {c['city_id']}: {e}")
            logger.error(f"[M3] Erreur {c['city_id']}: {e}")
    return state


async def node_refresh_opportunities(state: PipelineState) -> PipelineState:
    now = datetime.now(timezone.utc)
    logger.info(f"[Scheduler] Cycle {state['cycle']} — OPPORTUNITIES h={now.hour}")
    results = await asyncio.gather(
        *[run_m4(c["city_id"], hour=now.hour, dow=now.weekday()) for c in CITIES_CONFIG],
        return_exceptions=True,
    )
    for c, r in zip(CITIES_CONFIG, results):
        if isinstance(r, Exception):
            state["errors"].append(f"M4 {c['city_id']}: {r}")
            logger.error(f"[M4] Erreur {c['city_id']}: {r}")
        else:
            logger.info(f"[M4] {c['city_id']} — {r.get('opportunities_written', 0)} opportunités")
    state["cycle"] += 1
    state["last_run"] = {c["city_id"]: now.isoformat() for c in CITIES_CONFIG}
    return state


def build_pipeline_graph():
    g = StateGraph(PipelineState)
    g.add_node("ingest",        node_ingest_all)
    g.add_node("features",      node_compute_features_all)
    g.add_node("retrain",       node_retrain_if_needed)
    g.add_node("opportunities", node_refresh_opportunities)
    g.set_entry_point("ingest")
    g.add_edge("ingest",        "features")
    g.add_edge("features",      "retrain")
    g.add_edge("retrain",       "opportunities")
    g.add_edge("opportunities", END)
    return g.compile()


async def main():
    graph = build_pipeline_graph()
    state: PipelineState = {
        "cities": [c["city_id"] for c in CITIES_CONFIG],
        "phase":  "ingestion",
        "cycle":  0,
        "errors": [],
        "last_run": {},
    }
    logger.info("[Scheduler] Démarrage du pipeline VTC Price Gaps")

    # Premier cycle immédiat incluant l'entraînement initial (cycle 0)
    while True:
        try:
            state = await graph.ainvoke(state)
            if state["errors"]:
                logger.warning(f"[Scheduler] {len(state['errors'])} erreur(s) ce cycle")
                state["errors"] = []
        except Exception as e:
            logger.error(f"[Scheduler] Erreur critique cycle {state['cycle']}: {e}")

        logger.info(f"[Scheduler] Cycle {state['cycle']} terminé — attente 1h")
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
