#!/bin/bash
# =================================================================
# VTC Price Gaps — Script de setup complet
# Usage: bash scripts/setup.sh
# =================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo ""
echo "=========================================="
echo "  VTC Price Gaps — Setup"
echo "=========================================="
echo ""

# ─── Vérifications prérequis ──────────────────
command -v docker  &>/dev/null || error "Docker non trouvé. Installez Docker: https://docs.docker.com/get-docker/"
command -v docker compose &>/dev/null || \
    docker compose version &>/dev/null   || \
    error "Docker Compose non trouvé."

info "Docker OK: $(docker --version)"

# ─── Environnement ────────────────────────────
if [ ! -f .env ]; then
    cp .env.example .env
    warn ".env créé depuis .env.example"
    warn ">>> Pensez à changer POSTGRES_PASSWORD avant de lancer en production !"
else
    info ".env déjà présent"
fi

# ─── Dossiers ─────────────────────────────────
mkdir -p ml_service/models
info "Dossiers créés"

# ─── Build des images ─────────────────────────
info "Build des images Docker (peut prendre 5-10 min pour h3-pg)..."
docker compose build

# ─── Démarrage de la DB et Redis seuls d'abord ─
info "Démarrage Postgres + Redis..."
docker compose up -d db redis

# ─── Attente Postgres ─────────────────────────
info "Attente que Postgres soit prêt..."
MAX_WAIT=60; WAITED=0
until docker compose exec db pg_isready -U vtc -d vtc_db &>/dev/null; do
    sleep 2; WAITED=$((WAITED+2))
    [ $WAITED -ge $MAX_WAIT ] && error "Postgres n'a pas démarré après ${MAX_WAIT}s"
done
info "Postgres prêt !"

# ─── Démarrage de tous les services ───────────
info "Démarrage de tous les services..."
docker compose up -d

# ─── Attente ML Service ───────────────────────
info "Attente que le ML Service soit prêt..."
MAX_WAIT=30; WAITED=0
until curl -sf http://localhost:8001/health &>/dev/null; do
    sleep 2; WAITED=$((WAITED+2))
    [ $WAITED -ge $MAX_WAIT ] && warn "ML Service pas encore prêt (normal si premier démarrage)"
    break
done

echo ""
echo "=========================================="
echo "  ✅ Infrastructure démarrée !"
echo "=========================================="
echo ""
echo "  Services:"
echo "    API Gateway : http://localhost:8000"
echo "    ML Service  : http://localhost:8001"
echo "    Docs API    : http://localhost:8000/docs"
echo "    Postgres    : localhost:5432"
echo "    Redis       : localhost:6379"
echo ""
echo "  Prochaines étapes:"
echo "    1. Insérer vos données brutes dans trips_raw_<ville>"
echo "    2. Entraîner le premier modèle:"
echo "       docker compose exec agents python -c \"
echo "         \"import asyncio; from agent_m3_model import run_m3; asyncio.run(run_m3('paris'))\""
echo "    3. Tester l'API:"
echo "       curl -X POST http://localhost:8000/opportunities \\"
echo "         -H 'Content-Type: application/json' \\"
echo "         -d '{"pickup_lat":48.8566,"pickup_lon":2.3522,\"
echo "              "dropoff_lat":48.8738,"dropoff_lon":2.2950,\"
echo "              "city_id":"paris","max_walk_minutes":6}'"
echo ""
echo "  Logs: docker compose logs -f"
echo "  Stop: docker compose down"
echo ""
