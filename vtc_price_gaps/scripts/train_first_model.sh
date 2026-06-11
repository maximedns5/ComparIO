#!/bin/bash
# Lance l'entraînement du premier modèle pour toutes les villes configurées
# Usage: bash scripts/train_first_model.sh [city_id]
set -euo pipefail

CITY=${1:-paris}
echo "[Train] Lancement de l'entraînement pour la ville: $CITY"

docker compose exec agents python -c "
import asyncio
from agent_m3_model import run_m3

async def main():
    result = await run_m3('$CITY')
    print(f'\n=== Résultats entraînement $CITY ===')
    print(f'  Train MAE : {result.get(\"train_mae\", \"N/A\")}€')
    print(f'  Val MAE   : {result.get(\"val_mae\", \"N/A\")}€')
    print(f'  Test MAE  : {result.get(\"test_mae\", \"N/A\")}€')
    print(f'  Modèle    : {result.get(\"model_path\", \"N/A\")}')
    print(f'  Statut    : {result.get(\"status\", \"N/A\")}')
    if result.get('errors'):
        print(f'  Erreurs   : {result[\"errors\"]}')

asyncio.run(main())
"
echo "[Train] Rechargement du modèle dans le ML Service..."
curl -sf -X POST http://localhost:8001/reload/$CITY | python3 -m json.tool || true
echo "[Train] Terminé !"
