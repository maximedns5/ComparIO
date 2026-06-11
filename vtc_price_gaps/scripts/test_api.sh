#!/bin/bash
# Test rapide de l'API opportunities
# Usage: bash scripts/test_api.sh
set -euo pipefail

echo "=== Health checks ==="
echo -n "API Gateway : "; curl -sf http://localhost:8000/health | python3 -m json.tool
echo -n "ML Service  : "; curl -sf http://localhost:8001/health | python3 -m json.tool

echo ""
echo "=== Test opportunities (Paris) ==="
curl -s -X POST http://localhost:8000/opportunities \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_lat": 48.8566,
    "pickup_lon": 2.3522,
    "dropoff_lat": 48.8738,
    "dropoff_lon": 2.2950,
    "city_id": "paris",
    "max_walk_minutes": 6
  }' | python3 -m json.tool
