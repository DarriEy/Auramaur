#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

if [ -f runtime/state/KILL_SWITCH ]; then
    echo "FAIL: runtime/state/KILL_SWITCH is present" >&2
    exit 1
fi

running=$(docker compose ps --status running --services | sort)
echo "$running" | grep -qx auramaur || { echo "FAIL: Auramaur is not running" >&2; exit 1; }
echo "$running" | grep -qx ibgateway || { echo "FAIL: IB Gateway is not running" >&2; exit 1; }

count=$(docker compose ps --status running --services | grep -cx auramaur)
[ "$count" -eq 1 ] || { echo "FAIL: expected one Auramaur service" >&2; exit 1; }

docker compose exec -T auramaur python -m auramaur.health
integrity=$(docker compose exec -T auramaur sqlite3 /app/state/auramaur.db 'PRAGMA quick_check;')
[ "$integrity" = "ok" ] || { echo "FAIL: database quick_check: $integrity" >&2; exit 1; }
docker compose exec -T ibgateway /usr/local/bin/healthcheck.sh

echo "Verify OK: one Auramaur service, primary DB healthy, IB Gateway API reachable."
echo "Compare account IDs, cash, positions, and paper/live mode against each venue UI."
