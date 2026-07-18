#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

check() {
    docker compose ps --status running --services | grep -qx auramaur
    docker compose ps --status running --services | grep -qx ibgateway
    docker compose exec -T -e AURAMAUR_HEALTH_REQUIRE_IBKR=true \
        auramaur python -m auramaur.health
    docker compose exec -T ibgateway /usr/local/bin/healthcheck.sh
}

if output=$(check 2>&1); then
    echo "$output"
    exit 0
fi

message="Auramaur deployment unhealthy on $(hostname): $output"
echo "$message" >&2
if [ -n "${AURAMAUR_MONITOR_WEBHOOK_URL:-}" ]; then
    payload=$(printf '%s' "$message" | python3 -c \
        'import json,sys; print(json.dumps({"content": sys.stdin.read()}))')
    curl --fail --silent --show-error -X POST \
        -H 'Content-Type: application/json' \
        --data "$payload" \
        "$AURAMAUR_MONITOR_WEBHOOK_URL" >/dev/null
fi
exit 1
