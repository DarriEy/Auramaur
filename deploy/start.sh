#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
export AURAMAUR_UID=${AURAMAUR_UID:-$(id -u)}
export AURAMAUR_GID=${AURAMAUR_GID:-$(id -g)}
if [ -f runtime/state/KILL_SWITCH ]; then
    echo "KILL_SWITCH present; refusing to start." >&2
    exit 1
fi
docker compose up -d --remove-orphans
docker compose ps
