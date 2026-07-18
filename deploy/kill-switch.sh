#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
mkdir -p runtime/state
touch runtime/state/KILL_SWITCH
docker compose stop -t 120 auramaur
echo "KILL_SWITCH armed and Auramaur stopped. IB Gateway remains available for inspection."
