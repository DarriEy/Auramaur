#!/usr/bin/env bash
# Restart the live Auramaur bot safely.
#
# Exists because the naive relaunch has two silent traps (both hit 2026-07-02):
#   1. Launching without stopping the old process runs TWO bots at once.
#   2. If the old process still holds the auramaur.db flock when the new one
#      starts, the new bot silently falls back to a stale numbered instance
#      DB (auramaur_2.db, ...) — right code, wrong book.
# This script stops every running bot first, waits for the lock holder to
# exit, launches exactly one instance, and FAILS LOUDLY if the new bot came
# up on a fallback database.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f ./KILL_SWITCH ]; then
    echo "KILL_SWITCH present — refusing to start. Remove it first." >&2
    exit 1
fi

# 1. Stop every running bot instance gracefully (SIGTERM -> shutdown path
#    cancels resting orders and releases the DB flock).
pids=$(pgrep -f 'auramaur run' || true)
if [ -n "$pids" ]; then
    echo "Stopping running bot instance(s): $pids"
    kill -TERM $pids 2>/dev/null || true
    for _ in $(seq 1 30); do
        pgrep -f 'auramaur run' > /dev/null || break
        sleep 1
    done
    if pgrep -f 'auramaur run' > /dev/null; then
        echo "Old instance did not exit within 30s — refusing to force-kill" >&2
        echo "a live trading process. Inspect it, then re-run." >&2
        exit 1
    fi
fi

# 2. Launch one instance, live-armed (still behind the config + per-order gates).
ts=$(date +%Y%m%d_%H%M%S)
log="logs/run_live_${ts}.out"
AURAMAUR_LIVE=true nohup .venv/bin/auramaur run --hybrid > "$log" 2>&1 &
pid=$!
echo "Launched PID ${pid}, log: ${log}"

# 3. Verify it came up on the PRIMARY database. The banner prints an
#    "Instance: auramaur_N.db" line only on lock-contention fallback.
for _ in $(seq 1 60); do
    grep -q 'strategy books' "$log" 2>/dev/null && break
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "Bot exited during startup — tail of ${log}:" >&2
        tail -20 "$log" >&2
        exit 1
    fi
    sleep 2
done
if grep -q 'Instance: auramaur_' "$log"; then
    echo "FALLBACK DB DETECTED — the primary auramaur.db was still locked." >&2
    echo "Stopping the wrong-book instance. Re-run once the lock holder exits." >&2
    kill -TERM "$pid"
    exit 1
fi
grep -E 'Build:|Mode:' "$log" | head -2
echo "OK: single instance on the primary database."
