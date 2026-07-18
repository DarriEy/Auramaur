#!/bin/sh
# Portable cutover helper for native macOS, WSL/Linux, and Compose hosts.
set -eu

cd "$(dirname "$0")/.."

fail() { echo "FAIL: $*" >&2; exit 1; }

case "${1:-}" in
  stage)
    command -v docker >/dev/null || fail "docker missing"
    docker compose version >/dev/null || fail "Docker Compose plugin missing"
    [ ! -f .env ] || fail "stage must run without live .env credentials"
    [ -f deploy/ibgateway/ibgateway-latest-standalone-linux-x64.sh ] \
      || fail "IB Gateway offline installer missing; see deploy/ibgateway/README.md"
    deploy/init-runtime.sh
    docker compose config >/dev/null
    echo "Stage OK. Build and run the stack in paper mode; authenticate IB Gateway paper."
    ;;
  freeze)
    mkdir -p runtime/state
    touch runtime/state/KILL_SWITCH KILL_SWITCH
    if command -v docker >/dev/null && docker compose ps --services 2>/dev/null | grep -qx auramaur; then
      docker compose stop -t 120 auramaur
    elif pgrep -f 'auramaur run' >/dev/null; then
      pkill -TERM -f 'auramaur run'
      i=0
      while pgrep -f 'auramaur run' >/dev/null && [ "$i" -lt 60 ]; do sleep 2; i=$((i + 1)); done
    fi
    pgrep -f 'auramaur run' >/dev/null && fail "Auramaur process still running"
    echo "FROZEN. Leave both kill switches in place on this host."
    ;;
  export)
    [ -f KILL_SWITCH ] || [ -f runtime/state/KILL_SWITCH ] \
      || fail "freeze this host before exporting"
    pgrep -f 'auramaur run' >/dev/null && fail "Auramaur process still running"
    mkdir -p runtime/state/backups
    if [ -f runtime/state/auramaur.db ]; then
      AURAMAUR_DB_PATH=runtime/state/auramaur.db deploy/backup.sh
    elif [ -f auramaur.db ]; then
      AURAMAUR_DB_PATH=auramaur.db AURAMAUR_BACKUP_DIR=runtime/state/backups deploy/backup.sh
    else
      fail "no Auramaur database found"
    fi
    echo "Export ready. Transfer the newest verified backup plus .env, secrets/,"
    echo "runtime/config/, runtime/ibgateway/, and Claude credentials separately."
    ;;
  verify)
    deploy/verify.sh
    ;;
  *)
    echo "usage: scripts/migrate_portable.sh stage|freeze|export|verify" >&2
    exit 1
    ;;
esac
