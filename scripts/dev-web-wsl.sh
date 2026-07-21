#!/bin/bash
# Dev launcher: run the web dashboard backend inside WSL, colocated with the
# deployed bot's SQLite state (opened read-only by the app), serving the code
# in this working tree. Pair with `cd web && npm run dev` on the Windows side.
#
# Why WSL and not the Windows tree directly: SQLite WAL reads over the \\wsl$
# 9P boundary have unreliable locking against the live writer. The reader must
# sit on the same filesystem as the DB — same topology as the compose service.
#
# Usage: bash scripts/dev-web-wsl.sh   (from WSL, or via `wsl -e bash ...`)
set -eu

DEPLOY_ROOT="${AURAMAUR_DEPLOY_ROOT:-$HOME/repos/Auramaur}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${AURAMAUR_WEB_VENV:-$HOME/.venvs/auramaur-web}"

if [ ! -x "$VENV/bin/python" ]; then
    echo "creating venv at $VENV"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q fastapi "uvicorn>=0.30" aiosqlite structlog rich \
        pydantic pydantic-settings pyyaml python-dotenv
fi

export AURAMAUR_DB_PATH="$DEPLOY_ROOT/runtime/state/auramaur.db"
export LOGGING__FILE="$DEPLOY_ROOT/runtime/logs/auramaur.log"
export AURAMAUR_KILL_SWITCH_PATH="$DEPLOY_ROOT/runtime/state/KILL_SWITCH"
export AURAMAUR_LOCAL_CONFIG="$DEPLOY_ROOT/runtime/config/defaults.local.yaml"
export PYTHONPATH="$REPO_ROOT"

# Mirror the deployment's live gate (same variable compose feeds the bot) so
# the dashboard's BOT:LIVE/PAPER badge tells the truth. View-only: this
# process opens the DB read-only and can place no orders regardless.
AURAMAUR_LIVE="$(grep -oE '^AURAMAUR_CONTAINER_LIVE=.*' "$DEPLOY_ROOT/.env" 2>/dev/null | cut -d= -f2 || true)"
export AURAMAUR_LIVE="${AURAMAUR_LIVE:-false}"

cd "$HOME"  # neutral cwd: never pick up a stray .env from a repo tree
exec "$VENV/bin/python" - <<'PY'
import uvicorn
from auramaur.web.app import create_app
uvicorn.run(create_app(), host="127.0.0.1", port=8484, log_level="warning")
PY
