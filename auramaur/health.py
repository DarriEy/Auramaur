"""Container/native liveness probe without placing orders or claiming an IB client ID."""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import sys
from pathlib import Path

from auramaur.killswitch import kill_switch_present
from auramaur.runtime import db_path


def _database_health(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "database missing"
    try:
        uri = f"file:{path.resolve()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=5) as conn:
            result = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        return result is not None, f"schema version {result[0]}" if result else "no schema version"
    except sqlite3.Error as exc:
        return False, str(exc)


def _ibkr_socket_health() -> tuple[bool, str]:
    host = os.environ.get("AURAMAUR_IBKR_HOST", "ibgateway")
    environment = os.environ.get("AURAMAUR_IBKR_ENVIRONMENT", "paper")
    default_port = "4002" if environment == "paper" else "4001"
    port = int(os.environ.get("AURAMAUR_IBKR_PORT", default_port))
    try:
        with socket.create_connection((host, port), timeout=3):
            return True, f"{host}:{port} reachable"
    except OSError as exc:
        return False, str(exc)


def main() -> int:
    db_ok, db_detail = _database_health(db_path())
    ib_ok, ib_detail = _ibkr_socket_health()
    require_ibkr = os.environ.get("AURAMAUR_HEALTH_REQUIRE_IBKR", "false").lower() == "true"
    payload = {
        "database": {"ok": db_ok, "detail": db_detail},
        "ibkr_socket": {"ok": ib_ok, "detail": ib_detail},
        "kill_switch": kill_switch_present(),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if db_ok and (ib_ok or not require_ibkr) else 1


if __name__ == "__main__":
    sys.exit(main())
