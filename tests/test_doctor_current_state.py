from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from auramaur.db.database import Database
from auramaur.monitoring.diagnostics import gather_doctor


def _settings(log_file):
    return SimpleNamespace(
        is_live=False,
        kill_switch_active=False,
        logging=SimpleNamespace(file=str(log_file)),
        monitoring=SimpleNamespace(doctor_health_window_seconds=1800),
    )


@pytest.mark.asyncio
async def test_doctor_ignores_old_log_debt_and_latest_provider_success_wins(
        tmp_path):
    now = datetime.now(timezone.utc)
    log_file = tmp_path / "auramaur.log"
    rows = [
        {
            "event": "historical.failure",
            "level": "error",
            "timestamp": (now - timedelta(hours=2)).isoformat(),
        },
        {
            "event": "kalshi.scan",
            "level": "info",
            "timestamp": now.isoformat(),
        },
    ]
    log_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    db = Database(":memory:")
    await db.connect()
    try:
        await db.execute(
            """INSERT INTO source_fetches
               (run_id, source, status, observed_at)
               VALUES ('old', 'newsapi', 'timeout',
                       datetime('now', '-10 minutes'))"""
        )
        await db.execute(
            """INSERT INTO source_fetches
               (run_id, source, status, observed_at)
               VALUES ('new', 'newsapi', 'ok', datetime('now'))"""
        )

        state = await gather_doctor(_settings(log_file), db)
        checks = {check["name"]: check for check in state["checks"]}

        assert checks["errors"]["status"] == "ok"
        assert "0 err" in checks["errors"]["detail"]
        assert checks["data sources"]["status"] == "ok"
        assert "recovered" in checks["data sources"]["detail"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_doctor_warns_when_latest_provider_state_is_failure(tmp_path):
    now = datetime.now(timezone.utc)
    log_file = tmp_path / "auramaur.log"
    log_file.write_text(json.dumps({
        "event": "kalshi.scan",
        "level": "info",
        "timestamp": now.isoformat(),
    }) + "\n", encoding="utf-8")

    db = Database(":memory:")
    await db.connect()
    try:
        await db.execute(
            """INSERT INTO source_fetches
               (run_id, source, status, observed_at)
               VALUES ('failed', 'metaculus', 'circuit_open', datetime('now'))"""
        )
        state = await gather_doctor(_settings(log_file), db)
        check = next(
            item for item in state["checks"] if item["name"] == "data sources")
        assert check["status"] == "warn"
        assert "metaculus" in check["detail"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_v53_labels_only_blank_settlements_as_venue_truth():
    db = Database(":memory:")
    await db.connect()
    try:
        await db.execute(
            """INSERT INTO pnl_ledger
               (market_id, strategy_source, kind, pnl, source_ref)
               VALUES ('settled', '', 'settlement', 1.0, 'settled-ref')"""
        )
        await db.execute(
            """INSERT INTO pnl_ledger
               (market_id, strategy_source, kind, pnl, source_ref)
               VALUES ('sold', '', 'sell', -1.0, 'sell-ref')"""
        )
        await db.execute("UPDATE schema_version SET version = 52")

        await db._migrate_v52_to_v53()

        rows = {
            row["source_ref"]: row["strategy_source"]
            for row in await db.fetchall(
                "SELECT source_ref, strategy_source FROM pnl_ledger")
        }
        version = await db.fetchone("SELECT version FROM schema_version")
        assert rows == {
            "settled-ref": "venue_unattributed",
            "sell-ref": "",
        }
        assert version["version"] == 53
    finally:
        await db.close()
