from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from auramaur.db.database import Database
from auramaur.exchange.models import TokenType
from auramaur.risk.exit_lifecycle import ExitState, position_key, record_exit_state


def _position():
    return SimpleNamespace(
        exchange="polymarket", market_id="m-1", token=TokenType.NO, is_paper=False,
    )


def test_position_key_is_the_canonical_economic_position():
    assert position_key(_position()) == ("polymarket", "m-1", "NO", 0)


@pytest.mark.asyncio
async def test_exit_state_is_durable_and_attempts_increment(tmp_path):
    db = Database(str(tmp_path / "exit.db"))
    await db.connect()
    try:
        pos = _position()
        await record_exit_state(
            db, pos, ExitState.REQUESTED, reason="STOP_LOSS",
            increment_attempt=True,
        )
        retry = datetime.now(timezone.utc) + timedelta(minutes=15)
        await record_exit_state(
            db, pos, ExitState.RETRYABLE, reason="STOP_LOSS",
            error="book unavailable", next_retry_at=retry,
        )
        row = await db.fetchone(
            """SELECT * FROM exit_lifecycle
                WHERE exchange=? AND market_id=? AND token=? AND is_paper=?""",
            position_key(pos),
        )
        assert row["state"] == "RETRYABLE"
        assert row["reason"] == "STOP_LOSS"
        assert row["attempt_count"] == 1
        assert row["last_error"] == "book unavailable"
        assert row["next_retry_at"] is not None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_fresh_schema_contains_exit_lifecycle(tmp_path):
    db = Database(str(tmp_path / "schema.db"))
    await db.connect()
    try:
        version = await db.fetchone("SELECT version FROM schema_version")

        table = await db.fetchone(
            """SELECT name FROM sqlite_master
                WHERE type='table' AND name='exit_lifecycle'"""
        )
        assert version["version"] == 51
        assert table["name"] == "exit_lifecycle"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_telemetry_failure_never_blocks_exit():
    class BrokenDatabase:
        async def execute(self, *args, **kwargs):
            raise RuntimeError("locked")

        async def commit(self):
            raise AssertionError("commit must not follow a failed write")

    await record_exit_state(
        BrokenDatabase(), _position(), ExitState.REQUESTED,
        reason="STOP_LOSS", increment_attempt=True,
    )
