"""Exit-lifecycle persistence: durability, episode reset, terminal advance,
migration, and the never-block-an-exit containment contract."""

from types import SimpleNamespace

import pytest

from auramaur.db.database import Database
from auramaur.db.models import SCHEMA_VERSION
from auramaur.exchange.models import TokenType
from auramaur.risk.exit_lifecycle import (
    ExitState,
    advance_terminal,
    position_key,
    record_exit_state,
)


def _position():
    return SimpleNamespace(
        exchange="polymarket", market_id="m-1", token=TokenType.NO, is_paper=False,
    )


def test_position_key_is_the_canonical_economic_position():
    assert position_key(_position(), "polymarket") == ("polymarket", "m-1", "NO", 0)


def test_position_key_fails_loud_on_a_partial_position():
    """A position object missing is_paper must RAISE, never default: the
    12-day exit outage was a position model missing this exact field, and a
    silent paper-default would file live rows where live-mode diagnostics
    never look."""
    partial = SimpleNamespace(exchange="polymarket", market_id="m-1",
                              token=TokenType.NO)
    with pytest.raises(AttributeError):
        position_key(partial, "polymarket")


@pytest.mark.asyncio
async def test_exit_state_is_durable_and_attempts_increment(tmp_path):
    db = Database(str(tmp_path / "exit.db"))
    await db.connect()
    try:
        key = position_key(_position(), "polymarket")
        await record_exit_state(
            db, key, ExitState.RETRYABLE, reason="STOP_LOSS",
            error="book unavailable", retry_after_seconds=900,
            increment_attempt=True,
        )
        await record_exit_state(
            db, key, ExitState.RETRYABLE, reason="STOP_LOSS",
            error="book unavailable", retry_after_seconds=900,
            increment_attempt=True,
        )
        row = await db.fetchone(
            """SELECT * FROM exit_lifecycle
                WHERE exchange=? AND market_id=? AND token=? AND is_paper=?""",
            key,
        )
        assert row["state"] == "RETRYABLE"
        assert row["reason"] == "STOP_LOSS"
        assert row["attempt_count"] == 2
        assert row["last_error"] == "book unavailable"
        assert row["next_retry_at"] is not None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_next_retry_at_compares_lexically_with_sqlite_now(tmp_path):
    """The wall is written SQL-side in datetime('now') space format, so the
    natural consumer predicate (col <= datetime('now')) orders correctly —
    a Python isoformat 'T' would sort after every same-day sqlite string."""
    db = Database(str(tmp_path / "fmt.db"))
    await db.connect()
    try:
        key = position_key(_position(), "polymarket")
        await record_exit_state(
            db, key, ExitState.RETRYABLE, retry_after_seconds=900)
        row = await db.fetchone(
            """SELECT next_retry_at,
                      next_retry_at > datetime('now') AS still_future,
                      next_retry_at <= datetime('now', '+16 minutes') AS due_soon
                 FROM exit_lifecycle WHERE market_id=?""", ("m-1",))
        assert "T" not in row["next_retry_at"]
        assert row["still_future"] == 1
        assert row["due_soon"] == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_closed_resets_the_episode_for_a_reentered_position(tmp_path):
    """A row parked at CLOSED belongs to a dead episode: the next write must
    restart attempt_count (and requested_at) instead of inheriting months of
    a prior position's counters."""
    db = Database(str(tmp_path / "episode.db"))
    await db.connect()
    try:
        key = position_key(_position(), "polymarket")
        for _ in range(3):
            await record_exit_state(db, key, ExitState.RETRYABLE,
                                    increment_attempt=True)
        await advance_terminal(db, "polymarket", "m-1", 0,
                               filled=True, status="filled")
        row = await db.fetchone(
            "SELECT state FROM exit_lifecycle WHERE market_id=?", ("m-1",))
        assert row["state"] == "CLOSED"

        await record_exit_state(db, key, ExitState.RETRYABLE,
                                increment_attempt=True)
        row = await db.fetchone(
            "SELECT state, attempt_count FROM exit_lifecycle WHERE market_id=?",
            ("m-1",))
        assert row["state"] == "RETRYABLE"
        assert row["attempt_count"] == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_terminal_cancel_reopens_as_retryable(tmp_path):
    """A TTL-cancelled/expired exit order is not a closed position: the
    suppression keys were just cleared, so the row re-opens RETRYABLE and the
    next tick's attempt overwrites it with the real outcome."""
    db = Database(str(tmp_path / "cancel.db"))
    await db.connect()
    try:
        key = position_key(_position(), "polymarket")
        await record_exit_state(db, key, ExitState.ORDER_WORKING,
                                increment_attempt=True)
        await advance_terminal(db, "polymarket", "m-1", 0,
                               filled=False, status="ttl_cancelled")
        row = await db.fetchone(
            "SELECT state, reason, next_retry_at FROM exit_lifecycle "
            "WHERE market_id=?", ("m-1",))
        assert row["state"] == "RETRYABLE"
        assert row["reason"] == "ttl_cancelled"
        assert row["next_retry_at"] is None
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
        assert version["version"] == SCHEMA_VERSION
        assert table["name"] == "exit_lifecycle"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_v51_migration_creates_exit_lifecycle(tmp_path):
    """Call the migration DIRECTLY, never through connect(): _init_schema
    runs executescript(TABLES) before dispatching migrations, so a
    reconnect-and-migrate test rebuilds the table from TABLES and a gutted
    migration stays green (the pattern test_exit_policy.py documents)."""
    db = Database(str(tmp_path / "migration.db"))
    await db.connect()
    try:
        await db.execute("DROP TABLE exit_lifecycle")
        await db.execute("UPDATE schema_version SET version = 50")

        await db._migrate_v50_to_v51()

        version = await db.fetchone("SELECT version FROM schema_version")
        columns = await db.fetchall("PRAGMA table_info(exit_lifecycle)")
        assert version["version"] == 51
        assert {row["name"] for row in columns} >= {
            "exchange", "market_id", "token", "is_paper", "state",
            "attempt_count", "last_error", "next_retry_at",
        }
        # Idempotent: a second dispatch must not raise.
        await db._migrate_v50_to_v51()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_telemetry_failure_never_blocks_exit():
    """A raising database is logged and contained; the caller never sees it.
    (The wall-clock side of the contract — a WAIT, not an exception — is
    bounded by the module's asyncio.wait_for timeout.)"""

    class BrokenDatabase:
        async def execute(self, *args, **kwargs):
            raise RuntimeError("locked")

    await record_exit_state(
        BrokenDatabase(), ("polymarket", "m-1", "NO", 0), ExitState.RETRYABLE,
        reason="STOP_LOSS", increment_attempt=True,
    )
