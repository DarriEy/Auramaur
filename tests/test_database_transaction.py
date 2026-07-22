"""Tests for Database.transaction() and the CLI schema fast path.

Context (docs/plans/db-contention-plan.md): ~30 pillar tasks share ONE
aiosqlite connection with implicit deferred transactions — task B's commit()
could land task A's half-written rows, and an error-path rollback() could
discard another task's writes. transaction() serializes and isolates
adopters; ensure_schema=False lets CLI/tooling connect without taking any
write lock when the schema is already current.
"""

from __future__ import annotations

import asyncio

import pytest

from auramaur.db.database import Database


async def _fresh_db(tmp_path) -> Database:
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    await db.execute(
        "CREATE TABLE IF NOT EXISTS t (k TEXT PRIMARY KEY, v INTEGER)")
    await db.commit()
    return db


@pytest.mark.asyncio
async def test_concurrent_transactions_serialize_and_both_land(tmp_path):
    db = await _fresh_db(tmp_path)
    try:
        order: list[str] = []

        async def writer(name: str):
            async with db.transaction():
                order.append(f"{name}:in")
                await db.execute(
                    "INSERT INTO t (k, v) VALUES (?, 1)", (name,))
                await asyncio.sleep(0.02)  # yield while holding the txn
                order.append(f"{name}:out")

        await asyncio.gather(writer("a"), writer("b"))

        # Strict serialization: no interleaving of in/out pairs.
        assert order in (["a:in", "a:out", "b:in", "b:out"],
                         ["b:in", "b:out", "a:in", "a:out"])
        rows = await db.fetchall("SELECT k FROM t ORDER BY k")
        assert [r["k"] for r in rows] == ["a", "b"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_rollback_discards_only_its_own_writes(tmp_path):
    db = await _fresh_db(tmp_path)
    try:
        async with db.transaction():
            await db.execute("INSERT INTO t (k, v) VALUES ('keep', 1)")

        with pytest.raises(RuntimeError):
            async with db.transaction():
                await db.execute("INSERT INTO t (k, v) VALUES ('drop', 1)")
                raise RuntimeError("boom")

        rows = await db.fetchall("SELECT k FROM t")
        assert [r["k"] for r in rows] == ["keep"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_transaction_waits_out_legacy_autocommit_writer(tmp_path):
    """A non-adopted writer mid-flight (implicit txn open, commit imminent)
    must NOT have its half-written rows committed or rolled back by
    transaction() — the exact bleed the guard exists to remove."""
    db = await _fresh_db(tmp_path)
    try:
        legacy_committed = asyncio.Event()

        async def legacy_writer():
            await db.execute("INSERT INTO t (k, v) VALUES ('legacy', 1)")
            await asyncio.sleep(0.05)  # implicit txn stays open across a yield
            await db.commit()
            legacy_committed.set()

        async def adopter():
            await asyncio.sleep(0.01)  # let the legacy txn open first
            async with db.transaction():
                # By the time BEGIN IMMEDIATE ran, the legacy writer must
                # have landed its own commit.
                assert legacy_committed.is_set()
                await db.execute("INSERT INTO t (k, v) VALUES ('adopted', 1)")

        await asyncio.gather(legacy_writer(), adopter())
        rows = await db.fetchall("SELECT k FROM t ORDER BY k")
        assert [r["k"] for r in rows] == ["adopted", "legacy"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_ensure_schema_false_skips_ddl_when_current(tmp_path, monkeypatch):
    path = str(tmp_path / "t.db")
    db = Database(path)
    await db.connect()  # full init stamps SCHEMA_VERSION
    await db.close()

    db2 = Database(path)
    called = False

    async def _boom():
        nonlocal called
        called = True

    monkeypatch.setattr(db2, "_init_schema", _boom)
    await db2.connect(ensure_schema=False)
    try:
        assert called is False  # no DDL, no write locks
        row = await db2.fetchone("SELECT version FROM schema_version")
        assert row is not None
    finally:
        await db2.close()


@pytest.mark.asyncio
async def test_ensure_schema_false_still_initializes_fresh_file(tmp_path):
    """The fast path must never leave a caller on a missing/stale schema."""
    db = Database(str(tmp_path / "fresh.db"))
    await db.connect(ensure_schema=False)
    try:
        row = await db.fetchone("SELECT version FROM schema_version")
        assert row is not None  # full init ran despite the flag
    finally:
        await db.close()

@pytest.mark.asyncio
async def test_order_position_heartbeat_and_lineage_writes_serialize(tmp_path):
    """The four writers active in a trading cycle cannot nest/bleed on the
    shared connection, even when they all become runnable together."""
    from auramaur.broker.execution_gateway import ExecutionGateway
    from auramaur.monitoring.heartbeat import beat

    db = await _fresh_db(tmp_path)
    gateway = object.__new__(ExecutionGateway)
    gateway.db = db

    async def owned_writer(owner: str, key: str):
        async with db.transaction(owner=owner):
            await db.execute("INSERT INTO t (k, v) VALUES (?, 1)", (key,))
            await asyncio.sleep(0.01)

    try:
        await asyncio.gather(
            gateway._serialized_write(
                "INSERT INTO t (k, v) VALUES (?, 1)", ("order",)),
            owned_writer("position_sync", "position"),
            beat(db, "concurrent_heartbeat", entries=1),
            owned_writer("lineage", "lineage"),
        )
        rows = await db.fetchall("SELECT k FROM t ORDER BY k")
        assert [row["k"] for row in rows] == ["lineage", "order", "position"]
        heartbeat = await db.fetchone(
            "SELECT cycles FROM strategy_heartbeats WHERE strategy = ?",
            ("concurrent_heartbeat",),
        )
        assert heartbeat["cycles"] == 1
        assert db._txn_task is None
        assert db._txn_owner is None
        assert db.db._conn.in_transaction is False
    finally:
        await db.close()
