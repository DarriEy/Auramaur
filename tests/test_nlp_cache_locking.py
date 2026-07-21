from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from auramaur.nlp.cache import NLPCache
from auramaur.db.database import Database


def _db(*, fetchone=None, execute=None):
    ns = SimpleNamespace(
        fetchone=fetchone or AsyncMock(return_value=None),
        execute=execute or AsyncMock(),
        commit=AsyncMock(),
        db=SimpleNamespace(rollback=AsyncMock(), in_transaction=False),
    )

    @asynccontextmanager
    async def _transaction():
        yield ns

    ns.transaction = _transaction
    return ns


@pytest.mark.asyncio
async def test_cache_put_retries_lock_without_losing_analysis(monkeypatch):
    db = _db(
        execute=AsyncMock(
            side_effect=[
                aiosqlite.OperationalError("database is locked"),
                None,
            ]
        )
    )
    monkeypatch.setattr("auramaur.nlp.cache.asyncio.sleep", AsyncMock())
    cache = NLPCache(db)

    # A transient cache collision is absorbed after the idempotent upsert succeeds.
    await cache.put("key", "kraken-dir:XBTUSDC", {"probability": 0.7}, 300)

    assert db.execute.await_count == 2
    db.db.rollback.assert_not_awaited()
    # Commit semantics moved inside Database.transaction(); the legacy
    # commit() must no longer be issued (it was the bleed vector).
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_cache_put_lock_exhaustion_is_nonfatal(monkeypatch):
    db = _db(execute=AsyncMock(side_effect=aiosqlite.OperationalError("database is locked")))
    monkeypatch.setattr("auramaur.nlp.cache.asyncio.sleep", AsyncMock())
    cache = NLPCache(db)

    # Cache persistence is an optimization: exhausting retries must not make
    # ClaudeAnalyzer discard a valid result or cause Kraken to repeat the LLM call.
    await cache.put("key", "kraken-dir:ETHUSDC", {"probability": 0.65}, 300)

    assert db.execute.await_count == 3
    db.db.rollback.assert_not_awaited()
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_cache_get_retries_transient_lock(monkeypatch):
    row = {
        "response": '{"probability": 0.61}',
        "ttl_seconds": 300,
        "created_at": "2026-07-17",
        "market_price": 0.5,
    }
    db = _db(
        fetchone=AsyncMock(
            side_effect=[
                aiosqlite.OperationalError("database table is locked"),
                row,
            ]
        )
    )
    monkeypatch.setattr("auramaur.nlp.cache.asyncio.sleep", AsyncMock())
    cache = NLPCache(db)

    assert await cache.get("key", current_price=0.5) == {"probability": 0.61}
    assert db.fetchone.await_count == 2
    db.db.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_cache_lock_does_not_rollback_sibling_transaction(monkeypatch):
    db = Database(":memory:")
    await db.connect()
    await db.execute("CREATE TABLE sibling_write (value TEXT)")
    await db.commit()
    await db.execute("INSERT INTO sibling_write VALUES ('must survive')")
    assert db.db.in_transaction

    original_execute = db.execute

    async def locked_cache_execute(sql, params=()):
        if "nlp_cache" in sql:
            raise aiosqlite.OperationalError("database is locked")
        return await original_execute(sql, params)

    monkeypatch.setattr(db, "execute", locked_cache_execute)
    monkeypatch.setattr("auramaur.nlp.cache.asyncio.sleep", AsyncMock())

    await NLPCache(db).put("key", "market", {"probability": 0.7}, 300)

    assert db.db.in_transaction
    row = await db.fetchone("SELECT value FROM sibling_write")
    assert row["value"] == "must survive"
    await db.db.rollback()
    await db.close()


@pytest.mark.asyncio
async def test_cache_does_not_hide_non_lock_database_errors():
    db = _db(execute=AsyncMock(side_effect=aiosqlite.OperationalError("disk I/O error")))
    cache = NLPCache(db)

    with pytest.raises(aiosqlite.OperationalError, match="disk I/O"):
        await cache.put("key", "market", {"probability": 0.5}, 300)
