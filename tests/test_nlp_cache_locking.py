from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from auramaur.nlp.cache import NLPCache


def _db(*, fetchone=None, execute=None):
    return SimpleNamespace(
        fetchone=fetchone or AsyncMock(return_value=None),
        execute=execute or AsyncMock(),
        commit=AsyncMock(),
        db=SimpleNamespace(rollback=AsyncMock()),
    )


@pytest.mark.asyncio
async def test_cache_put_retries_lock_without_losing_analysis(monkeypatch):
    db = _db(execute=AsyncMock(side_effect=[
        aiosqlite.OperationalError("database is locked"), None,
    ]))
    monkeypatch.setattr("auramaur.nlp.cache.asyncio.sleep", AsyncMock())
    cache = NLPCache(db)

    # A transient cache collision is absorbed after the idempotent upsert succeeds.
    await cache.put("key", "kraken-dir:XBTUSDC", {"probability": 0.7}, 300)

    assert db.execute.await_count == 2
    db.db.rollback.assert_awaited_once()
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_put_lock_exhaustion_is_nonfatal(monkeypatch):
    db = _db(execute=AsyncMock(
        side_effect=aiosqlite.OperationalError("database is locked")))
    monkeypatch.setattr("auramaur.nlp.cache.asyncio.sleep", AsyncMock())
    cache = NLPCache(db)

    # Cache persistence is an optimization: exhausting retries must not make
    # ClaudeAnalyzer discard a valid result or cause Kraken to repeat the LLM call.
    await cache.put("key", "kraken-dir:ETHUSDC", {"probability": 0.65}, 300)

    assert db.execute.await_count == 3
    assert db.db.rollback.await_count == 3
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_cache_get_retries_transient_lock(monkeypatch):
    row = {"response": '{"probability": 0.61}', "ttl_seconds": 300,
           "created_at": "2026-07-17", "market_price": 0.5}
    db = _db(fetchone=AsyncMock(side_effect=[
        aiosqlite.OperationalError("database is busy"), row,
    ]))
    monkeypatch.setattr("auramaur.nlp.cache.asyncio.sleep", AsyncMock())
    cache = NLPCache(db)

    assert await cache.get("key", current_price=0.5) == {"probability": 0.61}
    assert db.fetchone.await_count == 2
    db.db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_does_not_hide_non_lock_database_errors():
    db = _db(execute=AsyncMock(
        side_effect=aiosqlite.OperationalError("disk I/O error")))
    cache = NLPCache(db)

    with pytest.raises(aiosqlite.OperationalError, match="disk I/O"):
        await cache.put("key", "market", {"probability": 0.5}, 300)
