"""Tests for the shared local Ollama client (auramaur/nlp/local_llm.py)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from auramaur.db.database import Database
from auramaur.nlp import local_llm
from auramaur.nlp.local_llm import LocalLLMClient
from config.settings import LocalLLMConfig


def _settings(**overrides) -> SimpleNamespace:
    cfg = LocalLLMConfig(enabled=True, failure_cooldown_seconds=60, **overrides)
    return SimpleNamespace(local_llm=cfg)


def _ok_response(payload: dict, prompt_tokens: int = 10, output_tokens: int = 5):
    return (200, {
        "message": {"content": json.dumps(payload)},
        "prompt_eval_count": prompt_tokens,
        "eval_count": output_tokens,
    })


@pytest.fixture(autouse=True)
def _no_retry_delays(monkeypatch):
    monkeypatch.setattr(local_llm, "_RETRY_DELAYS", [])
    yield
    # Never leak the module singleton between tests.
    local_llm._client = None


async def _db() -> Database:
    db = Database(":memory:")
    await db.connect()
    return db


async def test_happy_path_records_ok_stats():
    db = await _db()
    try:
        client = LocalLLMClient(_settings(), db)
        client._request = AsyncMock(return_value=_ok_response({"score": 0.7}))
        result = await client.generate_json("prompt", purpose="triage")
        assert result == {"score": 0.7}
        row = await db.fetchone(
            "SELECT purpose, status, prompt_tokens, output_tokens "
            "FROM local_llm_calls")
        assert row["purpose"] == "triage"
        assert row["status"] == "ok"
        assert row["prompt_tokens"] == 10
        assert row["output_tokens"] == 5
    finally:
        await db.close()


async def test_malformed_json_is_parse_error():
    db = await _db()
    try:
        client = LocalLLMClient(_settings(), db)
        client._request = AsyncMock(
            return_value=(200, {"message": {"content": "not json at all"}}))
        result = await client.generate_json("prompt", purpose="distill")
        assert result is None
        row = await db.fetchone("SELECT status FROM local_llm_calls")
        assert row["status"] == "parse_error"
        # Parse failures are NOT transport failures — breaker stays closed.
        assert client.available
    finally:
        await db.close()


async def test_server_error_returns_none():
    client = LocalLLMClient(_settings(), None)
    client._request = AsyncMock(return_value=(500, {}))
    assert await client.generate_json("prompt") is None


async def test_timeout_returns_none_without_retry():
    client = LocalLLMClient(_settings(), None)
    client._request = AsyncMock(side_effect=asyncio.TimeoutError)
    assert await client.generate_json("prompt", timeout=1) is None
    assert client._request.await_count == 1


async def test_circuit_breaker_opens_after_three_failures():
    client = LocalLLMClient(_settings(), None)
    client._request = AsyncMock(side_effect=ConnectionError("refused"))
    for _ in range(3):
        assert await client.generate_json("prompt") is None
    assert not client.available
    # While open, calls short-circuit without touching the endpoint.
    before = client._request.await_count
    assert await client.generate_json("prompt") is None
    assert client._request.await_count == before


async def test_success_resets_failure_streak():
    client = LocalLLMClient(_settings(), None)
    client._request = AsyncMock(side_effect=ConnectionError("refused"))
    await client.generate_json("prompt")
    await client.generate_json("prompt")
    client._request = AsyncMock(return_value=_ok_response({"ok": True}))
    assert await client.generate_json("prompt") == {"ok": True}
    assert client._consecutive_failures == 0
    assert client.available


async def test_disabled_config_returns_none_immediately():
    client = LocalLLMClient(
        SimpleNamespace(local_llm=LocalLLMConfig(enabled=False)), None)
    client._request = AsyncMock()
    assert await client.generate_json("prompt") is None
    client._request.assert_not_awaited()


async def test_stats_skip_without_db():
    client = LocalLLMClient(_settings(), None)
    client._request = AsyncMock(return_value=_ok_response({"x": 1}))
    assert await client.generate_json("prompt") == {"x": 1}


async def test_schema_passed_as_format():
    client = LocalLLMClient(_settings(), None)
    client._request = AsyncMock(return_value=_ok_response({"x": 1}))
    schema = {"type": "object", "properties": {"x": {"type": "number"}}}
    await client.generate_json("prompt", schema=schema)
    payload = client._request.await_args.args[0]
    assert payload["format"] == schema
    assert payload["think"] is False
    assert payload["stream"] is False


def test_get_client_is_singleton():
    settings = _settings()
    a = local_llm.get_client(settings)
    b = local_llm.get_client(settings)
    assert a is b
