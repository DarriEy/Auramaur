from __future__ import annotations

from collections import deque

import pytest

from auramaur.data_sources.aggregator import Aggregator
from auramaur.data_sources.bluesky import BlueskySource


class _FailingSource:
    source_name = "failing"
    categories = None

    def __init__(self):
        self.fetch_count = 0

    async def fetch(self, query, limit=20):
        self.fetch_count += 1
        raise RuntimeError("provider unavailable")


@pytest.mark.asyncio
async def test_circuit_opens_after_consecutive_provider_failures():
    source = _FailingSource()
    aggregator = Aggregator(
        [source],
        circuit_failure_threshold=2,
        circuit_cooldown_seconds=60,
    )

    assert await aggregator.gather("query one") == []
    assert await aggregator.gather("query two") == []
    assert await aggregator.gather("query three") == []
    assert source.fetch_count == 2
    assert aggregator._source_circuits["failing"][1] > 0


class _Response:
    def __init__(self, status, payload=None, detail=""):
        self.status = status
        self._payload = payload or {}
        self._detail = detail

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def json(self):
        return self._payload

    async def text(self):
        return self._detail


class _Session:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.params = []

    def get(self, url, *, params, headers, timeout):
        self.params.append(dict(params))
        return self.responses.popleft()


@pytest.mark.asyncio
async def test_bluesky_retries_400_without_optional_since_parameter():
    session = _Session([
        _Response(400, detail="invalid since"),
        _Response(200, payload={"posts": []}),
    ])
    source = BlueskySource()

    result = await source._search(session, {
        "q": "election",
        "limit": "20",
        "sort": "latest",
        "since": "2026-08-01T00:00:00Z",
    })

    assert result == {"posts": []}
    assert "since" in session.params[0]
    assert "since" not in session.params[1]
