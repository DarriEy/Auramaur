"""Tests for the Google Trends parse guard — ET.fromstring must never run on
the event loop unbounded. Same failure mode as the RSS incident of 2026-07-23
(~25-minute parse froze exits and risk checks); the aiohttp ClientTimeout
bounds the fetch, not the parse. Mirrors tests/test_rss_parse_guard.py."""

import time

import pytest

from auramaur.data_sources import google_trends as trends_mod
from auramaur.data_sources.google_trends import GoogleTrendsSource


class _FakeResponse:
    def __init__(self, body: str, status: int = 200):
        self._body = body
        self.status = status

    async def text(self) -> str:
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, body: str):
        self._body = body

    def get(self, url, timeout=None):
        return _FakeResponse(self._body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _patch_body(monkeypatch, body: str) -> None:
    monkeypatch.setattr(
        trends_mod.aiohttp, "ClientSession", lambda *a, **k: _FakeSession(body)
    )


_FEED = """<?xml version="1.0"?>
<rss version="2.0" xmlns:ht="https://trends.google.com/trending/rss">
<channel><title>t</title>
<item><title>alpha beta</title><ht:approx_traffic>1000+</ht:approx_traffic></item>
</channel></rss>"""


@pytest.mark.asyncio
async def test_normal_feed_still_parses(monkeypatch):
    _patch_body(monkeypatch, _FEED)
    items = await GoogleTrendsSource().fetch("alpha beta", limit=5)
    assert isinstance(items, list)


@pytest.mark.asyncio
async def test_oversized_body_is_dropped(monkeypatch):
    monkeypatch.setattr(trends_mod, "_MAX_FEED_BYTES", 100)
    _patch_body(monkeypatch, _FEED + "x" * 200)
    assert await GoogleTrendsSource().fetch("alpha", limit=5) == []


@pytest.mark.asyncio
async def test_slow_parse_times_out_instead_of_blocking(monkeypatch):
    monkeypatch.setattr(trends_mod, "_PARSE_TIMEOUT_SECONDS", 0.1)

    def _slow_parse(text):
        time.sleep(1.0)
        raise AssertionError("parse should have been abandoned before this")

    monkeypatch.setattr(trends_mod.ET, "fromstring", _slow_parse)
    _patch_body(monkeypatch, _FEED)
    assert await GoogleTrendsSource().fetch("alpha", limit=5) == []


@pytest.mark.asyncio
async def test_malformed_xml_is_dropped_not_raised(monkeypatch):
    _patch_body(monkeypatch, "<rss><unclosed>")
    assert await GoogleTrendsSource().fetch("alpha", limit=5) == []
