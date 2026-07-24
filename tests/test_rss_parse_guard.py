"""Tests for the RSS parse guard — feedparser.parse must never run on the
event loop unbounded. (A pathological feed body observed 2026-07-23 parsed
for ~25 minutes, freezing exits and risk checks for the duration.)"""

import time

import pytest

from auramaur.data_sources import rss as rss_mod
from auramaur.data_sources.rss import RSSSource


class _FakeResponse:
    def __init__(self, body: str):
        self._body = body

    def raise_for_status(self) -> None:
        pass

    async def text(self) -> str:
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, body: str):
        self._body = body
        self.closed = False

    def get(self, url, timeout=None):
        return _FakeResponse(self._body)


def _source_with_body(body: str) -> RSSSource:
    src = RSSSource(feed_urls=["https://example.test/feed.xml"])
    src._session = _FakeSession(body)  # type: ignore[assignment]
    return src


_FEED = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>t</title>
<item><title>alpha beta headline</title><link>https://example.test/a</link>
<description>alpha beta story</description></item>
</channel></rss>"""


@pytest.mark.asyncio
async def test_normal_feed_still_parses():
    src = _source_with_body(_FEED)
    items = await src._fetch_feed("https://example.test/feed.xml", "alpha beta", 5)
    assert len(items) == 1
    assert "alpha" in items[0].title.lower()


@pytest.mark.asyncio
async def test_oversized_body_is_dropped(monkeypatch):
    monkeypatch.setattr(rss_mod, "_MAX_FEED_BYTES", 100)
    src = _source_with_body(_FEED + "x" * 200)
    items = await src._fetch_feed("https://example.test/feed.xml", "", 5)
    assert items == []


@pytest.mark.asyncio
async def test_slow_parse_times_out_instead_of_blocking(monkeypatch):
    monkeypatch.setattr(rss_mod, "_PARSE_TIMEOUT_SECONDS", 0.1)

    def _slow_parse(body):
        time.sleep(1.0)
        raise AssertionError("parse should have been abandoned before this")

    monkeypatch.setattr(rss_mod.feedparser, "parse", _slow_parse)
    src = _source_with_body(_FEED)
    items = await src._fetch_feed("https://example.test/feed.xml", "", 5)
    assert items == []
