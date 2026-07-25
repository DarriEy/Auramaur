"""Tests for the data-source Aggregator — fan-out, category gating, dedup,
graceful per-source failure. (Individual fetchers hit external APIs; the
aggregator is the pure orchestration logic that actually matters.)"""

from datetime import datetime, timezone

import pytest

from auramaur.data_sources.aggregator import Aggregator
from auramaur.data_sources.base import NewsItem


class _FakeSource:
    def __init__(self, name, items, categories=None, raises=False):
        self.source_name = name
        self.categories = categories
        self._items = items
        self._raises = raises
        self.fetched = False
        self.fetch_count = 0

    async def fetch(self, query, limit=20):
        self.fetched = True
        self.fetch_count += 1
        if self._raises:
            raise RuntimeError("source boom")
        return list(self._items)

    async def close(self):
        pass


def _item(source, title):
    return NewsItem(id=f"{source}:{title}", source=source, title=title, content="x")


@pytest.mark.asyncio
async def test_safe_fetch_swallows_source_errors():
    """One source raising must not break the gather — others still return."""
    good = _FakeSource("good", [_item("good", "alpha")])
    bad = _FakeSource("bad", [], raises=True)
    agg = Aggregator([bad, good])
    out = await agg.gather("q", category=None)
    titles = [i.title for i in out]
    assert "alpha" in titles  # good source survived the bad one


@pytest.mark.asyncio
async def test_category_gating():
    """Agnostic (categories=None) always fires; domain source only on a match."""
    agnostic = _FakeSource("web", [_item("web", "general")], categories=None)
    crypto = _FakeSource("coingecko", [_item("coingecko", "btc")], categories={"crypto"})

    agg = Aggregator([agnostic, crypto])
    crypto_out = await agg.gather("q", category="crypto")
    assert agnostic.fetched and crypto.fetched
    assert {"general", "btc"} == {i.title for i in crypto_out}

    # Reset and fire a non-crypto category — domain source must stay silent.
    agnostic.fetched = crypto.fetched = False
    sports_out = await agg.gather("q", category="sports")
    assert agnostic.fetched and not crypto.fetched
    assert {"general"} == {i.title for i in sports_out}


@pytest.mark.asyncio
async def test_none_category_skips_domain_sources():
    """A category-less query fires only the None-gated sources."""
    agnostic = _FakeSource("web", [_item("web", "general")], categories=None)
    domain = _FakeSource("usgs", [_item("usgs", "quake")], categories={"weather"})
    agg = Aggregator([agnostic, domain])
    out = await agg.gather("q", category=None)
    assert agnostic.fetched and not domain.fetched
    assert {"general"} == {i.title for i in out}


@pytest.mark.asyncio
async def test_dedup_by_normalised_title():
    """Same headline from two sources collapses to one item."""
    a = _FakeSource("reuters", [_item("reuters", "Big News!")])
    b = _FakeSource("ap", [_item("ap", "big news")])  # same title, different case/punct
    agg = Aggregator([a, b])
    out = await agg.gather("q", category=None)
    assert len(out) == 1


@pytest.mark.asyncio
async def test_ttl_cache_coalesces_equivalent_evidence_reads():
    source = _FakeSource("wire", [_item("wire", "Headline")])
    agg = Aggregator([source], cache_ttl_seconds=60)
    first = await agg.gather("Big News!", category="finance")
    second = await agg.gather("big news", category="finance")
    assert source.fetch_count == 1
    assert first[0].ingestion_run_id == second[0].ingestion_run_id
    assert first[0] is not second[0]


@pytest.mark.asyncio
async def test_transient_source_failure_is_never_cached():
    source = _FakeSource("wire", [], raises=True)
    agg = Aggregator([source], cache_ttl_seconds=60)
    assert await agg.gather("query") == []
    assert await agg.gather("query") == []
    assert source.fetch_count == 2


def test_news_item_normalizes_naive_provider_time_to_utc():
    item = NewsItem(
        id="n1", source="provider", title="Headline",
        published_at=datetime(2026, 7, 24, 12, 0),
    )
    assert item.published_at.tzinfo is timezone.utc
