"""Tests for the data-source Aggregator — fan-out, category gating, dedup,
graceful per-source failure. (Individual fetchers hit external APIs; the
aggregator is the pure orchestration logic that actually matters.)"""

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

    async def fetch(self, query, limit=20):
        self.fetched = True
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
