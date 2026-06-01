"""Regression: PolymarketClient.get_order_book must parse the py_clob_client_v2
response, which is a plain dict ({"bids": [...], "asks": [...]}) — NOT a dataclass.

The original code read the levels via getattr(raw, "bids", []), which silently
returned [] for a dict, so every order book came back empty. That single bug
made the market maker skip every market on "empty_book" and silently killed the
order-flow nudge. These tests lock the dict-parsing in place.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from auramaur.exchange.client import PolymarketClient


def _client() -> PolymarketClient:
    c = PolymarketClient(MagicMock(), MagicMock())
    # Pre-set the clob client so _init_clob_client() short-circuits (no live init).
    c._clob_client = MagicMock()
    return c


@pytest.mark.asyncio
async def test_parses_v2_dict_response():
    """A dict with string-priced levels (exactly what v2 returns) parses fully."""
    c = _client()
    c._clob_client.get_order_book = MagicMock(return_value={
        "market": "0xabc",
        "asset_id": "72094069823942324362",
        "bids": [{"price": "0.61", "size": "1200"}, {"price": "0.60", "size": "800"}],
        "asks": [{"price": "0.63", "size": "1610076.25"}, {"price": "0.64", "size": "500"}],
    })

    book = await c.get_order_book("72094069823942324362")

    assert len(book.bids) == 2
    assert len(book.asks) == 2
    assert book.best_bid == 0.61
    assert book.best_ask == 0.63
    assert book.bids[0].size == 1200.0
    assert book.asks[0].size == 1610076.25


@pytest.mark.asyncio
async def test_one_sided_dict_book():
    """One-sided book (common near resolution) parses without error."""
    c = _client()
    c._clob_client.get_order_book = MagicMock(return_value={
        "bids": [],
        "asks": [{"price": "0.999", "size": "1610076.25"}],
    })

    book = await c.get_order_book("tok")

    assert book.bids == []
    assert len(book.asks) == 1
    assert book.best_ask == 0.999


@pytest.mark.asyncio
async def test_still_parses_object_response():
    """Falls back to attribute access if a client version returns an object."""
    c = _client()
    level = MagicMock(price="0.5", size="100")
    raw = MagicMock(bids=[level], asks=[level])
    c._clob_client.get_order_book = MagicMock(return_value=raw)

    book = await c.get_order_book("tok")

    assert len(book.bids) == 1
    assert len(book.asks) == 1


@pytest.mark.asyncio
async def test_fetch_error_returns_empty_book():
    """A raised exception still yields an empty OrderBook, not a crash."""
    c = _client()
    c._clob_client.get_order_book = MagicMock(side_effect=RuntimeError("404"))

    book = await c.get_order_book("tok")

    assert book.bids == []
    assert book.asks == []
