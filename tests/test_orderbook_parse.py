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


# ---------------------------------------------------------------------------
# Kalshi: the book publishes resting BIDS on each side. `yes` = YES bids,
# `no` = NO bids. A NO bid at p is an offer to sell YES at (1 - p), so asks
# must be derived as (1 - no_price). Modelled on a live KXGDPYEAR tail bin.
# ---------------------------------------------------------------------------

def _kalshi_client(payload: dict):
    import json
    from unittest.mock import AsyncMock
    from auramaur.exchange.kalshi import KalshiClient

    c = KalshiClient.__new__(KalshiClient)
    c._init_api = lambda: None
    c._markets_api = MagicMock()
    resp = MagicMock(raw_data=json.dumps(payload))
    c._call = AsyncMock(return_value=resp)
    return c


@pytest.mark.asyncio
async def test_kalshi_ask_derived_from_no_bid():
    """best_ask must be (1 - best NO bid), not the raw NO-bid price.

    Regression for the zero-price churn loop: with the raw NO-bid price,
    best_ask = min(no_bids) collapsed to the cheapest longshot bid (~0.01),
    yielding a negative spread and an unfillable 1c entry that re-approved
    every cycle.
    """
    c = _kalshi_client({
        "orderbook": {
            # YES bids: best is 0.07
            "yes": [["0.01", "922"], ["0.07", "1500"]],
            # NO bids: best is 0.86 -> YES ask 0.14; a 0.01 longshot bid -> 0.99
            "no": [["0.01", "4401"], ["0.86", "2"], ["0.45", "777"]],
        }
    })

    book = await c.get_order_book("KXGDPYEAR-30-B2.3")

    assert book.best_bid == 0.07
    assert book.best_ask == 0.14          # 1 - 0.86, NOT the raw 0.01 NO bid
    assert book.spread == pytest.approx(0.07)   # positive, not -0.06
