"""Tests for market-maker quote construction."""

from types import SimpleNamespace

import pytest

from auramaur.exchange.models import Market, OrderBook, OrderBookLevel
from auramaur.strategy.market_maker import MarketMaker


def _maker(*, quote_size=10.0, max_inventory=50.0) -> MarketMaker:
    settings = SimpleNamespace(
        is_live=False,
        market_maker=SimpleNamespace(
            min_spread_bps=40,
            quote_size=quote_size,
            max_inventory=max_inventory,
            max_markets=5,
            refresh_seconds=30,
        ),
    )
    return MarketMaker(settings=settings, exchange=SimpleNamespace(), db=SimpleNamespace())


def test_market_maker_bumps_size_to_satisfy_min_notional():
    mm = _maker(quote_size=10.0, max_inventory=50.0)
    market = Market(
        id="m1",
        question="Will this happen?",
        clob_token_yes="yes",
        clob_token_no="no",
    )
    book = OrderBook(
        bids=[OrderBookLevel(price=0.04, size=100)],
        asks=[OrderBookLevel(price=0.10, size=100)],
    )

    quote, reason = mm._compute_quotes(market, book)

    assert reason is None
    assert quote is not None
    assert quote.bid_price == pytest.approx(0.05)
    assert quote.no_leg_price == pytest.approx(0.91)
    assert quote.size == pytest.approx(20.0)
    assert quote.size * quote.bid_price >= 1.0
    assert quote.size * quote.no_leg_price >= 1.0


def test_market_maker_still_skips_when_capacity_cannot_meet_min_notional():
    mm = _maker(quote_size=10.0, max_inventory=15.0)
    market = Market(
        id="m1",
        question="Will this happen?",
        clob_token_yes="yes",
        clob_token_no="no",
    )
    book = OrderBook(
        bids=[OrderBookLevel(price=0.04, size=100)],
        asks=[OrderBookLevel(price=0.10, size=100)],
    )

    quote, reason = mm._compute_quotes(market, book)

    assert quote is None
    assert reason == "min_notional"
