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
            max_spread_bps=1500,
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


def test_compute_quotes_rejects_dead_book_at_price_bounds():
    """A live book sitting at the 0.02/0.98 bounds is no real market — reject it
    rather than quote into a 9600 bps 'spread' that never fills."""
    mm = _maker()
    market = Market(
        id="m1",
        question="Will this happen?",
        clob_token_yes="yes",
        clob_token_no="no",
    )
    book = OrderBook(
        bids=[OrderBookLevel(price=0.02, size=100)],
        asks=[OrderBookLevel(price=0.98, size=100)],
    )

    quote, reason = mm._compute_quotes(market, book)

    assert quote is None
    assert reason == "dead_book"


def test_compute_quotes_accepts_book_just_under_max_spread():
    """A wide-but-real book (within max_spread_bps) is still quotable."""
    mm = _maker()
    market = Market(
        id="m1",
        question="Will this happen?",
        clob_token_yes="yes",
        clob_token_no="no",
    )
    # 0.45 / 0.58 = 1300 bps < 1500 max
    book = OrderBook(
        bids=[OrderBookLevel(price=0.45, size=100)],
        asks=[OrderBookLevel(price=0.58, size=100)],
    )

    quote, reason = mm._compute_quotes(market, book)

    assert reason is None
    assert quote is not None


def test_select_skips_dead_book_but_keeps_real_spread():
    """Selection rejects a near-100% summary spread (dead book) while keeping a
    normal one — and no longer ranks the dead book first."""
    mm = _maker()

    def _mkt(mid, spread):
        return Market(
            id=mid,
            question="Will this happen?",
            clob_token_yes="yes",
            clob_token_no="no",
            liquidity=10000,
            spread=spread,
        )

    dead = _mkt("dead", 0.96)   # 9600 bps — empty book
    real = _mkt("real", 0.05)   # 500 bps — legitimate

    suitable = mm._select_mm_markets([dead, real])

    ids = [m.id for m in suitable]
    assert "dead" not in ids
    assert ids == ["real"]
