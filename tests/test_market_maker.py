"""Tests for market-maker quote construction."""

from types import SimpleNamespace

import pytest

from auramaur.exchange.models import Market, OrderBook, OrderBookLevel
from auramaur.strategy.market_maker import MarketMaker


def _maker(*, quote_size=10.0, max_inventory=50.0, is_live=False,
           op_timeout_seconds=15.0,
           blocked_categories=("sports", "politics_us"),
           allowed_categories_live=("crypto", "tech", "politics_intl")) -> MarketMaker:
    settings = SimpleNamespace(
        is_live=is_live,
        market_maker=SimpleNamespace(
            min_spread_bps=40,
            max_spread_bps=1500,
            quote_size=quote_size,
            max_inventory=max_inventory,
            max_markets=5,
            refresh_seconds=30,
            op_timeout_seconds=op_timeout_seconds,
        ),
        risk=SimpleNamespace(
            blocked_categories=list(blocked_categories),
            allowed_categories_live=list(allowed_categories_live),
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


@pytest.mark.asyncio
async def test_requote_cancels_previous_legs_before_replacing():
    """Re-quoting a market must cancel the prior quote's legs first.

    The overwrite of _active_quotes used to orphan the old legs' order ids —
    a resting GTC pair leaked onto the book every refresh cycle (observed
    live 2026-06-10: three stacked identical quote pairs ~36s apart)."""
    from unittest.mock import AsyncMock

    mm = _maker()
    mm._settings = SimpleNamespace(is_live=True, market_maker=mm._settings.market_maker)

    placed: list[str] = []
    counter = {"n": 0}

    async def place_order(order):
        counter["n"] += 1
        oid = f"live-{counter['n']}"
        placed.append(oid)
        return SimpleNamespace(order_id=oid, status="pending", is_paper=False)

    cancelled: list[str] = []

    async def cancel_order(order_id):
        cancelled.append(order_id)
        return True

    book = OrderBook(
        bids=[OrderBookLevel(price=0.40, size=100)],
        asks=[OrderBookLevel(price=0.46, size=100)],
    )
    moved_book = OrderBook(
        bids=[OrderBookLevel(price=0.42, size=100)],
        asks=[OrderBookLevel(price=0.48, size=100)],
    )
    mm._exchange = SimpleNamespace(
        place_order=place_order,
        cancel_order=cancel_order,
        get_order_book=AsyncMock(side_effect=[book, moved_book]),
    )
    market = Market(
        id="m1",
        question="Will this happen?",
        clob_token_yes="yes",
        clob_token_no="no",
    )

    result1, _ = await mm._quote_market(market)
    assert result1 and result1.get("success")
    first_legs = placed[:2]
    assert cancelled == []  # nothing to replace yet

    result2, _ = await mm._quote_market(market)
    assert result2 and result2.get("success")
    # The first quote's two legs were cancelled before the new pair posted.
    assert cancelled == first_legs
    assert len(placed) == 4


@pytest.mark.asyncio
async def test_unchanged_quote_keeps_resting_legs():
    """An identical requote must KEEP the resting pair, not cancel+replace.

    Cancel+replace of an unchanged quote forfeits book time-priority every
    refresh cycle (observed live 2026-07-20: identical 0.46/0.48 requoted
    every ~34s on three markets, zero fills). The guard only holds while
    both legs are still tracked as pending — a filled/cancelled leg falls
    back to the normal cancel/replace path."""
    from unittest.mock import AsyncMock

    mm = _maker()
    mm._settings = SimpleNamespace(is_live=True, market_maker=mm._settings.market_maker)

    placed: list[str] = []
    counter = {"n": 0}

    async def place_order(order):
        counter["n"] += 1
        oid = f"live-{counter['n']}"
        placed.append(oid)
        return SimpleNamespace(order_id=oid, status="pending", is_paper=False)

    cancelled: list[str] = []

    async def cancel_order(order_id):
        cancelled.append(order_id)
        return True

    book = OrderBook(
        bids=[OrderBookLevel(price=0.40, size=100)],
        asks=[OrderBookLevel(price=0.46, size=100)],
    )
    mm._exchange = SimpleNamespace(
        place_order=place_order,
        cancel_order=cancel_order,
        get_order_book=AsyncMock(return_value=book),
    )
    market = Market(
        id="m1",
        question="Will this happen?",
        clob_token_yes="yes",
        clob_token_no="no",
    )

    result1, _ = await mm._quote_market(market)
    assert result1 and result1.get("success")
    stamp_before = mm._active_quotes["m1"].placed_at

    # Same book -> same computed quote -> resting pair kept untouched.
    result2, reason = await mm._quote_market(market)
    assert result2 is None
    assert reason == "quote_unchanged"
    assert cancelled == []
    assert len(placed) == 2
    assert mm._active_quotes["m1"].placed_at >= stamp_before  # stale-reaper reset

    # One leg leaves pending tracking (filled/venue-cancelled) -> guard must
    # fail and the survivor is cancelled before a fresh pair posts.
    lost_leg = placed[0]
    mm._pending_orders.pop(lost_leg)
    result3, _ = await mm._quote_market(market)
    assert result3 and result3.get("success")
    assert placed[1] in cancelled  # surviving leg cancelled
    assert len(placed) == 4


def test_select_rejects_blocked_category_markets():
    """Regression (2026-06-12): the MM filled $6.30 of a tennis match live
    ('Stuttgart Open: X vs Y') — it was the last order path with no category
    policy. Blocked categories must never be quoted, whether the label is
    stored or only derivable by classification."""
    mm = _maker()

    def _mkt(mid, question, category=""):
        return Market(
            id=mid, question=question, category=category,
            clob_token_yes="yes", clob_token_no="no",
            liquidity=10000, spread=0.05,
        )

    tennis = _mkt("tennis",
                  "Stuttgart Open: Giovanni Mpetshi Perricard vs Alexander Bublik")
    labeled = _mkt("labeled", "Will this happen?", category="sports")
    fine = _mkt("fine", "Will Bitcoin reach $200k?", category="crypto")

    ids = [m.id for m in mm._select_mm_markets([tennis, labeled, fine])]
    assert ids == ["fine"]


def test_select_live_mode_requires_allowlist():
    """In live mode the MM only quotes allowlisted categories — unknown/
    'other' markets are paper-only, same policy as the directional books."""
    mm = _maker(is_live=True)

    def _mkt(mid, question, category=""):
        return Market(
            id=mid, question=question, category=category,
            clob_token_yes="yes", clob_token_no="no",
            liquidity=10000, spread=0.05,
        )

    unknown = _mkt("unknown", "Will the gadget ship this quarter?")
    allowed = _mkt("allowed", "Will Ethereum flip Bitcoin?", category="crypto")

    ids = [m.id for m in mm._select_mm_markets([unknown, allowed])]
    assert ids == ["allowed"]

    # Paper mode keeps exploring unknown categories.
    mm_paper = _maker(is_live=False)
    ids_paper = [m.id for m in mm_paper._select_mm_markets([unknown, allowed])]
    assert set(ids_paper) == {"unknown", "allowed"}


@pytest.mark.asyncio
async def test_partial_quote_cancels_surviving_leg():
    """If one leg is rejected (e.g. a post-only leg that would cross a moved
    book), the surviving leg must be cancelled. Otherwise it rests untracked —
    success is False so _active_quotes never records it — and the next refresh
    stacks a fresh pair on top, the orphaned-BUY cash-lock root cause."""
    from unittest.mock import AsyncMock

    mm = _maker()
    mm._settings = SimpleNamespace(is_live=True, market_maker=mm._settings.market_maker)

    placed: list[str] = []
    counter = {"n": 0}

    async def place_order(order):
        counter["n"] += 1
        oid = f"live-{counter['n']}"
        placed.append(oid)
        # Bid (1st) rests; ask (2nd) rejected by post-only crossing the book.
        status = "pending" if counter["n"] % 2 == 1 else "rejected"
        return SimpleNamespace(order_id=oid, status=status, filled_size=0, is_paper=False)

    cancelled: list[str] = []

    async def cancel_order(order_id):
        cancelled.append(order_id)
        return True

    book = OrderBook(
        bids=[OrderBookLevel(price=0.40, size=100)],
        asks=[OrderBookLevel(price=0.46, size=100)],
    )
    mm._exchange = SimpleNamespace(
        place_order=place_order,
        cancel_order=cancel_order,
        get_order_book=AsyncMock(return_value=book),
    )
    market = Market(
        id="m1",
        question="Will this happen?",
        clob_token_yes="yes",
        clob_token_no="no",
    )

    result, _ = await mm._quote_market(market)
    assert result and not result.get("success")   # partial => not a live quote
    assert cancelled == ["live-1"]                 # surviving bid leg cancelled
    assert "m1" not in mm._active_quotes           # nothing orphaned/tracked


@pytest.mark.asyncio
async def test_mm_orders_self_stamp_market_maker_source():
    """The MM routes placement through the gateway's place_quote_pair, but it
    still builds the orders, so they MUST self-stamp source='market_maker' and
    post_only — that source is what lets the order monitor write a
    correctly-attributed trades-mirror for the fill (preserving fill<->trades
    parity), and post_only keeps it maker-only."""
    from types import SimpleNamespace

    from auramaur.exchange.models import TokenType
    from auramaur.strategy.market_maker import MMQuote

    mm = _maker()
    placed = []

    async def place_order(order):
        placed.append(order)
        return SimpleNamespace(order_id=f"oid-{len(placed)}", status="pending",
                               is_paper=False, filled_size=0.0)

    mm._exchange = SimpleNamespace(place_order=place_order)

    quote = MMQuote(market_id="m1", token_yes_id="yes", token_no_id="no",
                    bid_price=0.40, ask_price=0.46, size=20, spread_bps=600)
    await mm._place_two_sided(quote, is_live=True)

    assert len(placed) == 2
    assert all(o.source == "market_maker" for o in placed)   # monitor attribution
    assert all(o.post_only for o in placed)                  # maker-only
    assert all(o.dry_run is False for o in placed)           # is_live=True
    # Bid leg buys YES, ask leg buys NO (synthetic sell-YES).
    assert {o.token for o in placed} == {TokenType.YES, TokenType.NO}


def test_run_cycle_times_out_a_hung_quote_and_continues():
    """A stuck per-market quote op (e.g. a Polymarket call with no timeout) must
    NOT hang the whole MM loop — the watchdog abandons it after op_timeout and the
    cycle completes. Regression for the 2026-06-30 MM hangs."""
    import asyncio
    from datetime import datetime, timedelta, timezone

    mm = _maker(op_timeout_seconds=0.05)
    mkt = Market(id="m1", exchange="polymarket", question="q", category="crypto",
                 active=True, outcome_yes_price=0.5, outcome_no_price=0.5,
                 liquidity=5000.0, volume=5000.0,
                 end_date=datetime.now(timezone.utc) + timedelta(days=5),
                 clob_token_yes="ty", clob_token_no="tn")

    mm._select_mm_markets = lambda markets: [mkt]

    async def _no_stale():
        return 0
    mm._cancel_stale_quotes = _no_stale

    async def _hang(market):
        await asyncio.sleep(10)  # simulate a stuck Polymarket call
        return {}, None
    mm._quote_market = _hang

    async def run():
        # Must return promptly (~op_timeout), NOT hang for 10s.
        return await asyncio.wait_for(mm.run_cycle([mkt]), timeout=3.0)

    results = asyncio.run(run())
    assert results == []  # the hung market produced no quote, loop survived
