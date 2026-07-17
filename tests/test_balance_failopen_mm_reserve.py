"""Balance fail-open + the MM cash-reserve floor.

1. A transient balance-API failure is NOT a zero balance: _get_live_balance
   must serve the last good reading instead of 0.0 (returning 0 stomped the
   monitor's cash to $0 — a false LOW CASH alarm against a funded venue
   account — and could wrongly gate entries).
2. MM must not place NEW quote pairs while spendable live cash sits at or
   below its reserve floor — without the floor, the only always-live cell
   auto-claims every deposited dollar as inventory working capital.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from auramaur.broker.sync import PositionSyncer


def _syncer(gross_usd_6dp: int | Exception) -> PositionSyncer:
    s = PositionSyncer.__new__(PositionSyncer)
    s._settings = MagicMock()
    s._settings.is_live = True
    s._last_balance = 0.0
    exchange = MagicMock()
    exchange._init_clob_client = MagicMock()
    client = MagicMock()
    if isinstance(gross_usd_6dp, Exception):
        client.get_balance_allowance = MagicMock(side_effect=gross_usd_6dp)
    else:
        client.get_balance_allowance = MagicMock(
            return_value={"balance": gross_usd_6dp})
    exchange._clob_client = client
    exchange._reserved_buy_collateral_from_open_orders = MagicMock(return_value=0.0)
    s._exchange = exchange
    return s


@pytest.mark.asyncio
async def test_balance_error_serves_last_good_reading():
    s = _syncer(138_520_000)  # $138.52
    assert await s._get_live_balance() == pytest.approx(138.52)
    # API starts throwing — the last good reading is served, not 0.0.
    s._exchange._clob_client.get_balance_allowance = MagicMock(
        side_effect=RuntimeError("Request exception!"))
    assert await s._get_live_balance() == pytest.approx(138.52)


@pytest.mark.asyncio
async def test_balance_error_before_any_reading_is_zero():
    s = _syncer(RuntimeError("down"))
    # No good reading yet — 0.0 is the only honest floor.
    assert await s._get_live_balance() == 0.0


# ---------------------------------------------------------------------------
# MM reserve floor
# ---------------------------------------------------------------------------


def _mm(free: float | None, reserve: float):
    from auramaur.strategy.market_maker import MarketMaker

    mm = MarketMaker.__new__(MarketMaker)
    mm._settings = MagicMock()
    mm._settings.is_live = True
    mm._settings.market_maker.cash_reserve_usd = reserve
    mm._exchange = MagicMock()
    mm._exchange._free_collateral_usd = AsyncMock(return_value=free)
    # get_order_book must not be reached when the floor blocks.
    mm._exchange.get_order_book = AsyncMock(
        side_effect=AssertionError("quote path reached past the reserve floor"))
    return mm


@pytest.mark.asyncio
async def test_mm_skips_new_quotes_at_reserve_floor():
    mm = _mm(free=42.0, reserve=50.0)
    result, reason = await mm._quote_market(MagicMock())
    assert result is None
    assert reason == "cash_reserve_floor"


@pytest.mark.asyncio
async def test_mm_quotes_above_reserve_and_fails_open_on_probe_error():

    for free in (120.0, None):  # ample cash / probe unavailable -> proceed
        mm = _mm(free=free, reserve=50.0)
        # Past the floor the quote path runs; stub it to observe passage.
        mm._exchange.get_order_book = AsyncMock(
            return_value=MagicMock(bids=[], asks=[]))
        result, reason = await mm._quote_market(MagicMock())
        assert reason == "empty_book"  # i.e. we got PAST the floor
