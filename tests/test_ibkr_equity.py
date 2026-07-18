"""Tests for the IBKR directional-equity client — fractional-share sizing.

The book caps each order at a small USD amount (default $50). With whole-share
sizing `int($50/price)` rounded to 0 for every watchlist name (cheapest ~$200),
so the book could never open a position. These tests pin the fractional sizing
that fixes that, plus the surrounding guards (cap, no-price, zero-rounding).
"""

import sys
import math
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from auramaur.exchange.ibkr_equity import IBKREquityClient
from auramaur.exchange.models import OrderSide


def _settings(*, is_live=False, readonly=False, cap=50.0):
    return SimpleNamespace(
        is_live=is_live,
        ibkr=SimpleNamespace(
            equity_max_order_usd=cap,
            readonly=readonly,
            environment="live",
            host="127.0.0.1",
            paper_port=7497,
            live_port=7496,
            equity_client_id=2,
            market_data_type=3,
        ),
    )


async def test_forced_paper_quote_connection_ignores_shared_live_defaults(monkeypatch):
    captured = {}

    class FakeIB:
        async def connectAsync(self, **kwargs):
            captured.update(kwargs)

        def reqMarketDataType(self, value):
            captured["market_data_type"] = value

    fake = types.ModuleType("ib_async")
    fake.IB = FakeIB
    monkeypatch.setitem(sys.modules, "ib_async", fake)
    client = IBKREquityClient(_settings(readonly=False), force_paper_readonly=True)
    await client._ensure_connected()
    assert captured["port"] == 7497
    assert captured["readonly"] is True


async def test_quote_without_exchange_timestamp_fails_closed(monkeypatch):
    fake = types.ModuleType("ib_async")
    fake.Stock = lambda *a, **k: SimpleNamespace(symbol=a[0])
    monkeypatch.setitem(sys.modules, "ib_async", fake)
    client = IBKREquityClient(_settings())
    client._connected = True
    client._ib = SimpleNamespace(
        qualifyContractsAsync=AsyncMock(),
        reqTickersAsync=AsyncMock(return_value=[
            SimpleNamespace(bid=100.0, ask=100.1, time=None)]),
    )
    assert await client.get_quote("SPY") is None


async def test_nan_quote_fails_closed(monkeypatch):
    fake = types.ModuleType("ib_async")
    fake.Stock = lambda *a, **k: SimpleNamespace(symbol=a[0])
    monkeypatch.setitem(sys.modules, "ib_async", fake)
    client = IBKREquityClient(_settings())
    client._connected = True
    client._ib = SimpleNamespace(
        qualifyContractsAsync=AsyncMock(),
        reqTickersAsync=AsyncMock(return_value=[
            SimpleNamespace(bid=math.nan, ask=math.nan, time=SimpleNamespace(
                timestamp=lambda: 1.0))]),
    )
    assert await client.get_quote("SPY") is None


async def test_high_priced_stock_sizes_fractionally_not_blocked():
    """$50 into a $737 stock used to round to 0 shares (BLOCKED). It must now
    take a fractional position instead."""
    client = IBKREquityClient(_settings())
    client.get_price = AsyncMock(return_value=737.55)

    res = await client.place_order("SPY", OrderSide.BUY, 50.0)

    assert res.order_id == "PAPER"
    assert res.status == "paper"
    assert res.filled_size == pytest.approx(round(50.0 / 737.55, 4))
    assert 0 < res.filled_size < 1  # genuinely fractional


async def test_midpriced_stock_fractional():
    client = IBKREquityClient(_settings())
    client.get_price = AsyncMock(return_value=205.10)

    res = await client.place_order("NVDA", OrderSide.BUY, 50.0)

    assert res.order_id == "PAPER"
    assert res.filled_size == pytest.approx(round(50.0 / 205.10, 4))


async def test_per_order_cap_still_enforced():
    """Fractional sizing must not weaken the hard per-order USD ceiling."""
    client = IBKREquityClient(_settings(cap=50.0))
    client.get_price = AsyncMock(return_value=205.10)

    res = await client.place_order("NVDA", OrderSide.BUY, 75.0)

    assert res.order_id == "BLOCKED"
    assert "exceeds cap" in res.error_message
    client.get_price.assert_not_awaited()  # capped before any price fetch


async def test_no_price_paper_still_returns_paper():
    """A missing quote no longer errors the order (live sizing is server-side via
    cashQty); paper just reports a 0-qty estimate."""
    client = IBKREquityClient(_settings())
    client.get_price = AsyncMock(return_value=None)

    res = await client.place_order("SPY", OrderSide.BUY, 50.0)

    assert res.order_id == "PAPER"
    assert res.filled_size == 0.0


def _fake_ib_async(monkeypatch, created):
    """Install a fake ib_async whose MarketOrder records the order object so a
    test can assert on cashQty / totalQuantity."""
    fake = types.ModuleType("ib_async")
    fake.Stock = lambda *a, **k: SimpleNamespace(symbol=a[0] if a else "X")

    def _market_order(action, qty):
        o = SimpleNamespace(action=action, totalQuantity=qty, cashQty=None)
        created["order"] = o
        return o

    fake.MarketOrder = _market_order
    monkeypatch.setitem(sys.modules, "ib_async", fake)


async def test_live_order_uses_cashqty_not_fractional_qty(monkeypatch):
    """The live order must be dollar-denominated (cashQty), never a fractional
    share quantity — IBKR rejects the latter over the API with Error 10243."""
    created = {}
    _fake_ib_async(monkeypatch, created)

    client = IBKREquityClient(_settings(is_live=True, readonly=False))
    client.get_price = AsyncMock(return_value=205.10)
    client._ensure_connected = AsyncMock()
    client._ib = SimpleNamespace(
        qualifyContractsAsync=AsyncMock(),
        placeOrder=lambda stock, order: SimpleNamespace(order=SimpleNamespace(orderId=42)),
    )

    res = await client.place_order("NVDA", OrderSide.BUY, 50.0)

    assert res.is_paper is False
    assert res.order_id == "42"
    o = created["order"]
    assert o.action == "BUY"
    assert o.totalQuantity == 0                  # never a fractional share qty
    assert o.cashQty == pytest.approx(50.0)      # dollar-sized instead


async def test_live_order_places_without_price_via_cashqty(monkeypatch):
    """No market-data entitlement (price None) must NOT block a live order —
    cashQty sizes server-side, so it still goes out."""
    created = {}
    _fake_ib_async(monkeypatch, created)

    client = IBKREquityClient(_settings(is_live=True, readonly=False))
    client.get_price = AsyncMock(return_value=None)
    client._ensure_connected = AsyncMock()
    client._ib = SimpleNamespace(
        qualifyContractsAsync=AsyncMock(),
        placeOrder=lambda stock, order: SimpleNamespace(order=SimpleNamespace(orderId=43)),
    )

    res = await client.place_order("SPY", OrderSide.BUY, 50.0)

    assert res.order_id == "43"
    assert created["order"].cashQty == pytest.approx(50.0)


# --- close_position (cashQty-valued exit of a fractional position) ---

async def test_close_position_no_holding_returns_none():
    client = IBKREquityClient(_settings())
    client.get_positions = AsyncMock(return_value={})

    assert await client.close_position("NVDA") is None


async def test_close_position_paper():
    client = IBKREquityClient(_settings())
    client.get_positions = AsyncMock(return_value={"NVDA": 0.16})
    client.get_price = AsyncMock(return_value=205.10)

    res = await client.close_position("NVDA")

    assert res.order_id == "PAPER"
    assert res.filled_size == pytest.approx(0.16)


async def test_close_position_live_sells_full_value_via_cashqty(monkeypatch):
    """Exit sells the position's market VALUE via cashQty (a fractional sell qty
    would hit Error 10243 just like a buy)."""
    created = {}
    _fake_ib_async(monkeypatch, created)

    client = IBKREquityClient(_settings(is_live=True, readonly=False))
    client.get_positions = AsyncMock(return_value={"NVDA": 0.16})
    client.get_price = AsyncMock(return_value=205.10)
    client._ensure_connected = AsyncMock()
    client._ib = SimpleNamespace(
        qualifyContractsAsync=AsyncMock(),
        placeOrder=lambda stock, order: SimpleNamespace(order=SimpleNamespace(orderId=44)),
    )

    res = await client.close_position("NVDA")

    assert res.order_id == "44"
    o = created["order"]
    assert o.action == "SELL"
    assert o.totalQuantity == 0
    assert o.cashQty == pytest.approx(round(0.16 * 205.10, 2))


async def test_close_position_readonly_blocked_when_live():
    client = IBKREquityClient(_settings(is_live=True, readonly=True))
    client.get_positions = AsyncMock(return_value={"NVDA": 0.16})
    client.get_price = AsyncMock(return_value=205.10)

    res = await client.close_position("NVDA")

    assert res.order_id == "BLOCKED"
    assert "readonly" in res.error_message


# --- get_price historical-close fallback (no live data subscription) ---

async def test_get_price_falls_back_to_historical_close(monkeypatch):
    """With no US-equity entitlement, live/delayed quotes are NaN; get_price must
    fall back to the most recent historical daily close rather than return None."""
    fake = types.ModuleType("ib_async")
    fake.Stock = lambda *a, **k: SimpleNamespace(symbol=a[0] if a else "X")
    monkeypatch.setitem(sys.modules, "ib_async", fake)

    nan = float("nan")
    ticker = SimpleNamespace(marketPrice=lambda: nan, close=nan)
    client = IBKREquityClient(_settings())
    client._ensure_connected = AsyncMock()
    client._ib = SimpleNamespace(
        qualifyContractsAsync=AsyncMock(),
        reqTickersAsync=AsyncMock(return_value=[ticker]),
        reqHistoricalDataAsync=AsyncMock(return_value=[SimpleNamespace(close=205.10)]),
    )

    px = await client.get_price("NVDA")

    assert px == pytest.approx(205.10)
    client._ib.reqHistoricalDataAsync.assert_awaited_once()


# --- usd_cash reads the FX position (converted USD shows there, not CashBalance) ---

async def test_usd_cash_reads_fx_position():
    """Converted USD lands as a virtual FX position (symbol USD / secType CASH),
    while accountValues().CashBalance is empty — usd_cash must read the position
    so it sees the funding and doesn't re-convert every cooldown."""
    client = IBKREquityClient(_settings())
    client._ensure_connected = AsyncMock()
    usd_pos = SimpleNamespace(
        contract=SimpleNamespace(symbol="USD", secType="CASH"), position=120.0
    )
    client._ib = SimpleNamespace(positions=lambda: [usd_pos], accountValues=lambda: [])

    assert await client.usd_cash() == pytest.approx(120.0)


async def test_usd_cash_zero_when_no_usd():
    client = IBKREquityClient(_settings())
    client._ensure_connected = AsyncMock()
    client._ib = SimpleNamespace(positions=lambda: [], accountValues=lambda: [])

    assert await client.usd_cash() == 0.0


# --- auto FX top-up (CAD -> USD) ---

async def test_fx_topup_below_target_converts_dry_run():
    client = IBKREquityClient(_settings(is_live=False))
    client.usd_cash = AsyncMock(return_value=0.0)

    res = await client.ensure_usd_float(
        target_usd=120.0, max_convert_usd=150.0, min_convert_usd=20.0, source_ccy="CAD"
    )

    assert res is not None
    assert res.order_id == "PAPER"
    assert res.market_id == "USDCAD"
    assert res.filled_size == pytest.approx(120.0)  # tops up the full shortfall


async def test_fx_topup_capped_per_conversion():
    client = IBKREquityClient(_settings(is_live=False))
    client.usd_cash = AsyncMock(return_value=0.0)

    res = await client.ensure_usd_float(
        target_usd=300.0, max_convert_usd=150.0, min_convert_usd=20.0, source_ccy="CAD"
    )

    assert res.filled_size == pytest.approx(150.0)  # clamped to the cap, not 300


async def test_fx_topup_already_funded_noops():
    client = IBKREquityClient(_settings(is_live=False))
    client.usd_cash = AsyncMock(return_value=200.0)

    res = await client.ensure_usd_float(
        target_usd=120.0, max_convert_usd=150.0, min_convert_usd=20.0, source_ccy="CAD"
    )

    assert res is None  # at/above target -> no conversion


async def test_fx_topup_below_dust_floor_noops():
    client = IBKREquityClient(_settings(is_live=False))
    client.usd_cash = AsyncMock(return_value=110.0)  # shortfall 10 < min 20

    res = await client.ensure_usd_float(
        target_usd=120.0, max_convert_usd=150.0, min_convert_usd=20.0, source_ccy="CAD"
    )

    assert res is None


async def test_fx_topup_unreachable_balance_noops():
    client = IBKREquityClient(_settings(is_live=False))
    client.usd_cash = AsyncMock(return_value=None)  # gateway unreachable

    res = await client.ensure_usd_float(
        target_usd=120.0, max_convert_usd=150.0, min_convert_usd=20.0, source_ccy="CAD"
    )

    assert res is None


async def test_fx_topup_live_places_forex_buy(monkeypatch):
    """Live path must BUY the USD.<ccy> forex pair for the (whole-USD) shortfall."""
    captured = {}

    fake = types.ModuleType("ib_async")
    fake.Forex = lambda pair: SimpleNamespace(pair=pair)

    def _market_order(action, qty):
        captured["action"] = action
        captured["qty"] = qty
        return SimpleNamespace(order=SimpleNamespace(orderId=0))

    fake.MarketOrder = _market_order
    monkeypatch.setitem(sys.modules, "ib_async", fake)

    client = IBKREquityClient(_settings(is_live=True, readonly=False))
    client.usd_cash = AsyncMock(return_value=0.0)
    client._ensure_connected = AsyncMock()
    client._ib = SimpleNamespace(
        qualifyContractsAsync=AsyncMock(),
        placeOrder=lambda contract, order: SimpleNamespace(
            order=SimpleNamespace(orderId=7)
        ),
    )

    res = await client.ensure_usd_float(
        target_usd=120.0, max_convert_usd=150.0, min_convert_usd=20.0, source_ccy="CAD"
    )

    assert res.is_paper is False
    assert res.order_id == "7"
    assert captured["action"] == "BUY"
    assert captured["qty"] == pytest.approx(120.0)


async def test_fx_topup_cooldown_prevents_double_convert():
    """A second call right after a conversion must no-op, so a fast loop can't
    double-convert while the first fill is still posting/settling."""
    client = IBKREquityClient(_settings(is_live=False))
    client.usd_cash = AsyncMock(return_value=0.0)
    kw = dict(target_usd=120.0, max_convert_usd=150.0, min_convert_usd=20.0, source_ccy="CAD")

    first = await client.ensure_usd_float(**kw)
    second = await client.ensure_usd_float(**kw)

    assert first is not None and first.order_id == "PAPER"
    assert second is None  # suppressed by the cooldown


async def test_fx_topup_readonly_blocked_when_live():
    client = IBKREquityClient(_settings(is_live=True, readonly=True))
    client.usd_cash = AsyncMock(return_value=0.0)

    res = await client.ensure_usd_float(
        target_usd=120.0, max_convert_usd=150.0, min_convert_usd=20.0, source_ccy="CAD"
    )

    assert res.order_id == "BLOCKED"
    assert "readonly" in res.error_message
