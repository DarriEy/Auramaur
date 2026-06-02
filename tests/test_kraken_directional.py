"""Tests for the Kraken directional spot pillar — asymmetric long bias."""

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auramaur.treasury.kraken_pillar import KrakenPillar
from auramaur.exchange.kraken import KrakenSpotClient
from auramaur.exchange.models import OrderResult, OrderSide


# ---------------------------------------------------------------------------
# Quote-currency-aware sizing — arbitrary pairs / quote currencies
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_quote_usd_rate_pegged_and_fiat():
    c = KrakenSpotClient(MagicMock())
    assert await c.quote_usd_rate("USDC") == 1.0
    assert await c.quote_usd_rate("ZUSD") == 1.0
    assert await c.quote_usd_rate("USDT") == 1.0
    c.get_price = AsyncMock(return_value=1.16)  # EURUSD
    assert await c.quote_usd_rate("ZEUR") == 1.16
    # cached: a second call must not hit the API again
    c.get_price = AsyncMock(side_effect=AssertionError("rate should be cached"))
    assert await c.quote_usd_rate("ZEUR") == 1.16


@pytest.mark.asyncio
async def test_size_for_usd_usdc_no_fx():
    c = KrakenSpotClient(MagicMock())
    c.get_pair_quote = AsyncMock(return_value="USDC")
    vol = await c.size_for_usd("XBTUSDC", 30.0, price=60000.0)
    assert abs(vol - 30.0 / 60000.0) < 1e-12


@pytest.mark.asyncio
async def test_size_for_usd_is_quote_aware_for_eur():
    """$30 of a EUR-quoted pair buys fewer base units than the raw price implies,
    because EUR is worth >$1 — sizing must convert via the EURUSD rate."""
    c = KrakenSpotClient(MagicMock())
    c.get_pair_quote = AsyncMock(return_value="ZEUR")
    c.get_price = AsyncMock(return_value=1.16)  # EURUSD rate lookup
    vol = await c.size_for_usd("XBTEUR", 30.0, price=60000.0)
    assert abs(vol - 30.0 / (60000.0 * 1.16)) < 1e-12


@pytest.mark.asyncio
async def test_usd_notional_converts_quote():
    c = KrakenSpotClient(MagicMock())
    c.get_pair_quote = AsyncMock(return_value="ZEUR")
    c.get_price = AsyncMock(return_value=1.16)
    # 1 unit at price 100 EUR = 116 USD
    assert abs(await c.usd_notional("XEUR", 1.0, price=100.0) - 116.0) < 1e-9


def _pillar(*, holding: bool):
    s = MagicMock()
    s.risk_tolerance = 50.0
    s.is_live = False                      # paper: skip live fill recording
    k = s.kraken
    k.directional_pairs = ["XBTUSDC"]
    k.directional_entry_momentum_pct = 2.0
    k.directional_exit_momentum_pct = 4.0
    k.directional_momentum_pct = 3.0       # legacy fallback (unused when split set)
    k.directional_budget_usd = 600.0
    k.max_order_usd = 30.0
    # Numeric exit params so _exit_reason isn't evaluated against MagicMocks.
    k.directional_stop_loss_pct = 12.0
    k.directional_take_profit_pct = 0.0
    k.directional_trailing_stop_pct = 0.0
    k.directional_fee_pct = 0.26
    k.directional_reentry_cooldown_min = 0.0
    k.directional_liquidate_orphans = True
    client = MagicMock()
    # _directional sizes off FREE balance; exits read the reconciled held qty.
    bal = {"XBT": 0.5, "USDC": 1000.0} if holding else {"USDC": 1000.0}
    client.get_balance = AsyncMock(return_value=bal)
    client.get_free_balance = AsyncMock(return_value=bal)
    client.get_price = AsyncMock(return_value=100.0)
    client.get_pair_quote = AsyncMock(return_value="USDC")
    client.size_for_usd = AsyncMock(return_value=0.3)   # USD-aware volume
    client.usd_notional = AsyncMock(return_value=30.0)
    client.place_spot_order = AsyncMock(return_value=OrderResult(
        order_id="OK", market_id="XBTUSDC", status="filled", is_paper=False))
    p = KrakenPillar(settings=s, kraken_client=client)
    # Pre-resolve pair metadata so _directional skips the catalog fetch and
    # _managed_pairs treats XBT as a configured (non-orphan) base.
    p._pair_base = {"XBTUSDC": "XBT"}
    p._valid_pairs = ["XBTUSDC"]
    p._pair_min = {"XBTUSDC": 0.0}
    p._pair_lot_dec = {"XBTUSDC": 8}
    p._base_to_pair = {"XBT": "XBTUSDC"}
    # Reconcile is covered by its own tests; here return a fixed held book so the
    # exit path sizes off the actual qty without re-reading balances.
    held = {"XBTUSDC": (90.0, 100.0, 0.5)} if holding else {}
    p._reconcile_positions = AsyncMock(return_value=held)
    if holding:
        p._dir_long = {"XBTUSDC": 90.0}
    return p, client


@contextlib.contextmanager
def _stable_budget():
    # Isolate from the risk-tolerance budget scaler (identity at neutral).
    with patch("auramaur.risk.tolerance.scale_budget", lambda b, t: b), \
         patch("auramaur.risk.tolerance.current_tolerance", lambda s: 50.0):
        yield


@pytest.mark.asyncio
async def test_enters_long_at_entry_threshold():
    """+2.5% momentum (>= 2% entry) opens a long."""
    p, client = _pillar(holding=False)
    p._momentum = AsyncMock(return_value=2.5)
    with _stable_budget():
        await p._directional()
    client.place_spot_order.assert_awaited_once()
    assert client.place_spot_order.await_args.args[1] == OrderSide.BUY
    assert "XBTUSDC" in p._dir_long


@pytest.mark.asyncio
async def test_no_entry_below_entry_threshold():
    """+1.5% (< 2% entry) does not open a position."""
    p, client = _pillar(holding=False)
    p._momentum = AsyncMock(return_value=1.5)
    with _stable_budget():
        await p._directional()
    client.place_spot_order.assert_not_called()


@pytest.mark.asyncio
async def test_holds_through_moderate_drop():
    """Asymmetric bias: -3.5% does NOT exit (needs <= -4%) — winner rides on.
    The old symmetric -3% rule would have sold here."""
    p, client = _pillar(holding=True)
    p._momentum = AsyncMock(return_value=-3.5)
    with _stable_budget():
        await p._directional()
    client.place_spot_order.assert_not_called()
    assert "XBTUSDC" in p._dir_long


@pytest.mark.asyncio
async def test_mirror_to_portfolio_upserts_held_and_deletes_closed():
    """The spec book is reflected into the portfolio table for dashboard
    visibility: held pairs upsert with unrealized P&L, closed pairs are deleted."""
    s = MagicMock()
    s.is_live = True
    db = AsyncMock()
    bot = MagicMock()
    bot._components = {"db": db}
    p = KrakenPillar(settings=s, kraken_client=MagicMock(), bot=bot)

    await p._mirror_to_portfolio({"XBTUSDC": (90.0, 100.0, 0.5)}, ["ETHUSDC"])

    sqls = [c.args[0] for c in db.execute.await_args_list]
    assert any("INSERT INTO portfolio" in q and "kraken" in q for q in sqls)
    assert any("DELETE FROM portfolio" in q for q in sqls)
    db.commit.assert_awaited_once()
    # unrealized_pnl = (100-90)*0.5 = 5.0 passed for the held pair
    insert_args = next(c.args[1] for c in db.execute.await_args_list
                       if "INSERT INTO portfolio" in c.args[0])
    assert 5.0 in insert_args


@pytest.mark.asyncio
async def test_mirror_to_portfolio_noop_without_bot():
    """No bot/db wired -> mirror is a safe no-op."""
    s = MagicMock()
    p = KrakenPillar(settings=s, kraken_client=MagicMock(), bot=None)
    await p._mirror_to_portfolio({"XBTUSDC": (90.0, 100.0, 0.5)}, [])  # must not raise


@pytest.mark.asyncio
async def test_exits_on_large_drop():
    """-4.5% (<= -4% exit) closes the long."""
    p, client = _pillar(holding=True)
    p._momentum = AsyncMock(return_value=-4.5)
    with _stable_budget():
        await p._directional()
    client.place_spot_order.assert_awaited_once()
    assert client.place_spot_order.await_args.args[1] == OrderSide.SELL
    assert "XBTUSDC" not in p._dir_long
