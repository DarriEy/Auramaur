"""Kraken directional (spec) exit wiring.

Two gaps were fixed:
- Gap A: the exit recomputed a USD notional instead of selling the actual held
  quantity, so ordermin-bumped / fee-trimmed positions under-sold (residual) or
  over-sold (insufficient-funds reject). It now sells the reconciled balance.
- Gap B: momentum-reversal was the only exit (no floor under a loser). A hard
  stop-loss (directional_stop_loss_pct) now exits regardless of momentum.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from auramaur.exchange.models import OrderSide
from auramaur.treasury.kraken_pillar import KrakenPillar


def _kcfg(**over):
    base = dict(
        directional_enabled=True,
        directional_pairs=["APEUSDC"],
        directional_momentum_pct=3.0,
        directional_entry_momentum_pct=2.0,
        directional_exit_momentum_pct=4.0,
        directional_stop_loss_pct=12.0,
        directional_lookback=12,
        directional_budget_usd=50.0,
        max_order_usd=25.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _client(price, *, base_amt=10.0):
    """Fake KrakenSpotClient. base asset APE; current `price` for APEUSDC."""
    k = SimpleNamespace()
    k.get_balance = AsyncMock(return_value={"APE": base_amt})
    k.get_price = AsyncMock(return_value=price)
    k._public = AsyncMock(return_value={
        "APEUSDC": {"altname": "APEUSDC", "base": "APE", "ordermin": "1"},
    })
    k.usd_notional = AsyncMock(side_effect=lambda pair, vol, px=None: vol * (px or price))
    k.place_spot_order = AsyncMock(return_value=SimpleNamespace(
        order_id="OK", status="filled", error_message=""))
    k.size_for_usd = AsyncMock(return_value=999.0)  # must NOT be used by the exit
    return k


def _pillar(price, kcfg, base_amt=10.0):
    settings = SimpleNamespace(kraken=kcfg, is_live=False)
    p = KrakenPillar(settings, _client(price, base_amt=base_amt), bot=None)
    # Pre-seed an open long at entry 1.00 (a held position from a prior cycle).
    p._dir_long = {"APEUSDC": 1.00}
    return p


def test_stop_loss_exits_and_sells_actual_held_qty():
    async def run():
        kcfg = _kcfg()
        # Down 20% from entry (1.00 -> 0.80): below the 12% stop. Momentum is
        # only mildly negative (-1%), so the momentum exit (-4%) would NOT fire —
        # only the stop-loss should.
        p = _pillar(0.80, kcfg, base_amt=10.0)
        p._momentum = AsyncMock(return_value=-1.0)

        await p._directional()

        p._k.place_spot_order.assert_called_once()
        _, kwargs = p._k.place_spot_order.call_args
        args = p._k.place_spot_order.call_args.args
        assert OrderSide.SELL in args  # SELL side
        # Sold the ACTUAL held qty (10.0), not the recomputed size_for_usd notional.
        assert kwargs["volume"] == 10.0
        # Cap headroom passed so an exit is never blocked by the per-order cap.
        assert kwargs["max_usd"] > 0
        p._k.size_for_usd.assert_not_called()
        # Position cleared on a successful exit.
        assert "APEUSDC" not in p._dir_long

    asyncio.run(run())


def test_no_exit_when_neither_momentum_nor_stop_triggers():
    async def run():
        kcfg = _kcfg()
        # Down only 5% (1.00 -> 0.95): above the 12% stop. Momentum -1% > -4%.
        p = _pillar(0.95, kcfg, base_amt=10.0)
        p._momentum = AsyncMock(return_value=-1.0)

        await p._directional()

        p._k.place_spot_order.assert_not_called()
        assert "APEUSDC" in p._dir_long  # still held

    asyncio.run(run())


def test_stop_disabled_falls_back_to_momentum_only():
    async def run():
        kcfg = _kcfg(directional_stop_loss_pct=0.0)
        # Down 20% but stop disabled, momentum mild -> no exit.
        p = _pillar(0.80, kcfg, base_amt=10.0)
        p._momentum = AsyncMock(return_value=-1.0)

        await p._directional()

        p._k.place_spot_order.assert_not_called()
        assert "APEUSDC" in p._dir_long

    asyncio.run(run())


def test_momentum_reversal_still_exits_with_actual_qty():
    async def run():
        kcfg = _kcfg()
        # Flat price (no stop) but strong negative momentum (-5% <= -4% exit).
        p = _pillar(1.00, kcfg, base_amt=7.0)
        p._momentum = AsyncMock(return_value=-5.0)

        await p._directional()

        p._k.place_spot_order.assert_called_once()
        _, kwargs = p._k.place_spot_order.call_args
        assert kwargs["volume"] == 7.0  # actual held qty
        assert "APEUSDC" not in p._dir_long

    asyncio.run(run())


if __name__ == "__main__":
    test_stop_loss_exits_and_sells_actual_held_qty()
    test_no_exit_when_neither_momentum_nor_stop_triggers()
    test_stop_disabled_falls_back_to_momentum_only()
    test_momentum_reversal_still_exits_with_actual_qty()
    print("ok")
