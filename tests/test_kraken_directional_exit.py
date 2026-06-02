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


# ---------------------------------------------------------------------------
# Live-mode: realized P&L recording (#1) + entry-basis recovery (#2)
# ---------------------------------------------------------------------------

from auramaur.db.database import Database  # noqa: E402


def _live_pillar(db, query_fills):
    """Live pillar wired to a real in-memory DB; query_fills feeds query_fill."""
    settings = SimpleNamespace(kraken=_kcfg(), is_live=True)
    k = _client(1.0)
    k.query_fill = AsyncMock(side_effect=query_fills)
    bot = SimpleNamespace(_components={"db": db})
    return KrakenPillar(settings, k, bot=bot)


def test_directional_fills_record_realized_pnl():
    """A live BUY then SELL writes fills + realized P&L into cost_basis."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        p = _live_pillar(db, query_fills=[
            {"price": 1.00, "vol": 10.0, "fee": 0.026},   # entry fill
            {"price": 1.20, "vol": 10.0, "fee": 0.0312},  # exit fill
        ])
        res = SimpleNamespace(order_id="TX1")

        await p._record_directional_fill("APEUSDC", OrderSide.BUY, res, 1.00, 10.0)
        await p._record_directional_fill("APEUSDC", OrderSide.SELL, res, 1.20, 10.0)

        # Realized = (1.20 - 1.00) * 10 - sell_fee(0.0312) = 1.9688 (live → is_paper 0)
        row = await db.fetchone(
            "SELECT realized_pnl, size FROM cost_basis WHERE market_id='APEUSDC' AND is_paper=0")
        assert row is not None
        assert abs(row["realized_pnl"] - 1.9688) < 1e-6
        assert row["size"] == 0.0  # fully closed

        fills = await db.fetchone("SELECT COUNT(*) c FROM fills WHERE market_id='APEUSDC'")
        assert fills["c"] == 2
        await db.close()

    asyncio.run(run())


def test_paper_mode_records_nothing():
    """Paper orders are validate-only; no fills/cost_basis should be written."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        settings = SimpleNamespace(kraken=_kcfg(), is_live=False)
        bot = SimpleNamespace(_components={"db": db})
        p = KrakenPillar(settings, _client(1.0), bot=bot)

        out = await p._record_directional_fill(
            "APEUSDC", OrderSide.BUY, SimpleNamespace(order_id="VALIDATED"), 1.0, 10.0)
        assert out is None
        row = await db.fetchone("SELECT COUNT(*) c FROM fills")
        assert row["c"] == 0
        await db.close()

    asyncio.run(run())


def test_entry_basis_recovered_from_cost_basis_on_restart():
    """After a 'restart' (_dir_long cleared), reconcile recovers the true entry
    from cost_basis instead of resetting it to the current price."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        p = _live_pillar(db, query_fills=[{"price": 1.00, "vol": 10.0, "fee": 0.026}])
        # Establish a real entry basis of 1.00 via a recorded BUY fill.
        await p._record_directional_fill(
            "APEUSDC", OrderSide.BUY, SimpleNamespace(order_id="TX1"), 1.00, 10.0)

        # Simulate a restart: no in-memory entry, price has since risen to 1.50.
        p._dir_long = {}
        p._k.get_balance = AsyncMock(return_value={"APE": 10.0})
        p._k.get_price = AsyncMock(return_value=1.50)

        held_prices = await p._reconcile_positions({"APE": 10.0})

        entry, current, qty = held_prices["APEUSDC"]
        assert entry == 1.00     # recovered true basis, NOT reset to 1.50
        assert current == 1.50
        assert qty == 10.0
        # And a stop-loss now measures from the real entry: 1.50 vs 1.00 = +50%,
        # correctly NOT stopped (a reset to 1.50 would have hidden the real gain).
        assert p._dir_long["APEUSDC"] == 1.00
        await db.close()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Exit-reason priority, fee-netting, trailing stop (#3/#4, #6) + cooldown (#7)
# ---------------------------------------------------------------------------

def _exit_kcfg(**over):
    base = dict(
        directional_stop_loss_pct=12.0,
        directional_take_profit_pct=10.0,
        directional_trailing_stop_pct=8.0,
        directional_fee_pct=0.26,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _reason(mom, gain, peak, exit_thr=4.0, **over):
    return KrakenPillar._exit_reason(mom, gain, peak, exit_thr, _exit_kcfg(**over))


def test_exit_reason_stop_loss_wins():
    # Down past the stop — even if momentum is calm, cut the loser first.
    assert _reason(mom=0.0, gain=-15.0, peak=-15.0) == "stop_loss"


def test_exit_reason_take_profit_is_net_of_fees():
    # Gross +11 with round-trip fee 0.52 -> net 10.48 >= 10 target.
    assert _reason(mom=0.0, gain=11.0, peak=11.0) == "take_profit"
    # Gross +10.3 -> net 9.78 < 10: NOT a take-profit (fees fold into the bar).
    assert _reason(mom=0.0, gain=10.3, peak=10.3) is None


def test_exit_reason_trailing_protects_a_winner():
    # Peaked +20, now +11 -> gave back 9 >= 8 trailing.
    assert _reason(mom=0.0, gain=11.0, peak=20.0, directional_take_profit_pct=0.0) == "trailing_stop"
    # Give-back only 5 (<8): still holding.
    assert _reason(mom=0.0, gain=15.0, peak=20.0, directional_take_profit_pct=0.0) is None


def test_exit_reason_trailing_ignored_when_never_in_profit():
    # Peak barely positive (< round-trip fee): not a real winner, no trailing.
    assert _reason(mom=0.0, gain=-3.0, peak=0.3,
                   directional_stop_loss_pct=0.0, directional_take_profit_pct=0.0) is None


def test_exit_reason_momentum_is_last_resort():
    assert _reason(mom=-5.0, gain=2.0, peak=2.0,
                   directional_take_profit_pct=0.0, directional_trailing_stop_pct=0.0) == "momentum"


def test_exit_reason_hold_when_calm():
    assert _reason(mom=-1.0, gain=2.0, peak=3.0) is None


def test_reentry_cooldown_blocks_then_clears():
    async def run():
        kcfg = _kcfg()
        kcfg.directional_reentry_cooldown_min = 30.0
        p = _pillar(1.0, kcfg)
        assert not p._in_cooldown("APEUSDC")
        p._set_cooldown("APEUSDC")
        assert p._in_cooldown("APEUSDC")
        # A zero cooldown config never gates.
        kcfg.directional_reentry_cooldown_min = 0.0
        p._set_cooldown("XBTUSDC")
        assert not p._in_cooldown("XBTUSDC")

    asyncio.run(run())


def test_resolve_pairs_prunes_unknown():
    """Pairs Kraken doesn't list are dropped from the working set (no per-cycle
    'Unknown asset pair' spam), validated against the full catalog."""
    async def run():
        kcfg = _kcfg()
        kcfg.directional_pairs = ["APEUSDC", "FAKEPAIR", "MANAUSDC"]
        p = _pillar(1.0, kcfg)
        # Catalog returns only APE and MANA (keyed by Kraken internal name).
        p._k._public = AsyncMock(return_value={
            "XAPEZUSD": {"altname": "APEUSDC", "base": "APE", "ordermin": "1"},
            "MANAUSD": {"altname": "MANAUSDC", "base": "MANA", "ordermin": "5"},
        })
        await p._resolve_pairs(kcfg.directional_pairs)

        assert p._valid_pairs == ["APEUSDC", "MANAUSDC"]  # FAKEPAIR pruned
        assert p._pair_base == {"APEUSDC": "APE", "MANAUSDC": "MANA"}
        assert p._pair_min["MANAUSDC"] == 5.0

    asyncio.run(run())


def test_resolve_pairs_catalog_error_falls_back_to_all():
    async def run():
        kcfg = _kcfg()
        kcfg.directional_pairs = ["APEUSDC", "MANAUSDC"]
        p = _pillar(1.0, kcfg)
        p._k._public = AsyncMock(side_effect=RuntimeError("kraken down"))
        await p._resolve_pairs(kcfg.directional_pairs)
        # On a catalog fetch failure, keep all pairs (per-call errors absorb).
        assert p._valid_pairs == ["APEUSDC", "MANAUSDC"]

    asyncio.run(run())


def test_peak_tracks_high_water_mark():
    """Trailing-stop foundation: peak only ratchets up, persists, clears on exit."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        p = _live_pillar(db, query_fills=[])
        assert await p._update_and_get_peak("APEUSDC", 5.0) == 5.0
        assert await p._update_and_get_peak("APEUSDC", 12.0) == 12.0
        # A pullback does NOT lower the high-water mark.
        assert await p._update_and_get_peak("APEUSDC", 7.0) == 12.0
        await p._clear_peak("APEUSDC")
        # After an exit clears it, the peak resets to the next observation.
        assert await p._update_and_get_peak("APEUSDC", 3.0) == 3.0
        await db.close()

    asyncio.run(run())


if __name__ == "__main__":
    test_stop_loss_exits_and_sells_actual_held_qty()
    test_no_exit_when_neither_momentum_nor_stop_triggers()
    test_stop_disabled_falls_back_to_momentum_only()
    test_momentum_reversal_still_exits_with_actual_qty()
    test_directional_fills_record_realized_pnl()
    test_paper_mode_records_nothing()
    test_entry_basis_recovered_from_cost_basis_on_restart()
    print("ok")
