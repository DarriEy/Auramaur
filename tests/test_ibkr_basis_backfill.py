"""The legacy-position basis backfill.

The failure this repairs is silent and permanent: a SELL against an untracked
position books only its commission and the round trip's gross P&L is gone. So
these tests care most about idempotency (never restate an open position's cost)
and about the USD convention (a local-currency basis is 1000x wrong for FX).
"""

import pytest

from auramaur.broker.ibkr_basis_backfill import (
    apply_basis, load_existing_basis, plan_etf_basis, plan_multiasset_basis,
)
from auramaur.broker.pnl import PnLTracker
from auramaur.db.database import Database
from auramaur.exchange.models import Fill, OrderSide
from config.settings import Settings


def _pos(book, key, quantity, avg_cost, multiplier=1.0, fx=1.0):
    return {"book": book, "instrument_key": key, "quantity": quantity,
            "avg_cost": avg_cost, "multiplier": multiplier, "fx_to_usd": fx}


def test_fx_basis_is_stored_in_usd_not_local_currency():
    """record_fill has no multiplier or FX, so the basis must already be USD."""
    planned = plan_multiasset_basis(
        [_pos("fx", "USDJPY", 1.0, 163.098, multiplier=1000, fx=0.006133)], set())
    assert len(planned) == 1
    row = planned[0]
    assert row.market_id == "ibkr:fx:USDJPY"
    assert row.usd_avg_cost == pytest.approx(1000.2, abs=1.0)
    assert row.total_cost == pytest.approx(1000.2, abs=1.0)
    # The local-currency number that would have been booked instead.
    assert 163.098 != pytest.approx(row.usd_avg_cost, abs=1.0)


def test_unleveraged_position_keeps_its_price_as_basis():
    planned = plan_multiasset_basis(
        [_pos("global_etf", "UUP", 21.0304, 28.536)], set())
    assert planned[0].usd_avg_cost == pytest.approx(28.536)
    assert planned[0].total_cost == pytest.approx(600.1, abs=1.0)


def test_existing_basis_is_never_restated():
    """Re-running must be a no-op. Overwriting would silently change the cost
    of a position that is still open."""
    positions = [_pos("fx", "USDJPY", 1.0, 163.098, 1000, 0.006133)]
    assert plan_multiasset_basis(positions, {"ibkr:fx:USDJPY"}) == []
    assert plan_multiasset_basis(positions, set())


def test_closed_and_malformed_positions_are_skipped():
    assert plan_multiasset_basis([_pos("fx", "X", 0.0, 100.0)], set()) == []
    assert plan_multiasset_basis([_pos("fx", "X", 1.0, 0.0)], set()) == []


def test_etf_positions_are_cell_scoped():
    """Four arms trade the same symbols; a shared id would merge their bases."""
    planned = plan_etf_basis(
        [{"model_alias": "luna", "symbol": "SPY", "quantity": 2.5, "avg_cost": 100.02},
         {"model_alias": "sol", "symbol": "SPY", "quantity": 1.0, "avg_cost": 99.0}],
        set())
    assert [p.market_id for p in planned] == ["ibkr:luna:SPY", "ibkr:sol:SPY"]


@pytest.mark.asyncio
async def test_backfill_restores_the_pnl_a_legacy_exit_would_have_lost():
    """End to end: without a basis the sell books only the fee."""
    settings = Settings()

    async def _sell(db):
        tracker = PnLTracker(db, settings)
        await tracker.record_fill(Fill(
            market_id="ibkr:global_etf:UUP", token="YES", token_id="",
            side=OrderSide.SELL, size=21.0304, price=30.0, fee=1.0,
            is_paper=True, order_id="exit-1"))
        row = await db.fetchone(
            "SELECT COALESCE(SUM(pnl), 0) AS p FROM pnl_ledger "
            "WHERE market_id = 'ibkr:global_etf:UUP'")
        return float(row["p"])

    # No basis: the position size is 0, so the sell realizes only -fee.
    bare = Database(":memory:")
    await bare.connect()
    lost = await _sell(bare)
    assert lost == pytest.approx(-1.0)
    await bare.close()

    # With the backfill: real gross P&L on a $28.536 -> $30.00 move.
    healed = Database(":memory:")
    await healed.connect()
    planned = plan_multiasset_basis(
        [_pos("global_etf", "UUP", 21.0304, 28.536)],
        await load_existing_basis(healed))
    assert await apply_basis(healed, planned) == 1
    recovered = await _sell(healed)
    assert recovered == pytest.approx((30.0 - 28.536) * 21.0304 - 1.0, abs=0.01)
    assert recovered > 29.0
    # And a second run changes nothing.
    assert await apply_basis(
        healed, plan_multiasset_basis(
            [_pos("global_etf", "UUP", 21.0304, 28.536)],
            await load_existing_basis(healed))) == 0
    await healed.close()


@pytest.mark.asyncio
async def test_backfill_writes_a_markets_row_so_the_ledger_can_scope_venue():
    db = Database(":memory:")
    await db.connect()
    await apply_basis(db, plan_multiasset_basis(
        [_pos("fx", "USDCAD", 1.0, 1.408, 1000, 0.711)], set()))
    market = await db.fetchone("SELECT * FROM markets WHERE id='ibkr:fx:USDCAD'")
    assert market["exchange"] == "ibkr"
    assert market["active"] == 0
    assert market["category"] == "fx"
    await db.close()


@pytest.mark.asyncio
async def test_backfilled_rows_stay_out_of_the_shared_paper_wallet():
    from auramaur.exchange.paper import PaperTrader

    db = Database(":memory:")
    await db.connect()
    wallet = PaperTrader(db, initial_balance=1_000.0)
    await apply_basis(db, plan_multiasset_basis(
        [_pos("fx", "GBPJPY", 1.0, 218.305, 1000, 0.006133)], set()))
    # $1,338 of open basis must not become $1,338 of unspendable wallet cash.
    assert await wallet._compute_balance() == 1_000.0
    await db.close()
