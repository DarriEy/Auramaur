"""An IBKR instrument routed through the ExecutionGateway contract.

The IBKR books sit outside the graduation ladder: strategy_experiments held 17
prospective clocks on 2026-07-27 and none was an IBKR book, because neither
pillar references ExecutionGateway, gateway.submit or record_fill. The ladder
reads pnl_ledger and decision_snapshots; those books write to neither.

The fix is not a gateway refactor. The gateway reads six fields off an intent
and places through exchange.place_order — a protocol method the IBKR clients
implement. These tests pin the adapter that supplies those six fields, and that
an instrument survives the gateway's prediction-market couplings.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.db.database import Database
from auramaur.exchange.ibkr_instruments import BY_KEY
from auramaur.exchange.ibkr_intent import (
    INSTRUMENT_ID_PREFIX,
    SimulatedInstrumentExchange,
    instrument_market,
    instrument_market_id,
    instrument_signal,
    prepare_instrument_order,
)
from auramaur.exchange.models import OrderResult, OrderSide
from config.settings import Settings

_SPEC = BY_KEY["UUP"]


def test_adapter_supplies_exactly_what_the_gateway_reads():
    market = instrument_market(_SPEC, mark=28.57)
    signal = instrument_signal(
        _SPEC, strategy_source="ibkr_global_etf_paper", mark=28.57,
        fair=None, side=OrderSide.BUY)

    # The six fields, and nothing about them is a placeholder.
    assert market.id == "ibkr:UUP"
    assert market.category == _SPEC.asset_class
    assert market.neg_risk_market_id in ("", None)   # family falls back to id
    assert signal.strategy_source == "ibkr_global_etf_paper"
    assert signal.market_prob == 28.57

    # A price book makes no forecast, so the signal must not assert an edge —
    # the ladder's Brier bar would otherwise score a forecast never made.
    assert signal.claude_prob == signal.market_prob
    assert signal.edge == 0.0


def test_order_is_placeable_and_defaults_to_paper():
    order = prepare_instrument_order(
        _SPEC, side=OrderSide.BUY, quantity=21.0, price=28.57,
        is_live=False, strategy_source="ibkr_global_etf_paper")
    assert order.market_id == "ibkr:UUP"
    assert order.size == 21.0 and order.price == 28.57
    assert order.dry_run is True
    # Quantity is contracts/shares, never re-derived from a dollar stake.
    assert order.size == 21.0

    assert prepare_instrument_order(
        _SPEC, side=OrderSide.BUY, quantity=0, price=28.57, is_live=False,
        strategy_source="x") is None
    assert prepare_instrument_order(
        _SPEC, side=OrderSide.BUY, quantity=1, price=0, is_live=False,
        strategy_source="x") is None


def test_instrument_survives_the_gateway_and_is_captured():
    """The point of the exercise: submit() books the fill AND captures a
    decision, which is what registers a prospective graduation clock."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            order = prepare_instrument_order(
                _SPEC, side=OrderSide.BUY, quantity=21.0, price=28.57,
                is_live=False, strategy_source="ibkr_global_etf_paper")
            result = OrderResult(order_id="ib1", market_id=order.market_id,
                                 status="paper", filled_size=21.0,
                                 filled_price=28.57, is_paper=True)
            exchange = SimpleNamespace(
                place_order=AsyncMock(return_value=result),
                prepare_order=lambda sig, mkt, size, live: order,
            )
            gw = ExecutionGateway(
                router=None, exchange=exchange, exchange_name="ibkr",
                settings=Settings(), db=db, pnl_tracker=None)

            intent = TradeIntent(
                signal=instrument_signal(
                    _SPEC, strategy_source="ibkr_global_etf_paper", mark=28.57,
                    fair=None, side=OrderSide.BUY),
                market=instrument_market(_SPEC, mark=28.57),
                size_dollars=600.0,
            )
            res = await gw.submit(intent)
            assert res.status in ("paper", "filled"), res.reason

            # A decision snapshot is the thing the ladder counts as prospective
            # evidence; without it an IBKR book can never start a clock.
            rows = await db.fetchall(
                "SELECT strategy_source, venue FROM decision_snapshots")
            assert rows, "no decision captured — no graduation clock"
            assert rows[0]["strategy_source"] == "ibkr_global_etf_paper"
            assert rows[0]["venue"] == "ibkr"
        finally:
            await db.close()
    asyncio.run(run())


def test_prediction_market_cap_is_not_weakened_for_ibkr():
    """The venue split must not touch the prediction-market ceiling.

    That guard exists because stacked sub-cap entries reached ~$90 against a
    documented $25 limit. Giving IBKR a broker-sized envelope must not become a
    global raise.
    """
    settings = Settings()
    gw = ExecutionGateway(router=None, exchange=SimpleNamespace(),
                          exchange_name="polymarket", settings=settings,
                          db=None, pnl_tracker=None)

    poly = prepare_instrument_order(
        _SPEC, side=OrderSide.BUY, quantity=1, price=1.0, is_live=False,
        strategy_source="llm")
    poly = poly.model_copy(update={"exchange": "polymarket"})
    assert gw._per_market_cap(poly) == settings.risk.max_stake_abs_ceiling

    ib = prepare_instrument_order(
        _SPEC, side=OrderSide.BUY, quantity=1, price=1.0, is_live=False,
        strategy_source="ibkr_global_etf_paper")
    assert gw._per_market_cap(ib) == settings.ibkr.paper_budget_usd
    assert gw._per_market_cap(ib) > gw._per_market_cap(poly)


def test_cell_scoping_keeps_concurrent_arms_off_one_cost_basis_row():
    """cost_basis is keyed by (market_id, is_paper, token).

    Four ETF arms trade the SAME symbols concurrently. Sharing one market_id
    per symbol would merge their basis rows: realized P&L computed against a
    blended average, the aggregate exposure guard counting four arms as one
    position, and the ledger's earliest-entrant attribution booking one arm's
    exit into another arm's graduation cell.
    """
    assert instrument_market_id(_SPEC) == "ibkr:UUP"
    assert instrument_market_id(_SPEC, "luna") == "ibkr:luna:UUP"
    assert instrument_market_id(_SPEC, "luna") != instrument_market_id(_SPEC, "sol")
    # Every minted id stays inside the one namespace the shared paper wallet
    # and the resolution tracker scope themselves off.
    for cell in ("", "luna", "fx"):
        assert instrument_market_id(_SPEC, cell).startswith(INSTRUMENT_ID_PREFIX)
    assert instrument_market(_SPEC, mark=28.57, cell="luna").id == "ibkr:luna:UUP"
    assert instrument_signal(
        _SPEC, strategy_source="s", mark=28.57, fair=None,
        side=OrderSide.BUY, cell="luna").market_id == "ibkr:luna:UUP"
    assert prepare_instrument_order(
        _SPEC, side=OrderSide.BUY, quantity=1, price=1.0, is_live=False,
        strategy_source="s", cell="luna").market_id == "ibkr:luna:UUP"


def test_the_simulator_is_not_an_order_path():
    """PAPER_SIMULATED means "no order path exists, AT ANY GATE".

    Routing a structurally paper-only book through the gateway must not
    quietly give it one. The simulator has no connection and refuses to carry
    an order whose dry_run has been cleared, so the gateway's place_order call
    terminates here — it never becomes a live broker order.
    """
    sim = SimulatedInstrumentExchange()
    assert not hasattr(sim, "connect")
    order = prepare_instrument_order(
        _SPEC, side=OrderSide.BUY, quantity=21.0, price=28.57,
        is_live=False, strategy_source="ibkr_etf_luna", cell="luna")
    sim.stage(order, order_id="fill-ref-1")
    result = asyncio.run(sim.place_order(order))
    assert result.status == "paper" and result.is_paper is True
    assert result.order_id == "fill-ref-1"   # joins back to the venue-native row
    assert result.filled_size == 21.0 and result.filled_price == 28.57

    live = order.model_copy(update={"dry_run": False})
    with pytest.raises(RuntimeError):
        asyncio.run(sim.place_order(live))


@pytest.mark.asyncio
async def test_a_price_book_stores_a_probability_not_a_price():
    """decision_snapshots.fair_probability held 4334.45 for SHEL.L — the USD
    mark, in a column named for a probability.

    It degraded safely (fair == reference gives a Brier edge of exactly zero,
    the intended "no forecast claimed"), but nothing reading that table could
    interpret it. A price book passes fair=None and the neutral 0.5 stands.
    """
    from auramaur.db.database import Database
    from auramaur.exchange.ibkr_instruments import BY_KEY
    from auramaur.broker.instrument_booking import (
        InstrumentFill, book_instrument_fill,
    )
    from auramaur.exchange.ibkr_intent import SimulatedInstrumentExchange
    from auramaur.broker.execution_gateway import ExecutionGateway
    from auramaur.broker.pnl import PnLTracker
    from auramaur.exchange.models import OrderSide
    from config.settings import Settings

    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    sim = SimulatedInstrumentExchange()
    gw = ExecutionGateway(router=None, exchange=sim, exchange_name="ibkr",
                          settings=settings, db=db,
                          pnl_tracker=PnLTracker(db, settings))
    spec = BY_KEY["SHEL.L"]
    ok = await book_instrument_fill(db, gw, sim, InstrumentFill(
        spec=spec, cell="international_equity",
        strategy_source="ibkr_international_equity_paper", side=OrderSide.BUY,
        quantity=1.0, price=4334.45, fee_usd=1.0, fill_ref="ref-1",
        session_date="2026-07-28", bid=4330.0, ask=4340.0,
        fair=None, usd_per_point=0.0126, usd_capital_per_unit=54.6))
    assert ok

    snap = await db.fetchone("SELECT * FROM decision_snapshots")
    assert 0.0 <= snap["fair_probability"] <= 1.0, "a price is in a probability column"
    assert snap["fair_probability"] == snap["reference_price"]   # claims no edge
    # The BBO is recorded, so the fill can be judged executable rather than
    # stamped 'synthetic' and dropped by require_executable_fills.
    assert (await db.fetchone("SELECT * FROM orderbook_snapshots")) is not None
    # And the family advances with time rather than being the bare market_id.
    assert snap["event_family"].startswith("ibkr:international_equity:SHEL.L:")
    assert snap["event_family"] != snap["market_id"]
    await db.close()
