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

from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.db.database import Database
from auramaur.exchange.ibkr_instruments import BY_KEY
from auramaur.exchange.ibkr_intent import (
    instrument_market,
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
