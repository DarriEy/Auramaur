"""Cross-venue semantic-equivalence arb: arb math + adversarial gate + entry."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from auramaur.broker.pnl import PnLTracker
from auramaur.db.database import Database
from auramaur.exchange.models import (
    Market, Order, OrderResult, OrderSide, OrderType, TokenType,
)
from auramaur.strategy.cross_venue_arb import CrossVenueArbPillar
from config.settings import Settings


def _market(mid, exchange, yes, q="Will the Fed cut rates at the July meeting?",
            liquidity=5000.0, days_out=20.0) -> Market:
    return Market(
        id=mid, exchange=exchange, question=q, active=True,
        outcome_yes_price=yes, outcome_no_price=round(1 - yes, 2),
        liquidity=liquidity, volume=liquidity, spread=0.01, category="economics",
        end_date=datetime.now(timezone.utc) + timedelta(days=days_out),
        clob_token_yes="ty", clob_token_no="tn",
    )


def _settings(**kw):
    s = Settings()
    s.cross_venue_arb.enabled = True
    s.cross_venue_arb.paper = True
    for k, v in kw.items():
        setattr(s.cross_venue_arb, k, v)
    return s


def _exchange(reject: bool = False):
    ex = MagicMock()

    def prepare_order(signal, market, size, is_live):
        token = TokenType.NO if signal.recommended_side == OrderSide.SELL else TokenType.YES
        price = market.outcome_yes_price if token == TokenType.YES else 1 - market.outcome_yes_price
        return Order(
            market_id=market.id, exchange=market.exchange, token_id="tok",
            side=OrderSide.BUY, token=token,
            size=round(size / max(price, 0.01), 2),
            price=round(price, 2), order_type=OrderType.LIMIT,
            dry_run=not is_live,
        )

    def place_order(o):
        if reject:
            return OrderResult(
                order_id=f"err-{o.market_id}", market_id=o.market_id,
                status="rejected", is_paper=o.dry_run,
                error_message="venue rejected",
            )
        return OrderResult(
            order_id=f"ord-{o.market_id}", market_id=o.market_id,
            status="paper" if o.dry_run else "filled",
            filled_size=o.size, filled_price=o.price, is_paper=o.dry_run,
        )

    ex.prepare_order = MagicMock(side_effect=prepare_order)
    ex.place_order = AsyncMock(side_effect=place_order)
    ex.cancel_order = AsyncMock(return_value=True)
    return ex


def _risk(approved=True, size=8.0):
    rm = MagicMock()
    d = MagicMock()
    d.approved = approved
    d.position_size = size if approved else 0.0
    d.reason = "" if approved else "blk"
    d.force_paper = False
    rm.evaluate = AsyncMock(return_value=d)
    return rm


def _analyzer(orientation="same", conf=0.95):
    a = MagicMock()
    a._call_llm = AsyncMock(return_value=(
        f'{{"orientation": "{orientation}", "confidence": {conf}, "counterexample": "none found"}}'))
    return a


def _pillar(db, settings, poly, kalshi, exchange=None, risk=None, analyzer=None,
            exchanges=None):
    disc = MagicMock(); disc.get_markets = AsyncMock(return_value=poly)
    kdisc = MagicMock(); kdisc.get_markets = AsyncMock(return_value=kalshi)
    exchanges = exchanges or {
        "polymarket": exchange or _exchange(),
        "kalshi": _exchange(),
    }
    return CrossVenueArbPillar(
        db=db, settings=settings, discovery=disc,
        exchange=exchanges.get("polymarket"), risk_manager=risk or _risk(),
        pnl_tracker=PnLTracker(db, settings),
        analyzer=analyzer if analyzer is not None else _analyzer(),
        kalshi_discovery=kdisc,
        exchanges=exchanges)


# -- arb math (pure) ---------------------------------------------------

def test_arb_math_same_orientation():
    s = _settings()
    p = _pillar(MagicMock(), s, [], [])
    a = _market("p", "polymarket", 0.40)
    b = _market("k", "kalshi", 0.55)
    edge, side_a, side_b = p._arb(a, b, "same")
    assert abs(edge - 0.15) < 1e-9
    # A_YES cheaper -> buy A_YES, sell (NO) the dearer B
    assert side_a == OrderSide.BUY and side_b == OrderSide.SELL


def test_arb_math_inverted_orientation():
    s = _settings()
    p = _pillar(MagicMock(), s, [], [])
    a = _market("p", "polymarket", 0.40)
    b = _market("k", "kalshi", 0.45)   # complementary, sum 0.85 < 1 -> both YES
    edge, side_a, side_b = p._arb(a, b, "inverted")
    assert abs(edge - 0.15) < 1e-9
    assert side_a == OrderSide.BUY and side_b == OrderSide.BUY


# -- full cycle --------------------------------------------------------

def test_equivalent_mispriced_pair_enters_both_legs():
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            a = _market("p1", "polymarket", 0.40)
            b = _market("k1", "kalshi", 0.55)
            ex = _exchange()
            kex = _exchange()
            pillar = _pillar(db, _settings(), [a], [b], exchange=ex,
                             analyzer=_analyzer("same", 0.95),
                             exchanges={"polymarket": ex, "kalshi": kex})
            entered = await pillar.run_once()
            assert entered == 1
            # both legs placed
            assert ex.place_order.await_count == 1
            assert kex.place_order.await_count == 1
            row = await db.fetchone(
                "SELECT traded_at FROM cross_venue_verdicts WHERE poly_id='p1' AND kalshi_id='k1'")
            assert row["traded_at"] is not None
        finally:
            await db.close()
    asyncio.run(run())


def test_non_equivalent_pair_does_not_trade():
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            a = _market("p1", "polymarket", 0.40)
            b = _market("k1", "kalshi", 0.55)
            ex = _exchange()
            pillar = _pillar(db, _settings(), [a], [b], exchange=ex,
                             analyzer=_analyzer("none", 0.2))
            assert await pillar.run_once() == 0
            assert ex.place_order.await_count == 0
        finally:
            await db.close()
    asyncio.run(run())


def test_subfee_gap_does_not_trade():
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            # equivalent + confident, but YES prices nearly equal -> gap < fees+buffer
            a = _market("p1", "polymarket", 0.50)
            b = _market("k1", "kalshi", 0.505)
            ex = _exchange()
            pillar = _pillar(db, _settings(), [a], [b], exchange=ex,
                             analyzer=_analyzer("same", 0.95))
            assert await pillar.run_once() == 0
            assert ex.place_order.await_count == 0
        finally:
            await db.close()
    asyncio.run(run())


def test_no_kalshi_discovery_is_noop():
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            disc = MagicMock(); disc.get_markets = AsyncMock(return_value=[])
            pillar = CrossVenueArbPillar(
                db=db, settings=_settings(), discovery=disc, exchange=_exchange(),
                risk_manager=_risk(), pnl_tracker=PnLTracker(db, _settings()),
                analyzer=_analyzer(), kalshi_discovery=None)
            assert await pillar.run_once() == 0
        finally:
            await db.close()
    asyncio.run(run())


def test_uses_venue_specific_exchange_clients():
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            a = _market("p1", "polymarket", 0.40)
            b = _market("k1", "kalshi", 0.55)
            poly_ex = _exchange()
            kalshi_ex = _exchange()
            pillar = _pillar(
                db, _settings(), [a], [b],
                exchanges={"polymarket": poly_ex, "kalshi": kalshi_ex},
                analyzer=_analyzer("same", 0.95),
            )

            assert await pillar.run_once() == 1
            poly_order = poly_ex.place_order.await_args.args[0]
            kalshi_order = kalshi_ex.place_order.await_args.args[0]
            assert poly_order.exchange == "polymarket"
            assert kalshi_order.exchange == "kalshi"
        finally:
            await db.close()
    asyncio.run(run())


def test_second_leg_rejection_marks_partial_not_traded_and_does_not_retry():
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            a = _market("p1", "polymarket", 0.40)
            b = _market("k1", "kalshi", 0.55)
            poly_ex = _exchange()
            kalshi_ex = _exchange(reject=True)
            pillar = _pillar(
                db, _settings(), [a], [b],
                exchanges={"polymarket": poly_ex, "kalshi": kalshi_ex},
                analyzer=_analyzer("same", 0.95),
            )

            assert await pillar.run_once() == 0
            row = await db.fetchone(
                "SELECT traded_at, partial_at, last_error FROM cross_venue_verdicts "
                "WHERE poly_id='p1' AND kalshi_id='k1'")
            assert row["traded_at"] is None
            assert row["partial_at"] is not None
            assert row["last_error"] == "venue rejected"

            assert await pillar.run_once() == 0
            assert poly_ex.place_order.await_count == 1
            assert kalshi_ex.place_order.await_count == 1
        finally:
            await db.close()
    asyncio.run(run())
