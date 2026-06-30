"""Tests for the Kalshi informed-flow follower pillar.

Locks in: it follows the detected informed side (BUY yes / SELL = buy no), is
paper-forced, fetches a tape only for eligible markets, skips when there's no
signal or the market is ineligible, one-shot per market, and respects risk.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from auramaur.broker.pnl import PnLTracker
from auramaur.db.database import Database
from auramaur.exchange.models import (
    Market,
    Order,
    OrderResult,
    OrderSide,
    OrderType,
    TokenType,
)
from auramaur.strategy.informed_flow_pillar import InformedFlowPillar
from config.settings import Settings


def _market(mid="K1", ticker="KXTEST", yes=0.50, liquidity=5000.0, category="economics",
            active=True, days_out=10.0, exchange="kalshi") -> Market:
    return Market(
        id=mid, exchange=exchange, ticker=ticker, question=f"q-{mid}",
        category=category, active=active, outcome_yes_price=yes,
        outcome_no_price=round(1 - yes, 2), liquidity=liquidity, volume=10000.0,
        end_date=datetime.now(timezone.utc) + timedelta(days=days_out),
        clob_token_yes="ty", clob_token_no="tn",
    )


def _trades(yes_big=0, no_big=0, small=30, small_size=5, big_size=60):
    out = [{"count": small_size, "taker_side": "yes"} for _ in range(small)]
    out += [{"count": small_size, "taker_side": "no"} for _ in range(small)]
    out += [{"count": big_size, "taker_side": "yes"} for _ in range(yes_big)]
    out += [{"count": big_size, "taker_side": "no"} for _ in range(no_big)]
    return out


def _settings(**overrides) -> Settings:
    s = Settings()
    s.informed_flow.enabled = True
    s.informed_flow.paper = True
    for k, v in overrides.items():
        setattr(s.informed_flow, k, v)
    return s


def _exchange(trades=None, filled=True):
    ex = MagicMock()
    ex.get_trades = AsyncMock(return_value=trades if trades is not None else [])

    def prepare_order(signal, market, size, is_live):
        token = TokenType.NO if signal.recommended_side == OrderSide.SELL else TokenType.YES
        price = market.outcome_yes_price if token == TokenType.YES else 1 - market.outcome_yes_price
        return Order(
            market_id=market.id, token_id="tok", side=OrderSide.BUY, token=token,
            size=round(size / price, 2), price=round(price, 2),
            order_type=OrderType.LIMIT, dry_run=not is_live,
        )

    ex.prepare_order = MagicMock(side_effect=prepare_order)
    ex.place_order = AsyncMock(side_effect=lambda order: OrderResult(
        order_id="ord-1", market_id=order.market_id,
        status="paper" if order.dry_run else "filled",
        filled_size=order.size if filled else 0, filled_price=order.price,
        is_paper=order.dry_run,
    ))
    return ex


def _risk(approved=True, size=8.0):
    rm = MagicMock()
    d = MagicMock()
    d.approved = approved
    d.position_size = size if approved else 0.0
    d.reason = "" if approved else "blocked"
    d.force_paper = False
    rm.evaluate = AsyncMock(return_value=d)
    return rm


def _pillar(db, settings, markets, exchange, risk=None):
    disc = MagicMock()
    disc.get_markets = AsyncMock(return_value=markets)
    # The pillar scans the near-dated close-time window via this method.
    disc.get_markets_by_close_window = AsyncMock(return_value=markets)
    cal = MagicMock()
    cal.record_prediction = AsyncMock()
    return InformedFlowPillar(
        db=db, settings=settings, kalshi_discovery=disc, exchange=exchange,
        risk_manager=risk or _risk(), pnl_tracker=PnLTracker(db, settings),
        calibration=cal,
    ), cal


def test_follows_abnormal_yes_flow_as_buy():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ex = _exchange(trades=_trades(yes_big=4))   # abnormal YES flow
        pillar, cal = _pillar(db, _settings(), [_market(yes=0.50)], ex)
        assert await pillar.run_once() == 1

        ex.get_trades.assert_awaited_once()           # tape pulled for the market
        assert ex.prepare_order.call_args[0][3] is False  # paper-forced
        sig = await db.fetchone(
            "SELECT * FROM signals WHERE strategy_source='informed_flow'")
        assert sig["action"] == "BUY"
        assert abs(sig["claude_prob"] - 0.54) < 1e-9  # 0.50 + 0.04 uplift
        pos = await db.fetchone("SELECT * FROM portfolio WHERE market_id='K1'")
        assert pos["is_paper"] == 1 and pos["token"] == "YES"
        cal.record_prediction.assert_awaited_once()
        await db.close()

    asyncio.run(run())


def test_follows_abnormal_no_flow_as_sell():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ex = _exchange(trades=_trades(no_big=4))     # abnormal NO flow
        pillar, _ = _pillar(db, _settings(), [_market(yes=0.50)], ex)
        assert await pillar.run_once() == 1
        sig = await db.fetchone(
            "SELECT * FROM signals WHERE strategy_source='informed_flow'")
        assert sig["action"] == "SELL"
        assert abs(sig["claude_prob"] - 0.46) < 1e-9  # 0.50 - 0.04 uplift
        pos = await db.fetchone("SELECT token FROM portfolio WHERE market_id='K1'")
        assert pos["token"] == "NO"
        await db.close()

    asyncio.run(run())


def test_no_signal_no_entry():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ex = _exchange(trades=_trades())             # uniform — no abnormal flow
        pillar, _ = _pillar(db, _settings(), [_market(yes=0.50)], ex)
        assert await pillar.run_once() == 0
        ex.place_order.assert_not_awaited()
        await db.close()

    asyncio.run(run())


def test_ineligible_markets_skip_tape_fetch():
    """Cheap eligibility runs BEFORE the tape fetch — an extreme-priced /
    short-dated / thin / non-kalshi market never hits get_trades."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        ex = _exchange(trades=_trades(yes_big=4))
        markets = [
            _market("ext", yes=0.97),                 # outside band
            _market("short", yes=0.5, days_out=0.1),  # too soon
            _market("thin", yes=0.5, liquidity=10.0),  # illiquid
            _market("poly", yes=0.5, exchange="polymarket"),  # wrong venue
        ]
        pillar, _ = _pillar(db, _settings(), markets, ex)
        assert await pillar.run_once() == 0
        ex.get_trades.assert_not_awaited()            # no tape pulled for any
        await db.close()

    asyncio.run(run())


def test_one_entry_per_market():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ex = _exchange(trades=_trades(yes_big=4))
        settings = _settings()
        pillar, _ = _pillar(db, settings, [_market(yes=0.50)], ex)
        assert await pillar.run_once() == 1
        assert await pillar.run_once() == 0           # signal exists -> no re-entry
        await db.close()

    asyncio.run(run())


def test_risk_rejection_places_no_order():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ex = _exchange(trades=_trades(yes_big=4))
        pillar, _ = _pillar(db, _settings(), [_market(yes=0.50)], ex,
                            risk=_risk(approved=False))
        assert await pillar.run_once() == 0
        ex.place_order.assert_not_awaited()
        await db.close()

    asyncio.run(run())


def test_disabled_or_no_kalshi_noops():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ex = _exchange(trades=_trades(yes_big=4))
        pillar, _ = _pillar(db, _settings(enabled=False), [_market()], ex)
        assert await pillar.run_once() == 0
        await db.close()

    asyncio.run(run())


def test_kalshi_scale_liquidity_is_eligible():
    """A Kalshi-realistic market (liquidity ~60, far below the old 1000 Poly-scale
    floor) is now eligible — the floor that left informed_flow with zero markets
    is fixed."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        ex = _exchange(trades=_trades(yes_big=4))
        pillar, _ = _pillar(db, _settings(), [_market(yes=0.50, liquidity=60.0)], ex)
        assert await pillar.run_once() == 1     # would have been 0 at min_liquidity=1000
        ex.get_trades.assert_awaited_once()
        await db.close()

    asyncio.run(run())
