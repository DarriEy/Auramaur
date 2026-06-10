"""Tests for the favorite-longshot bias harvest pillar.

Locks in the safety and correctness properties:
  1. Band/eligibility math — only favored sides in [band_lo, band_hi) on
     active, liquid, in-window Polymarket markets enter.
  2. PAPER-FORCING — with bias_harvest.paper=true the order is built with
     is_live=False (dry_run) even when the global mode is live.
  3. One entry per market, ever; never enters a market already held.
  4. Risk rejection is respected (no order placed).
  5. A fill writes the full rails: fills (PnLTracker), trades mirror with
     strategy_source, portfolio row (is_paper=1), calibration prediction.
  6. Blocked categories are excluded.
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
from auramaur.strategy.bias_harvest import BiasHarvestPillar
from config.settings import Settings


def _market(mid="m1", yes=0.85, liquidity=5000.0, category="tech",
            active=True, days_out=10.0, exchange="polymarket") -> Market:
    return Market(
        id=mid,
        exchange=exchange,
        question=f"q-{mid}",
        category=category,
        active=active,
        outcome_yes_price=yes,
        outcome_no_price=round(1 - yes, 2),
        liquidity=liquidity,
        volume=10000.0,
        end_date=datetime.now(timezone.utc) + timedelta(days=days_out),
        clob_token_yes="tok-yes",
        clob_token_no="tok-no",
    )


def _settings(**overrides) -> Settings:
    s = Settings()
    s.bias_harvest.enabled = True
    s.bias_harvest.paper = True
    s.bias_harvest.band_lo = 0.80
    s.bias_harvest.band_hi = 0.97
    s.bias_harvest.stake_usd = 10.0
    for k, v in overrides.items():
        setattr(s.bias_harvest, k, v)
    return s


def _exchange(filled=True):
    ex = MagicMock()

    def prepare_order(signal, market, size, is_live):
        token = TokenType.NO if signal.recommended_side == OrderSide.SELL else TokenType.YES
        price = market.outcome_yes_price if token == TokenType.YES else 1 - market.outcome_yes_price
        return Order(
            market_id=market.id,
            token_id="tok",
            side=OrderSide.BUY,
            token=token,
            size=round(size / price, 2),
            price=round(price, 2),
            order_type=OrderType.LIMIT,
            dry_run=not is_live,
        )

    ex.prepare_order = MagicMock(side_effect=prepare_order)
    ex.place_order = AsyncMock(side_effect=lambda order: OrderResult(
        order_id="ord-1",
        market_id=order.market_id,
        status="paper" if order.dry_run else "filled",
        filled_size=order.size if filled else 0,
        filled_price=order.price,
        is_paper=order.dry_run,
    ))
    return ex


def _risk(approved=True, size=8.0):
    rm = MagicMock()
    decision = MagicMock()
    decision.approved = approved
    decision.position_size = size if approved else 0.0
    decision.reason = "" if approved else "blocked"
    rm.evaluate = AsyncMock(return_value=decision)
    return rm


def _pillar(db, settings, markets, exchange=None, risk=None):
    discovery = MagicMock()
    discovery.get_markets = AsyncMock(return_value=markets)
    calibration = MagicMock()
    calibration.record_prediction = AsyncMock()
    return BiasHarvestPillar(
        db=db,
        settings=settings,
        discovery=discovery,
        exchange=exchange or _exchange(),
        risk_manager=risk or _risk(),
        pnl_tracker=PnLTracker(db, settings),
        calibration=calibration,
    ), calibration


def test_enters_band_market_and_writes_all_rails():
    async def run():
        db = Database(":memory:")
        await db.connect()
        settings = _settings()
        ex = _exchange()
        pillar, calibration = _pillar(db, settings, [_market(yes=0.85)], exchange=ex)

        entered = await pillar.run_once()
        assert entered == 1

        # paper-forced: prepare_order called with is_live=False
        assert ex.prepare_order.call_args[0][3] is False

        sig = await db.fetchone("SELECT * FROM signals WHERE strategy_source='bias_harvest'")
        assert sig is not None
        assert abs(sig["claude_prob"] - 0.89) < 1e-9  # 0.85 + 0.04 uplift
        assert sig["action"] == "BUY"

        trade = await db.fetchone("SELECT * FROM trades WHERE strategy_source='bias_harvest'")
        assert trade is not None and trade["is_paper"] == 1

        fill = await db.fetchone("SELECT * FROM fills WHERE market_id='m1'")
        assert fill is not None and fill["is_paper"] == 1

        pos = await db.fetchone("SELECT * FROM portfolio WHERE market_id='m1'")
        assert pos is not None and pos["is_paper"] == 1 and pos["token"] == "YES"

        calibration.record_prediction.assert_awaited_once()
        await db.close()

    asyncio.run(run())


def test_paper_false_passes_global_mode():
    """With paper=false, is_live follows the global mode (here: paper)."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        settings = _settings(paper=False)
        ex = _exchange()
        pillar, _ = _pillar(db, settings, [_market()], exchange=ex)
        await pillar.run_once()
        # Global mode in tests is paper (no AURAMAUR_LIVE), so still False.
        assert ex.prepare_order.call_args[0][3] == (settings.is_live and True)
        await db.close()

    asyncio.run(run())


def test_favored_no_side_enters_as_sell_signal():
    """YES at 0.12 -> favored side is NO at 0.88 -> SELL signal (buys NO)."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        pillar, _ = _pillar(db, _settings(), [_market(yes=0.12)])
        entered = await pillar.run_once()
        assert entered == 1
        sig = await db.fetchone("SELECT * FROM signals WHERE strategy_source='bias_harvest'")
        assert sig["action"] == "SELL"
        assert abs(sig["claude_prob"] - 0.08) < 1e-9  # 0.12 - 0.04 uplift
        pos = await db.fetchone("SELECT token FROM portfolio WHERE market_id='m1'")
        assert pos["token"] == "NO"
        await db.close()

    asyncio.run(run())


def test_skips_out_of_band_and_ineligible():
    async def run():
        db = Database(":memory:")
        await db.connect()
        markets = [
            _market("coin_flip", yes=0.55),            # band starts at 0.80
            _market("too_deep", yes=0.98),             # beyond band_hi
            _market("illiquid", yes=0.85, liquidity=100.0),
            _market("inactive", yes=0.85, active=False),
            _market("too_long", yes=0.85, days_out=200.0),
            _market("too_soon", yes=0.85, days_out=0.1),
            _market("blocked", yes=0.85, category="sports"),
            _market("kalshi_m", yes=0.85, exchange="kalshi"),
        ]
        pillar, _ = _pillar(db, _settings(), markets)
        assert await pillar.run_once() == 0
        await db.close()

    asyncio.run(run())


def test_one_entry_per_market_and_skips_held():
    async def run():
        db = Database(":memory:")
        await db.connect()
        settings = _settings()
        pillar, _ = _pillar(db, settings, [_market()])
        assert await pillar.run_once() == 1
        # Second scan: signal exists -> no re-entry.
        assert await pillar.run_once() == 0

        # A market held by ANY strategy is skipped.
        await db.execute(
            "INSERT INTO portfolio (market_id, exchange, side, size, avg_price, "
            "current_price, is_paper) VALUES ('held', 'polymarket', 'BUY', 5, 0.5, 0.5, 0)"
        )
        await db.commit()
        pillar2, _ = _pillar(db, settings, [_market("held", yes=0.85)])
        assert await pillar2.run_once() == 0
        await db.close()

    asyncio.run(run())


def test_risk_rejection_places_no_order():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ex = _exchange()
        pillar, _ = _pillar(db, _settings(), [_market()], exchange=ex,
                            risk=_risk(approved=False))
        assert await pillar.run_once() == 0
        ex.place_order.assert_not_awaited()
        # Signal is still recorded (visible in research), but no trade/position.
        assert await db.fetchone("SELECT 1 FROM trades") is None
        assert await db.fetchone("SELECT 1 FROM portfolio") is None
        await db.close()

    asyncio.run(run())


def test_stake_cap_applies():
    async def run():
        db = Database(":memory:")
        await db.connect()
        ex = _exchange()
        # Risk approves $50; config stake cap is $10.
        pillar, _ = _pillar(db, _settings(stake_usd=10.0), [_market(yes=0.85)],
                            exchange=ex, risk=_risk(size=50.0))
        await pillar.run_once()
        size_arg = ex.prepare_order.call_args[0][2]
        assert abs(size_arg - 10.0) < 1e-9
        await db.close()

    asyncio.run(run())
