"""Econ-indicator pillar: prices Kalshi bins from FRED, trades mispricings."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from auramaur.broker.pnl import PnLTracker
from auramaur.db.database import Database
from auramaur.exchange.models import (
    Market, Order, OrderResult, OrderSide, TokenType,
)
from auramaur.strategy.econ_indicator import EconIndicatorPillar
from config.settings import Settings


def _bin(thr: float, yes: float, period="26NOV", series="KXU3") -> Market:
    return Market(
        id=f"{series}-{period}-T{thr}", exchange="kalshi",
        ticker=f"{series}-{period}-T{thr}", question="What will unemployment be?",
        description=f"Above {thr}%", active=True,
        outcome_yes_price=yes, outcome_no_price=round(1 - yes, 2),
        liquidity=500.0, volume=500.0, spread=0.02,
        end_date=datetime.now(timezone.utc) + timedelta(days=30),
    )


def _settings(**ov) -> Settings:
    s = Settings()
    s.econ_indicator.enabled = True
    s.econ_indicator.paper = True
    s.econ_indicator.series = ["KXU3"]
    s.econ_indicator.min_edge = 0.07
    s.fred_api_key = "test"
    for k, v in ov.items():
        setattr(s.econ_indicator, k, v)
    return s


def _exchange():
    ex = MagicMock()

    def prepare(signal, market, size, is_live):
        token = TokenType.NO if signal.recommended_side == OrderSide.SELL else TokenType.YES
        price = market.outcome_yes_price if token == TokenType.YES else 1 - market.outcome_yes_price
        return Order(market_id=market.id, exchange="kalshi", token_id="tok",
                     side=OrderSide.BUY, token=token, size=round(size / max(price, 0.01), 2),
                     price=round(price, 2), dry_run=not is_live)

    ex.prepare_order = MagicMock(side_effect=prepare)
    ex.place_order = AsyncMock(side_effect=lambda o: OrderResult(
        order_id="o1", market_id=o.market_id, status="paper",
        filled_size=o.size, filled_price=o.price, is_paper=o.dry_run))
    return ex


def _fred(values):
    """Oldest-first monthly observations ending today."""
    base = datetime.now(timezone.utc) - timedelta(days=30 * len(values))
    obs = [(base + timedelta(days=30 * i), v) for i, v in enumerate(values)]
    f = MagicMock()
    f.get_observations = AsyncMock(return_value=obs)
    return f


def _risk(approved=True):
    rm = MagicMock()
    d = MagicMock(approved=approved, position_size=10.0 if approved else 0.0,
                  reason="", force_paper=False)
    rm.evaluate = AsyncMock(return_value=d)
    return rm


def _pillar(db, settings, bins, fred_values):
    kalshi = MagicMock()
    kalshi.get_markets_by_series = AsyncMock(return_value=bins)
    cal = MagicMock(); cal.record_prediction = AsyncMock()
    return EconIndicatorPillar(
        db=db, settings=settings, kalshi_discovery=kalshi,
        fred_source=_fred(fred_values), exchange=_exchange(),
        risk_manager=_risk(), pnl_tracker=PnLTracker(db, settings),
        calibration=cal)


def test_enters_mispriced_bin_and_skips_fair():
    """Unemployment ~4.0%: model P(above 4.5) ~0.31. A bin priced 0.70 is a
    big SELL-NO edge -> enter; a bin priced ~0.31 is fair -> skip."""
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            vals = [4.0, 4.1, 3.9, 4.0, 4.1, 3.9, 4.0, 4.0]   # ~4.0, low vol
            bins = [_bin(4.5, 0.70), _bin(4.6, 0.31)]          # one rich, one fair
            pillar = _pillar(db, _settings(), bins, vals)
            entered = await pillar.run_once()
            assert entered == 1
            rows = await db.fetchall(
                "SELECT market_id, action FROM signals WHERE strategy_source='econ_indicator'")
            assert len(rows) == 1
            assert rows[0]["market_id"] == "KXU3-26NOV-T4.5"
            assert rows[0]["action"] == "SELL"               # market too high -> buy NO
        finally:
            await db.close()
    asyncio.run(run())


def test_disabled_or_thin_history_noops():
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            bins = [_bin(4.5, 0.70), _bin(4.6, 0.31)]
            # too little history -> no pricing
            pillar = _pillar(db, _settings(), bins, [4.0, 4.1])
            assert await pillar.run_once() == 0
            # disabled flag
            pillar2 = _pillar(db, _settings(enabled=False), bins, [4.0]*8)
            assert await pillar2.run_once() == 0
        finally:
            await db.close()
    asyncio.run(run())


def test_nearest_period_picks_soonest_live_with_two_bins():
    pillar = EconIndicatorPillar.__new__(EconIndicatorPillar)
    near = [_bin(4.5, 0.4, period="26JUL"), _bin(4.6, 0.3, period="26JUL")]
    far = [_bin(4.5, 0.4, period="26DEC"), _bin(4.6, 0.3, period="26DEC")]
    for m in near:
        m.end_date = datetime.now(timezone.utc) + timedelta(days=10)
    for m in far:
        m.end_date = datetime.now(timezone.utc) + timedelta(days=120)
    # a single-bin period must be ignored (need >=2 to order a ladder)
    lonely = [_bin(4.5, 0.4, period="26AUG")]
    lonely[0].end_date = datetime.now(timezone.utc) + timedelta(days=40)
    picked = pillar._nearest_period(near + far + lonely)
    assert picked is not None
    _release, members = picked
    assert {m.ticker for _v, m in members} == {b.ticker for b in near}
