"""Weather-temp pillar: price Poly temp bins from the ensemble, trade mispricings."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from auramaur.broker.pnl import PnLTracker
from auramaur.db.database import Database
from auramaur.exchange.models import Market, Order, OrderResult, OrderSide, TokenType
from auramaur.strategy.weather_temp import WeatherTempPillar
from config.settings import Settings


def _tmkt(mid, question, yes) -> Market:
    return Market(
        id=mid, exchange="polymarket", question=question,
        outcome_yes_price=yes, outcome_no_price=round(1 - yes, 2),
        liquidity=4000.0, volume=4000.0, category="weather",
        end_date=datetime.now(timezone.utc) + timedelta(days=1),
        clob_token_yes="ty", clob_token_no="tn")


def _settings(**ov):
    s = Settings()
    s.weather_temp.enabled = True
    s.weather_temp.paper = True
    s.weather_temp.min_edge = 0.10
    for k, v in ov.items():
        setattr(s.weather_temp, k, v)
    return s


def _exchange():
    ex = MagicMock()
    def prepare(signal, market, size, is_live):
        token = TokenType.NO if signal.recommended_side == OrderSide.SELL else TokenType.YES
        price = market.outcome_yes_price if token == TokenType.YES else 1 - market.outcome_yes_price
        return Order(market_id=market.id, exchange="polymarket", token_id="t",
                     side=OrderSide.BUY, token=token, size=round(size / max(price, .01), 2),
                     price=round(price, 2), dry_run=not is_live)
    ex.prepare_order = MagicMock(side_effect=prepare)
    ex.place_order = AsyncMock(side_effect=lambda o: OrderResult(
        order_id="o1", market_id=o.market_id, status="paper",
        filled_size=o.size, filled_price=o.price, is_paper=o.dry_run))
    return ex


def _risk():
    rm = MagicMock()
    rm.evaluate = AsyncMock(return_value=MagicMock(
        approved=True, position_size=10.0, reason="", force_paper=False))
    return rm


def _pillar(db, settings, markets, members):
    disc = MagicMock(); disc.get_markets = AsyncMock(return_value=markets)
    weather = MagicMock(); weather.daily_ensemble = AsyncMock(return_value=members)
    cal = MagicMock(); cal.record_prediction = AsyncMock()
    return WeatherTempPillar(db=db, settings=settings, discovery=disc,
                             exchange=_exchange(), risk_manager=_risk(),
                             pnl_tracker=PnLTracker(db, settings),
                             calibration=cal, weather=weather)


def test_enters_when_ensemble_disagrees_and_skips_fair():
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            # market says 0.27 for "Tokyo high = 23C"; ensemble puts ~3% in the bin
            mispriced = _tmkt("w1", "Will the highest temperature in Tokyo be 23°C on June 20?", 0.27)
            members = [26, 27, 28, 25, 29, 24, 30, 26, 27, 28]   # ~0 in [22.5,23.5)
            pillar = _pillar(db, _settings(), [mispriced], members)
            assert await pillar.run_once() == 1
            row = await db.fetchone(
                "SELECT action FROM signals WHERE strategy_source='weather_temp' AND market_id='w1'")
            assert row["action"] == "SELL"   # model << market -> buy NO
        finally:
            await db.close()
    asyncio.run(run())


def test_skips_fair_priced_bin():
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            # market 0.30; ensemble puts 3/10 in the bin -> ~fair -> skip
            fair = _tmkt("w2", "Will the highest temperature in Tokyo be 23°C on June 20?", 0.30)
            members = [23, 23, 23, 26, 27, 28, 25, 29, 24, 30]   # 3/10 in [22.5,23.5)
            pillar = _pillar(db, _settings(), [fair], members)
            assert await pillar.run_once() == 0
        finally:
            await db.close()
    asyncio.run(run())


def test_skips_implausible_divergence():
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            m = _tmkt("w3", "Will the highest temperature in Atlanta be between 82-83°F on June 19?", 0.50)
            members = [95.0] * 10   # model 0.0 vs market 0.50 -> 0.50 gap > cap
            pillar = _pillar(db, _settings(max_divergence=0.40), [m], members)
            assert await pillar.run_once() == 0
        finally:
            await db.close()
    asyncio.run(run())


@pytest.mark.asyncio
async def test_rejected_signal_does_not_burn_the_market_forever(tmp_path):
    """A signal without a fill must not blacklist the market.

    _persist_signal runs before the risk gate and signals rows are never
    deleted, so any rejection permanently excluded that market: 107 markets
    burned for this pillar on 2026-07-25 with zero trades to show for it.
    """
    from auramaur.db.database import Database
    from auramaur.strategy.weather_temp import WeatherTempPillar

    db = Database(str(tmp_path / "w.db"))
    await db.connect()
    try:
        pillar = WeatherTempPillar.__new__(WeatherTempPillar)
        pillar._db = db

        # Signal emitted, then rejected — no trade, no position.
        await db.execute(
            """INSERT INTO signals (market_id, claude_prob, claude_confidence,
               market_prob, edge, evidence_summary, action, strategy_source)
               VALUES ('BURNED', 0.8, 'MEDIUM', 0.7, 8.0, 'e', 'BUY',
                       'weather_temp')""")
        await db.commit()
        assert await pillar._already_entered_or_held("BURNED") is False

        # An actual fill still blocks re-entry.
        await db.execute(
            """INSERT INTO trades (market_id, side, size, price, is_paper,
               status, strategy_source, exchange)
               VALUES ('TRADED','BUY',10,0.5,1,'filled','weather_temp',
                       'polymarket')""")
        await db.commit()
        assert await pillar._already_entered_or_held("TRADED") is True

        # A closed position (size 0) no longer blocks.
        await db.execute(
            """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
               current_price, unrealized_pnl, category, token, token_id,
               is_paper, updated_at)
               VALUES ('CLOSED','polymarket','BUY',0,0.5,0.5,0,'weather','YES',
                       't', 1, datetime('now'))""")
        await db.commit()
        assert await pillar._already_entered_or_held("CLOSED") is False
    finally:
        await db.close()
