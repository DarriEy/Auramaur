"""Rejection cooldown: risk-rejected markets are benched from re-analysis.

Before this, a rejected market re-entered the candidate pool as soon as the
15-minute recently-analyzed window lapsed, so the same dud markets burned a
fresh evidence pass + LLM call every ~20 minutes all day — on Kalshi that
was the entire signal stream. The verdict can't flip until something moves:
bench until the cooldown expires, the market reprices, or news flags it.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock


from auramaur.db.database import Database
from auramaur.exchange.models import Market
from auramaur.strategy.engine import TradingEngine


def _market(market_id: str, yes_price: float = 0.20, exchange: str = "polymarket") -> Market:
    return Market(
        id=market_id,
        exchange=exchange,
        question=f"Q {market_id}?",
        outcome_yes_price=yes_price,
        outcome_no_price=1.0 - yes_price,
    )


async def _engine(db: Database, cooldown_min: int = 240, reprice: float = 0.03) -> TradingEngine:
    engine = TradingEngine.__new__(TradingEngine)
    engine.db = db
    engine.exchange_name = "polymarket"
    engine._news_flagged = {}
    settings = MagicMock()
    settings.nlp.rejection_cooldown_minutes = cooldown_min
    settings.nlp.rejection_reprice_threshold = reprice
    engine.settings = settings
    return engine


def test_rejection_benches_market_until_cooldown():
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            engine = await _engine(db)
            market = _market("m1", yes_price=0.20)

            await engine._record_rejection_state(market, approved=False, reason="Edge 0.5% below minimum")

            kept, benched = await engine._apply_rejection_cooldown([market])
            assert kept == []
            assert benched == 1
        finally:
            await db.close()

    asyncio.run(run())


def test_reprice_lifts_the_bench():
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            engine = await _engine(db, reprice=0.03)
            await engine._record_rejection_state(_market("m1", yes_price=0.20), approved=False, reason="r")

            moved = _market("m1", yes_price=0.24)  # 4 pts > 3 pt threshold
            kept, benched = await engine._apply_rejection_cooldown([moved])
            assert kept == [moved]
            assert benched == 0

            still = _market("m1", yes_price=0.21)  # 1 pt < threshold
            kept, benched = await engine._apply_rejection_cooldown([still])
            assert kept == []
            assert benched == 1
        finally:
            await db.close()

    asyncio.run(run())


def test_news_flag_lifts_the_bench():
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            engine = await _engine(db)
            market = _market("m1", yes_price=0.20)
            await engine._record_rejection_state(market, approved=False, reason="r")

            engine._news_flagged = {"m1": 0.0}
            kept, benched = await engine._apply_rejection_cooldown([market])
            assert kept == [market]
            assert benched == 0
        finally:
            await db.close()

    asyncio.run(run())


def test_approval_clears_the_bench():
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            engine = await _engine(db)
            market = _market("m1", yes_price=0.20)
            await engine._record_rejection_state(market, approved=False, reason="r")
            await engine._record_rejection_state(market, approved=True, reason="")

            kept, benched = await engine._apply_rejection_cooldown([market])
            assert kept == [market]
            assert benched == 0
        finally:
            await db.close()

    asyncio.run(run())


def test_expired_rejection_no_longer_benches():
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            engine = await _engine(db)
            market = _market("m1", yes_price=0.20)
            await engine._record_rejection_state(market, approved=False, reason="r")
            # Age the rejection past the cooldown window
            await db.execute(
                "UPDATE signal_rejections SET rejected_at = datetime('now', '-241 minutes') WHERE market_id = 'm1'"
            )
            await db.commit()

            kept, benched = await engine._apply_rejection_cooldown([market])
            assert kept == [market]
            assert benched == 0
        finally:
            await db.close()

    asyncio.run(run())


def test_repeat_rejections_bump_streak_and_refresh_clock():
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            engine = await _engine(db)
            market = _market("m1", yes_price=0.20)
            await engine._record_rejection_state(market, approved=False, reason="first")
            await engine._record_rejection_state(market, approved=False, reason="second")

            row = await db.fetchone("SELECT streak, reason FROM signal_rejections WHERE market_id = 'm1'")
            assert row["streak"] == 2
            assert row["reason"] == "second"
        finally:
            await db.close()

    asyncio.run(run())


def test_unbenched_markets_pass_through():
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            engine = await _engine(db)
            await engine._record_rejection_state(_market("m1"), approved=False, reason="r")
            benched_market = _market("m1")
            clean_market = _market("m2")

            kept, benched = await engine._apply_rejection_cooldown([benched_market, clean_market])
            assert kept == [clean_market]
            assert benched == 1
        finally:
            await db.close()

    asyncio.run(run())


def test_cooldown_disabled_is_a_noop():
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            engine = await _engine(db, cooldown_min=0)
            market = _market("m1")
            await engine._record_rejection_state(market, approved=False, reason="r")

            kept, benched = await engine._apply_rejection_cooldown([market])
            assert kept == [market]
            assert benched == 0
        finally:
            await db.close()

    asyncio.run(run())
