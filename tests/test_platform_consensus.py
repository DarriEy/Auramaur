"""Tests for the Platform Consensus follower pillar."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

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
from auramaur.data_sources.base import NewsItem
from auramaur.strategy.platform_consensus import PlatformConsensusPillar
from config.settings import Settings


def _market(mid="M1", question="Will SpaceX launch Starship in July?", yes=0.50, liquidity=5000.0, active=True, days_out=10.0) -> Market:
    return Market(
        id=mid,
        exchange="polymarket",
        question=question,
        category="science",
        active=active,
        outcome_yes_price=yes,
        outcome_no_price=round(1 - yes, 2),
        liquidity=liquidity,
        volume=10000.0,
        end_date=datetime.now(timezone.utc) + timedelta(days=days_out),
        clob_token_yes="ty",
        clob_token_no="tn",
        condition_id="cond-1",
        description="SpaceX Starship orbital flight test",
    )


def _settings(**overrides) -> Settings:
    s = Settings()
    s.platform_consensus.enabled = True
    s.platform_consensus.paper = True
    s.platform_consensus.min_edge = 0.05
    for k, v in overrides.items():
        setattr(s.platform_consensus, k, v)
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
    ex.place_order = AsyncMock(
        side_effect=lambda order: OrderResult(
            order_id="ord-1",
            market_id=order.market_id,
            status="paper" if order.dry_run else "filled",
            filled_size=order.size if filled else 0,
            filled_price=order.price,
            is_paper=order.dry_run,
        )
    )
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


def test_platform_consensus_triggers_buy_yes():
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            # Create schema for signals and portfolio
            await db.execute(
                """CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY,
                    market_id TEXT,
                    claude_prob REAL,
                    claude_confidence TEXT,
                    market_prob REAL,
                    edge REAL,
                    second_opinion_prob REAL,
                    divergence REAL,
                    evidence_summary TEXT,
                    action TEXT,
                    strategy_source TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            await db.execute(
                """CREATE TABLE IF NOT EXISTS portfolio (
                    market_id TEXT PRIMARY KEY,
                    side TEXT,
                    size REAL,
                    avg_price REAL,
                    current_price REAL,
                    category TEXT,
                    token TEXT,
                    token_id TEXT,
                    is_paper INTEGER DEFAULT 1
                )"""
            )
            await db.execute(
                """CREATE TABLE IF NOT EXISTS markets (
                    id TEXT PRIMARY KEY,
                    exchange TEXT,
                    condition_id TEXT,
                    question TEXT,
                    description TEXT,
                    category TEXT,
                    active INTEGER,
                    outcome_yes_price REAL,
                    outcome_no_price REAL,
                    volume REAL,
                    liquidity REAL,
                    last_updated TEXT
                )"""
            )
            await db.commit()

            # Target: Market YES price is 0.50. Consensus forecasts YES is 0.75.
            # Edge is 0.25 > 0.05 min_edge + fee.
            # It should trigger BUY YES.
            market = _market(yes=0.50)
            settings = _settings()
            ex = _exchange()
            rm = _risk()
            disc = MagicMock()
            disc.get_markets = AsyncMock(return_value=[market])

            pillar = PlatformConsensusPillar(
                db=db,
                settings=settings,
                discovery=disc,
                exchange=ex,
                risk_manager=rm,
                pnl_tracker=PnLTracker(db, settings),
                calibration=MagicMock(),
            )

            manifold_news = NewsItem(
                id="man-1",
                source="manifold",
                title="[Manifold: 75%] Will SpaceX launch Starship in July?",
                content="Community forecast is 75%\nUnique bettors: 80\nLiquidity: $5,000",
                url="https://manifold.markets/space",
            )

            with patch.object(pillar._manifold, "fetch", AsyncMock(return_value=[manifold_news])), \
                 patch.object(pillar._metaculus, "fetch", AsyncMock(return_value=[])):
                entered = await pillar.run_once()

                assert entered == 1
                # Check DB signals
                sig_row = await db.fetchone("SELECT * FROM signals WHERE market_id = ?", (market.id,))
                assert sig_row is not None
                assert sig_row["claude_prob"] == 0.75
                assert sig_row["action"] == "BUY"
                assert sig_row["strategy_source"] == "platform_consensus"
        finally:
            await db.close()

    asyncio.run(run())


def test_platform_consensus_triggers_buy_no():
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            # Create tables
            await db.execute(
                """CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY,
                    market_id TEXT,
                    claude_prob REAL,
                    claude_confidence TEXT,
                    market_prob REAL,
                    edge REAL,
                    second_opinion_prob REAL,
                    divergence REAL,
                    evidence_summary TEXT,
                    action TEXT,
                    strategy_source TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            await db.execute(
                """CREATE TABLE IF NOT EXISTS portfolio (
                    market_id TEXT PRIMARY KEY,
                    side TEXT,
                    size REAL,
                    avg_price REAL,
                    current_price REAL,
                    category TEXT,
                    token TEXT,
                    token_id TEXT,
                    is_paper INTEGER DEFAULT 1
                )"""
            )
            await db.execute(
                """CREATE TABLE IF NOT EXISTS markets (
                    id TEXT PRIMARY KEY,
                    exchange TEXT,
                    condition_id TEXT,
                    question TEXT,
                    description TEXT,
                    category TEXT,
                    active INTEGER,
                    outcome_yes_price REAL,
                    outcome_no_price REAL,
                    volume REAL,
                    liquidity REAL,
                    last_updated TEXT
                )"""
            )
            await db.commit()

            # Target: Market YES price is 0.50. Consensus forecasts YES is 0.25 (No is 0.75).
            # Edge is 0.25 > 0.05 min_edge + fee.
            # It should trigger BUY NO (which is recommended_side=OrderSide.SELL)
            market = _market(yes=0.50)
            settings = _settings()
            ex = _exchange()
            rm = _risk()
            disc = MagicMock()
            disc.get_markets = AsyncMock(return_value=[market])

            pillar = PlatformConsensusPillar(
                db=db,
                settings=settings,
                discovery=disc,
                exchange=ex,
                risk_manager=rm,
                pnl_tracker=PnLTracker(db, settings),
                calibration=MagicMock(),
            )

            metaculus_news = NewsItem(
                id="meta-1",
                source="metaculus",
                title="[Metaculus: 25%] Will SpaceX launch Starship in July?",
                content="Community forecast is 25%\nForecasters: 40",
                url="https://metaculus.com/space",
            )

            with patch.object(pillar._manifold, "fetch", AsyncMock(return_value=[])), \
                 patch.object(pillar._metaculus, "fetch", AsyncMock(return_value=[metaculus_news])):
                entered = await pillar.run_once()

                assert entered == 1
                # Check DB signals
                sig_row = await db.fetchone("SELECT * FROM signals WHERE market_id = ?", (market.id,))
                assert sig_row is not None
                assert sig_row["claude_prob"] == 0.25
                assert sig_row["action"] == "SELL"
                assert sig_row["strategy_source"] == "platform_consensus"
        finally:
            await db.close()

    asyncio.run(run())


def test_platform_consensus_insufficient_edge():
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            # Create tables
            await db.execute(
                """CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY,
                    market_id TEXT,
                    claude_prob REAL,
                    claude_confidence TEXT,
                    market_prob REAL,
                    edge REAL,
                    second_opinion_prob REAL,
                    divergence REAL,
                    evidence_summary TEXT,
                    action TEXT,
                    strategy_source TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            await db.execute(
                """CREATE TABLE IF NOT EXISTS portfolio (
                    market_id TEXT PRIMARY KEY,
                    side TEXT,
                    size REAL,
                    avg_price REAL,
                    current_price REAL,
                    category TEXT,
                    token TEXT,
                    token_id TEXT,
                    is_paper INTEGER DEFAULT 1
                )"""
            )
            await db.execute(
                """CREATE TABLE IF NOT EXISTS markets (
                    id TEXT PRIMARY KEY,
                    exchange TEXT,
                    condition_id TEXT,
                    question TEXT,
                    description TEXT,
                    category TEXT,
                    active INTEGER,
                    outcome_yes_price REAL,
                    outcome_no_price REAL,
                    volume REAL,
                    liquidity REAL,
                    last_updated TEXT
                )"""
            )
            await db.commit()

            # Target: Market YES price is 0.50. Consensus is 0.52. Edge is 0.02 < 0.05 min_edge.
            market = _market(yes=0.50)
            settings = _settings()
            ex = _exchange()
            rm = _risk()
            disc = MagicMock()
            disc.get_markets = AsyncMock(return_value=[market])

            pillar = PlatformConsensusPillar(
                db=db,
                settings=settings,
                discovery=disc,
                exchange=ex,
                risk_manager=rm,
                pnl_tracker=PnLTracker(db, settings),
                calibration=MagicMock(),
            )

            manifold_news = NewsItem(
                id="man-1",
                source="manifold",
                title="[Manifold: 52%] Will SpaceX launch Starship in July?",
                content="Community forecast is 52%\nUnique bettors: 80\nLiquidity: $5,000",
                url="https://manifold.markets/space",
            )

            with patch.object(pillar._manifold, "fetch", AsyncMock(return_value=[manifold_news])), \
                 patch.object(pillar._metaculus, "fetch", AsyncMock(return_value=[])):
                entered = await pillar.run_once()

                assert entered == 0
                # DB should have no signal
                sig_row = await db.fetchone("SELECT * FROM signals WHERE market_id = ?", (market.id,))
                assert sig_row is None
        finally:
            await db.close()

    asyncio.run(run())


def test_platform_consensus_quality_thresholds_fail_closed():
    cfg = _settings().platform_consensus
    thin = NewsItem(
        id="thin",
        source="manifold",
        title="[Manifold: 75%] Example",
        content="Unique bettors: 2\nLiquidity: $20",
        url="https://example.test",
    )
    missing = NewsItem(
        id="missing",
        source="metaculus",
        title="[Metaculus: 75%] Example",
        content="Community probability: 75%",
        url="https://example.test",
    )

    assert not PlatformConsensusPillar._quality_ok(thin, "Manifold", cfg)
    assert not PlatformConsensusPillar._quality_ok(missing, "Metaculus", cfg)


def test_zero_entry_cycle_still_logs():
    """A zero-entry cycle must still log cycle_done (the entered>0 guard made
    zero-entry cycles silent — weeks of 'running fine, entering nothing' were
    indistinguishable from a dead task)."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            disc = MagicMock()
            disc.get_markets = AsyncMock(return_value=[])
            pillar = PlatformConsensusPillar(
                db=db, settings=_settings(), discovery=disc,
                exchange=_exchange(), risk_manager=_risk(),
                pnl_tracker=PnLTracker(db, _settings()),
                calibration=MagicMock(),
            )
            with patch("auramaur.strategy.platform_consensus.log") as mock_log:
                entered = await pillar.run_once()
            assert entered == 0
            calls = [c for c in mock_log.info.call_args_list
                     if c.args and c.args[0] == "platform_consensus.cycle_done"]
            assert len(calls) == 1
            assert calls[0].kwargs["entered"] == 0
            assert calls[0].kwargs["scanned"] == 0
        finally:
            await db.close()
    asyncio.run(run())
