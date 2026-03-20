"""Tests for arbitrage execution."""

from __future__ import annotations

import asyncio

import pytest

from auramaur.db.database import Database
from auramaur.exchange.models import OrderSide
from auramaur.strategy.arbitrage import ArbitrageExecutor
from auramaur.strategy.correlation import CorrelationDetector


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def db(event_loop):
    async def _setup():
        database = Database(db_path=":memory:")
        await database.connect()
        return database
    database = event_loop.run_until_complete(_setup())
    yield database
    event_loop.run_until_complete(database.close())


@pytest.fixture
def executor(db):
    correlator = CorrelationDetector(db=db)
    return ArbitrageExecutor(db=db, correlator=correlator)


def run(coro, loop):
    return loop.run_until_complete(coro)


class TestArbitrageExecutor:
    def test_no_opportunities_returns_empty(self, executor, event_loop):
        result = run(executor.generate_arb_signals(), event_loop)
        assert result == []

    def test_conditional_violation_generates_pair(self, executor, event_loop):
        """Conditional violation should produce buy-B sell-A signals."""
        db = executor._db
        # Market A (primary): 70% — too high if it implies B
        run(db.execute(
            """INSERT INTO markets (id, condition_id, question, active, outcome_yes_price, outcome_no_price, last_updated)
               VALUES ('primary', 'c1', 'Win primary?', 1, 0.70, 0.30, datetime('now'))"""
        ), event_loop)
        # Market B (general): 60% — if A implies B, P(A) should be <= P(B)
        run(db.execute(
            """INSERT INTO markets (id, condition_id, question, active, outcome_yes_price, outcome_no_price, last_updated)
               VALUES ('general', 'c2', 'Win general?', 1, 0.60, 0.40, datetime('now'))"""
        ), event_loop)
        # Conditional relationship
        run(db.execute(
            """INSERT INTO market_relationships
               (market_id_a, market_id_b, relationship_type, strength, description, detected_at)
               VALUES ('primary', 'general', 'conditional', 0.9, 'Primary implies general', datetime('now'))"""
        ), event_loop)
        run(db.commit(), event_loop)

        pairs = run(executor.generate_arb_signals(), event_loop)
        assert len(pairs) == 1
        buy_sig, sell_sig, opp = pairs[0]
        # Should buy B (general, underpriced) and sell A (primary, overpriced)
        assert buy_sig.market_id == "general"
        assert buy_sig.recommended_side == OrderSide.BUY
        assert sell_sig.market_id == "primary"
        assert sell_sig.recommended_side == OrderSide.SELL

    def test_price_divergence_generates_pair(self, executor, event_loop):
        """Same-event divergence should buy cheap and sell expensive."""
        db = executor._db
        run(db.execute(
            """INSERT INTO markets (id, condition_id, question, active, outcome_yes_price, outcome_no_price, last_updated)
               VALUES ('cheap', 'c1', 'Event?', 1, 0.45, 0.55, datetime('now'))"""
        ), event_loop)
        run(db.execute(
            """INSERT INTO markets (id, condition_id, question, active, outcome_yes_price, outcome_no_price, last_updated)
               VALUES ('expensive', 'c2', 'Same event?', 1, 0.55, 0.45, datetime('now'))"""
        ), event_loop)
        run(db.execute(
            """INSERT INTO market_relationships
               (market_id_a, market_id_b, relationship_type, strength, description, detected_at)
               VALUES ('cheap', 'expensive', 'same_event', 0.95, 'Same event', datetime('now'))"""
        ), event_loop)
        run(db.commit(), event_loop)

        pairs = run(executor.generate_arb_signals(), event_loop)
        assert len(pairs) == 1
        buy_sig, sell_sig, _ = pairs[0]
        assert buy_sig.market_id == "cheap"
        assert sell_sig.market_id == "expensive"

    def test_missing_market_skipped(self, executor, event_loop):
        """Opportunities with missing markets should be skipped."""
        db = executor._db
        # Only create one market
        run(db.execute(
            """INSERT INTO markets (id, condition_id, question, active, outcome_yes_price, outcome_no_price, last_updated)
               VALUES ('exists', 'c1', 'Q?', 1, 0.50, 0.50, datetime('now'))"""
        ), event_loop)
        run(db.execute(
            """INSERT INTO market_relationships
               (market_id_a, market_id_b, relationship_type, strength, description, detected_at)
               VALUES ('exists', 'missing', 'same_event', 0.9, 'Related', datetime('now'))"""
        ), event_loop)
        run(db.commit(), event_loop)

        pairs = run(executor.generate_arb_signals(), event_loop)
        assert len(pairs) == 0
