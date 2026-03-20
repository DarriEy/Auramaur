"""Tests for correlation detection."""

from __future__ import annotations

import asyncio

import pytest

from auramaur.db.database import Database
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
def detector(db):
    return CorrelationDetector(db=db)


def run(coro, loop):
    return loop.run_until_complete(coro)


class TestCorrelationDetector:
    def test_get_related_empty(self, detector, event_loop):
        """No relationships stored should return empty list."""
        result = run(detector.get_related_markets("mkt_1"), event_loop)
        assert result == []

    def test_store_and_retrieve_relationship(self, detector, event_loop):
        """Manually stored relationships should be retrievable."""
        run(
            detector._db.execute(
                """INSERT INTO market_relationships
                   (market_id_a, market_id_b, relationship_type, strength, description, detected_at)
                   VALUES (?, ?, ?, ?, ?, datetime('now'))""",
                ("mkt_1", "mkt_2", "correlated", 0.8, "Same event"),
            ),
            event_loop,
        )
        run(detector._db.commit(), event_loop)

        result = run(detector.get_related_markets("mkt_1"), event_loop)
        assert len(result) == 1
        assert result[0]["market_id_b"] == "mkt_2"
        assert result[0]["strength"] == 0.8

    def test_detect_arbitrage_conditional_violation(self, detector, event_loop):
        """Should detect conditional probability violations."""
        # Set up markets
        run(
            detector._db.execute(
                """INSERT INTO markets (id, condition_id, question, active, outcome_yes_price, last_updated)
                   VALUES ('primary', 'c1', 'Win primary?', 1, 0.70, datetime('now'))""",
            ),
            event_loop,
        )
        run(
            detector._db.execute(
                """INSERT INTO markets (id, condition_id, question, active, outcome_yes_price, last_updated)
                   VALUES ('general', 'c2', 'Win general?', 1, 0.60, datetime('now'))""",
            ),
            event_loop,
        )
        # Conditional relationship: winning primary implies general
        run(
            detector._db.execute(
                """INSERT INTO market_relationships
                   (market_id_a, market_id_b, relationship_type, strength, description, detected_at)
                   VALUES ('primary', 'general', 'conditional', 0.9, 'Primary implies general', datetime('now'))""",
            ),
            event_loop,
        )
        run(detector._db.commit(), event_loop)

        opps = run(detector.detect_arbitrage(), event_loop)
        assert len(opps) == 1
        assert opps[0]["type"] == "conditional_violation"

    def test_no_arbitrage_when_consistent(self, detector, event_loop):
        """No arbitrage when prices are consistent."""
        run(
            detector._db.execute(
                """INSERT INTO markets (id, condition_id, question, active, outcome_yes_price, last_updated)
                   VALUES ('a', 'c1', 'Q1', 1, 0.50, datetime('now'))""",
            ),
            event_loop,
        )
        run(
            detector._db.execute(
                """INSERT INTO markets (id, condition_id, question, active, outcome_yes_price, last_updated)
                   VALUES ('b', 'c2', 'Q2', 1, 0.55, datetime('now'))""",
            ),
            event_loop,
        )
        run(
            detector._db.execute(
                """INSERT INTO market_relationships
                   (market_id_a, market_id_b, relationship_type, strength, description, detected_at)
                   VALUES ('a', 'b', 'conditional', 0.7, 'Related', datetime('now'))""",
            ),
            event_loop,
        )
        run(detector._db.commit(), event_loop)

        opps = run(detector.detect_arbitrage(), event_loop)
        assert len(opps) == 0
