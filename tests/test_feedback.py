"""Tests for the per-category performance feedback loop.

Focus: the calibration table holds one row per analysis *snapshot*, so a single
market re-analyzed across many cycles appears many times. Accuracy, Brier, the
avoid-list, and the hybrid whitelist must all measure per distinct *market* (the
resolution event), not per snapshot — otherwise one heavily-re-analyzed market
dominates a whole category.
"""

from __future__ import annotations

import asyncio

import pytest

from auramaur.broker.feedback import PerformanceFeedback
from auramaur.db.database import Database


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
def feedback(db):
    return PerformanceFeedback(db)


def run(coro, loop):
    return loop.run_until_complete(coro)


async def _add(db, market_id, pred, outcome, category):
    await db.execute(
        "INSERT INTO calibration (market_id, predicted_prob, actual_outcome, category) "
        "VALUES (?, ?, ?, ?)",
        (market_id, pred, outcome, category),
    )
    await db.commit()


class TestSnapshotDedup:
    def test_snapshots_of_one_market_count_once(self, db, feedback, event_loop):
        """One market snapshotted 12 times is a single resolution event."""
        for _ in range(12):
            run(_add(db, "m1", 0.2, 1, "sports"), event_loop)  # predicted NO, resolved YES

        stats = run(feedback.get_category_accuracy(), event_loop)
        assert stats["sports"]["trade_count"] == 1

        # One distinct market is below _AVOID_MIN_TRADES (10), so a single bad
        # market must NOT drag the whole category onto the avoid list.
        avoid = run(feedback.get_avoid_categories(), event_loop)
        assert "sports" not in avoid

    def test_avoid_fires_on_distinct_bad_markets(self, db, feedback, event_loop):
        """12 distinct markets all called wrong → category is avoided."""
        for i in range(12):
            run(_add(db, f"bad{i}", 0.2, 1, "badcat"), event_loop)

        stats = run(feedback.get_category_accuracy(), event_loop)
        assert stats["badcat"]["trade_count"] == 12
        assert stats["badcat"]["accuracy"] == 0.0

        avoid = run(feedback.get_avoid_categories(), event_loop)
        assert "badcat" in avoid

    def test_dedup_keeps_latest_prediction(self, db, feedback, event_loop):
        """When a market is re-forecast, the most recent snapshot wins."""
        run(_add(db, "m1", 0.1, 1, "tech"), event_loop)  # early: wrong
        run(_add(db, "m1", 0.9, 1, "tech"), event_loop)  # latest: right

        stats = run(feedback.get_category_accuracy(), event_loop)
        assert stats["tech"]["trade_count"] == 1
        assert stats["tech"]["accuracy"] == 1.0  # latest snapshot is correct

    def test_whitelist_counts_distinct_markets(self, db, feedback, event_loop):
        """A single good market snapshotted many times must not satisfy the
        whitelist's min-trades gate (which means min distinct markets)."""
        for _ in range(12):
            run(_add(db, "g1", 0.9, 1, "crypto"), event_loop)  # one market, called right

        whitelisted, probationary = run(
            feedback.get_whitelist_categories(min_accuracy=0.5, min_trades=10),
            event_loop,
        )
        assert "crypto" not in whitelisted  # only 1 distinct market
        assert "crypto" in probationary
