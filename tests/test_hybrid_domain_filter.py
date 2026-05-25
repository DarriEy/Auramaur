"""Tests for hybrid mode domain specialization filter."""

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
    return PerformanceFeedback(db=db)


def run(coro, loop):
    return loop.run_until_complete(coro)


async def _seed_calibration(db: Database, category: str, correct: int, incorrect: int):
    """Insert resolved calibration rows for a category."""
    for _ in range(correct):
        await db.execute(
            """INSERT INTO calibration (market_id, predicted_prob, actual_outcome, category)
               VALUES (?, ?, ?, ?)""",
            (f"m_{category}_{correct}_{_}", 0.7, 1, category),
        )
    for _ in range(incorrect):
        await db.execute(
            """INSERT INTO calibration (market_id, predicted_prob, actual_outcome, category)
               VALUES (?, ?, ?, ?)""",
            (f"m_{category}_wrong_{_}", 0.7, 0, category),
        )
    await db.commit()


class TestGetWhitelistCategories:
    def test_cold_start_returns_empty(self, feedback, event_loop):
        """With no calibration data, whitelist and probationary are both empty."""
        wl, prob = run(feedback.get_whitelist_categories(), event_loop)
        assert wl == set()
        assert prob == set()

    def test_category_whitelisted_when_accurate(self, feedback, db, event_loop):
        """Category with >=50% accuracy and >=10 trades is whitelisted."""
        run(_seed_calibration(db, "politics", correct=8, incorrect=4), event_loop)
        wl, prob = run(feedback.get_whitelist_categories(min_accuracy=0.50, min_trades=10), event_loop)
        # 12 trades, 8/12 = 66.7% accuracy -> whitelisted
        assert "politics" in wl
        assert "politics" not in prob

    def test_category_probationary_when_insufficient_trades(self, feedback, db, event_loop):
        """Category with good accuracy but <10 trades is probationary."""
        run(_seed_calibration(db, "sports", correct=4, incorrect=1), event_loop)
        wl, prob = run(feedback.get_whitelist_categories(min_accuracy=0.50, min_trades=10), event_loop)
        assert "sports" not in wl
        assert "sports" in prob

    def test_category_probationary_when_low_accuracy(self, feedback, db, event_loop):
        """Category with enough trades but <50% accuracy is probationary."""
        run(_seed_calibration(db, "crypto", correct=3, incorrect=9), event_loop)
        wl, prob = run(feedback.get_whitelist_categories(min_accuracy=0.50, min_trades=10), event_loop)
        assert "crypto" not in wl
        assert "crypto" in prob

    def test_multiple_categories(self, feedback, db, event_loop):
        """Mixed categories are correctly split between whitelisted and probationary."""
        run(_seed_calibration(db, "politics", correct=8, incorrect=4), event_loop)
        run(_seed_calibration(db, "crypto", correct=3, incorrect=9), event_loop)
        run(_seed_calibration(db, "tech", correct=2, incorrect=1), event_loop)
        wl, prob = run(feedback.get_whitelist_categories(min_accuracy=0.50, min_trades=10), event_loop)
        assert wl == {"politics"}
        assert prob == {"crypto", "tech"}

    def test_custom_thresholds(self, feedback, db, event_loop):
        """Custom min_accuracy and min_trades are respected."""
        run(_seed_calibration(db, "science", correct=4, incorrect=1), event_loop)
        wl, prob = run(feedback.get_whitelist_categories(min_accuracy=0.50, min_trades=5), event_loop)
        assert "science" in wl
