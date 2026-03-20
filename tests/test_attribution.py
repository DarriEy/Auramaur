"""Tests for performance attribution."""

from __future__ import annotations

import asyncio

import pytest

from auramaur.db.database import Database
from auramaur.monitoring.attribution import PerformanceAttributor


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
def attributor(db):
    return PerformanceAttributor(db=db)


def run(coro, loop):
    return loop.run_until_complete(coro)


class TestPerformanceAttributor:
    def test_record_first_trade(self, attributor, event_loop):
        run(attributor.record_trade_result("politics", 5.0, 0.08), event_loop)
        stats = run(attributor.get_category_stats(), event_loop)
        assert len(stats) == 1
        assert stats[0]["category"] == "politics"
        assert stats[0]["total_pnl"] == 5.0
        assert stats[0]["trade_count"] == 1
        assert stats[0]["win_count"] == 1

    def test_record_multiple_trades(self, attributor, event_loop):
        run(attributor.record_trade_result("crypto", 10.0, 0.1), event_loop)
        run(attributor.record_trade_result("crypto", -3.0, 0.06), event_loop)
        run(attributor.record_trade_result("crypto", 5.0, 0.08), event_loop)
        stats = run(attributor.get_category_stats(), event_loop)
        assert stats[0]["trade_count"] == 3
        assert stats[0]["win_count"] == 2
        assert abs(stats[0]["total_pnl"] - 12.0) < 0.01

    def test_kelly_multiplier_positive_ev(self, attributor, event_loop):
        """Winning category should get multiplier > 1.0."""
        for i in range(6):
            pnl = 5.0 if i < 4 else -2.0  # 4 wins, 2 losses
            run(attributor.record_trade_result("politics", pnl, 0.07), event_loop)

        mults = run(attributor.compute_kelly_multipliers(), event_loop)
        assert "politics" in mults
        assert mults["politics"] > 1.0

    def test_kelly_multiplier_negative_ev(self, attributor, event_loop):
        """Losing category should get multiplier < 1.0."""
        for i in range(6):
            pnl = -5.0 if i < 4 else 2.0  # 4 losses, 2 wins
            run(attributor.record_trade_result("sports", pnl, 0.05), event_loop)

        mults = run(attributor.compute_kelly_multipliers(), event_loop)
        assert "sports" in mults
        assert mults["sports"] < 1.0

    def test_get_kelly_multiplier_unknown(self, attributor, event_loop):
        """Unknown category should return 1.0."""
        mult = run(attributor.get_kelly_multiplier("unknown"), event_loop)
        assert mult == 1.0

    def test_insufficient_trades_excluded(self, attributor, event_loop):
        """Categories with < 5 trades shouldn't get multipliers."""
        for i in range(3):
            run(attributor.record_trade_result("tiny", 1.0, 0.05), event_loop)

        mults = run(attributor.compute_kelly_multipliers(), event_loop)
        assert "tiny" not in mults
