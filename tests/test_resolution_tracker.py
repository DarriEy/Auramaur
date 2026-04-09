"""Tests for the resolution tracker — auto-detection of market resolutions."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from auramaur.exchange.models import Market
from auramaur.strategy.resolution_tracker import ResolutionTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_market(
    market_id: str = "test-market",
    active: bool = False,
    yes_price: float = 1.0,
    exchange: str = "polymarket",
) -> Market:
    return Market(
        id=market_id,
        exchange=exchange,
        question="Will it rain tomorrow?",
        active=active,
        outcome_yes_price=yes_price,
        outcome_no_price=1.0 - yes_price,
    )


def _make_db(rows: list[dict] | None = None, pos_row: dict | None = None):
    """Build a mock Database."""
    db = AsyncMock()

    async def _fetchall(sql, params=None):
        if "calibration" in sql and "actual_outcome IS NULL" in sql:
            return rows or []
        return []

    async def _fetchone(sql, params=None):
        if "portfolio" in sql:
            return pos_row
        return None

    db.fetchall = AsyncMock(side_effect=_fetchall)
    db.fetchone = AsyncMock(side_effect=_fetchone)
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    return db


def _make_discovery(market: Market | None):
    disc = AsyncMock()
    disc.get_market = AsyncMock(return_value=market)
    return disc


# ---------------------------------------------------------------------------
# Tests — _detect_resolution
# ---------------------------------------------------------------------------

class TestDetectResolution:
    def test_active_market_returns_none(self):
        market = _make_market(active=True, yes_price=0.65)
        assert ResolutionTracker._detect_resolution(market, "polymarket") is None

    def test_resolved_yes(self):
        market = _make_market(active=False, yes_price=0.99)
        assert ResolutionTracker._detect_resolution(market, "polymarket") is True

    def test_resolved_no(self):
        market = _make_market(active=False, yes_price=0.02)
        assert ResolutionTracker._detect_resolution(market, "polymarket") is False

    def test_ambiguous_price_returns_none(self):
        """Market closed but price is in the middle — can't determine resolution."""
        market = _make_market(active=False, yes_price=0.55)
        assert ResolutionTracker._detect_resolution(market, "polymarket") is None

    def test_kalshi_settled_uses_price_tiebreak(self):
        """Kalshi settled market with ambiguous price uses >0.5 heuristic."""
        # Use MagicMock to simulate a market with a status attribute
        # (Pydantic Market model doesn't have status, but Kalshi raw data does)
        market = MagicMock()
        market.active = False
        market.outcome_yes_price = 0.70
        market.status = "settled"
        assert ResolutionTracker._detect_resolution(market, "kalshi") is True

    def test_kalshi_settled_no(self):
        market = MagicMock()
        market.active = False
        market.outcome_yes_price = 0.30
        market.status = "finalized"
        assert ResolutionTracker._detect_resolution(market, "kalshi") is False

    def test_boundary_yes_095(self):
        market = _make_market(active=False, yes_price=0.95)
        assert ResolutionTracker._detect_resolution(market, "polymarket") is True

    def test_boundary_no_005(self):
        market = _make_market(active=False, yes_price=0.05)
        assert ResolutionTracker._detect_resolution(market, "polymarket") is False

    def test_just_below_threshold_returns_none(self):
        market = _make_market(active=False, yes_price=0.94)
        assert ResolutionTracker._detect_resolution(market, "polymarket") is None


# ---------------------------------------------------------------------------
# Tests — check_resolutions
# ---------------------------------------------------------------------------

class TestCheckResolutions:
    @pytest.fixture
    def resolved_yes_market(self):
        return _make_market(market_id="mkt-1", active=False, yes_price=0.99)

    @pytest.fixture
    def resolved_no_market(self):
        return _make_market(market_id="mkt-2", active=False, yes_price=0.01)

    @pytest.fixture
    def active_market(self):
        return _make_market(market_id="mkt-3", active=True, yes_price=0.60)

    def test_resolves_yes_market(self, resolved_yes_market):
        rows = [{"market_id": "mkt-1", "exchange": "polymarket"}]
        db = _make_db(rows=rows)
        calibration = AsyncMock()
        discoveries = {"polymarket": _make_discovery(resolved_yes_market)}

        tracker = ResolutionTracker(db=db, calibration=calibration, discoveries=discoveries)
        count = asyncio.get_event_loop().run_until_complete(tracker.check_resolutions())

        assert count == 1
        calibration.record_resolution.assert_awaited_once_with("mkt-1", True)

    def test_resolves_no_market(self, resolved_no_market):
        rows = [{"market_id": "mkt-2", "exchange": "polymarket"}]
        db = _make_db(rows=rows)
        calibration = AsyncMock()
        discoveries = {"polymarket": _make_discovery(resolved_no_market)}

        tracker = ResolutionTracker(db=db, calibration=calibration, discoveries=discoveries)
        count = asyncio.get_event_loop().run_until_complete(tracker.check_resolutions())

        assert count == 1
        calibration.record_resolution.assert_awaited_once_with("mkt-2", False)

    def test_skips_active_market(self, active_market):
        rows = [{"market_id": "mkt-3", "exchange": "polymarket"}]
        db = _make_db(rows=rows)
        calibration = AsyncMock()
        discoveries = {"polymarket": _make_discovery(active_market)}

        tracker = ResolutionTracker(db=db, calibration=calibration, discoveries=discoveries)
        count = asyncio.get_event_loop().run_until_complete(tracker.check_resolutions())

        assert count == 0
        calibration.record_resolution.assert_not_awaited()

    def test_handles_missing_discovery(self):
        rows = [{"market_id": "mkt-1", "exchange": "unknown_exchange"}]
        db = _make_db(rows=rows)
        calibration = AsyncMock()
        discoveries = {}  # No discoveries at all

        tracker = ResolutionTracker(db=db, calibration=calibration, discoveries=discoveries)
        count = asyncio.get_event_loop().run_until_complete(tracker.check_resolutions())

        assert count == 0

    def test_no_pending_predictions(self):
        db = _make_db(rows=[])
        calibration = AsyncMock()
        discoveries = {"polymarket": _make_discovery(None)}

        tracker = ResolutionTracker(db=db, calibration=calibration, discoveries=discoveries)
        count = asyncio.get_event_loop().run_until_complete(tracker.check_resolutions())

        assert count == 0

    def test_market_not_found_skipped(self):
        rows = [{"market_id": "mkt-gone", "exchange": "polymarket"}]
        db = _make_db(rows=rows)
        calibration = AsyncMock()
        discoveries = {"polymarket": _make_discovery(None)}  # get_market returns None

        tracker = ResolutionTracker(db=db, calibration=calibration, discoveries=discoveries)
        count = asyncio.get_event_loop().run_until_complete(tracker.check_resolutions())

        assert count == 0

    def test_multi_exchange_resolution(self, resolved_yes_market, resolved_no_market):
        """Markets from different exchanges are resolved correctly."""
        resolved_no_market_kalshi = _make_market(
            market_id="kalshi-mkt", active=False, yes_price=0.02, exchange="kalshi",
        )
        rows = [
            {"market_id": "mkt-1", "exchange": "polymarket"},
            {"market_id": "kalshi-mkt", "exchange": "kalshi"},
        ]
        db = _make_db(rows=rows)
        calibration = AsyncMock()
        discoveries = {
            "polymarket": _make_discovery(resolved_yes_market),
            "kalshi": _make_discovery(resolved_no_market_kalshi),
        }

        tracker = ResolutionTracker(db=db, calibration=calibration, discoveries=discoveries)
        count = asyncio.get_event_loop().run_until_complete(tracker.check_resolutions())

        assert count == 2
        calls = calibration.record_resolution.await_args_list
        assert ("mkt-1", True) in [(c.args[0], c.args[1]) for c in calls]
        assert ("kalshi-mkt", False) in [(c.args[0], c.args[1]) for c in calls]


# ---------------------------------------------------------------------------
# Tests — _settle_position
# ---------------------------------------------------------------------------

class TestSettlePosition:
    def test_settle_buy_yes_resolved_yes(self):
        """BUY YES position, market resolves YES — should profit."""
        pos_row = {
            "avg_price": 0.60,
            "size": 10.0,
            "side": "BUY",
            "token": "YES",
        }
        db = _make_db(pos_row=pos_row)
        calibration = AsyncMock()
        tracker = ResolutionTracker(db=db, calibration=calibration, discoveries={})

        asyncio.get_event_loop().run_until_complete(
            tracker._settle_position("mkt-1", outcome=True)
        )

        # Should delete from portfolio
        delete_calls = [
            c for c in db.execute.await_args_list
            if "DELETE FROM portfolio" in str(c)
        ]
        assert len(delete_calls) >= 1

    def test_settle_no_position(self):
        """No portfolio entry — should return without error."""
        db = _make_db(pos_row=None)
        calibration = AsyncMock()
        tracker = ResolutionTracker(db=db, calibration=calibration, discoveries={})

        # Should not raise
        asyncio.get_event_loop().run_until_complete(
            tracker._settle_position("mkt-1", outcome=True)
        )

        # Should not try to delete anything
        delete_calls = [
            c for c in db.execute.await_args_list
            if "DELETE FROM portfolio" in str(c)
        ]
        assert len(delete_calls) == 0
