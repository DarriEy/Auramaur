"""Tests for the resolution tracker — auto-detection of market resolutions."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
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
    end_date=None,
    closed: bool = False,
) -> Market:
    return Market(
        id=market_id,
        exchange=exchange,
        question="Will it rain tomorrow?",
        active=active,
        closed=closed,
        outcome_yes_price=yes_price,
        outcome_no_price=1.0 - yes_price,
        end_date=end_date,
    )


def _stub_transaction(db):
    """Mock Database.transaction(): the real one is an async context manager,
    which AsyncMock's auto-attributes cannot impersonate."""

    @asynccontextmanager
    async def _txn():
        yield db

    db.transaction = _txn
    return db


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
    return _stub_transaction(db)


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

    def test_active_past_end_date_pinned_resolves(self):
        """Polymarket's active flag lags resolution: a market past its end_date
        with a pinned price IS resolved even though active=True. (Paper
        positions settle only through this path — the stale flag stranded them.)"""
        past = datetime.now(timezone.utc) - timedelta(days=2)
        market = _make_market(active=True, yes_price=0.0, end_date=past)
        assert ResolutionTracker._detect_resolution(market, "polymarket") is False
        market_yes = _make_market(active=True, yes_price=1.0, end_date=past)
        assert ResolutionTracker._detect_resolution(market_yes, "polymarket") is True

    def test_active_pre_end_date_pinned_stays_none(self):
        """The premature-settlement guard: a still-trading market (active and
        BEFORE end_date) is never resolved, even at an extreme price."""
        future = datetime.now(timezone.utc) + timedelta(days=10)
        market = _make_market(active=True, yes_price=0.99, end_date=future)
        assert ResolutionTracker._detect_resolution(market, "polymarket") is None

    def test_active_past_end_date_ambiguous_stays_none(self):
        """Past end_date but price mid (closed awaiting oracle) — still wait."""
        past = datetime.now(timezone.utc) - timedelta(days=1)
        market = _make_market(active=True, yes_price=0.55, end_date=past)
        assert ResolutionTracker._detect_resolution(market, "polymarket") is None

    def test_closed_active_future_end_date_pinned_resolves(self):
        """A resolved losing leg: venue says closed=True but the lagging active
        flag is still True with a FUTURE end_date and price pinned to 0/1.
        Before the fix this returned None and the position lingered at $0."""
        future = datetime.now(timezone.utc) + timedelta(days=160)
        won_yes = _make_market(active=True, closed=True, yes_price=1.0, end_date=future)
        assert ResolutionTracker._detect_resolution(won_yes, "polymarket") is True
        won_no = _make_market(active=True, closed=True, yes_price=0.0, end_date=future)
        assert ResolutionTracker._detect_resolution(won_no, "polymarket") is False

    def test_closed_but_ambiguous_price_stays_none(self):
        """closed=True but price mid (awaiting oracle) — don't guess a winner."""
        future = datetime.now(timezone.utc) + timedelta(days=10)
        market = _make_market(active=True, closed=True, yes_price=0.6, end_date=future)
        assert ResolutionTracker._detect_resolution(market, "polymarket") is None

    def test_active_uma_dispute_blocks_settlement_even_if_pinned(self):
        """A market mid-UMA-dispute is price-pinned to the PROPOSED outcome and
        can flip — never finalize it, even closed + pinned to 0/1."""
        past = datetime.now(timezone.utc) - timedelta(days=1)
        m = _make_market(active=False, closed=True, yes_price=1.0, end_date=past)
        m.uma_status = "disputed"
        assert m.dispute_risk == "DO_NOT_ACT"
        assert ResolutionTracker._detect_resolution(m, "polymarket") is None

    def test_resolved_after_past_dispute_still_settles(self):
        """uma_status 'resolved' is final even if its history shows disputes —
        the guard only holds back ACTIVE disputes."""
        past = datetime.now(timezone.utc) - timedelta(days=1)
        m = _make_market(active=False, closed=True, yes_price=1.0, end_date=past)
        m.uma_status = "resolved"
        m.uma_statuses = ["proposed", "disputed", "proposed", "disputed", "resolved"]
        assert m.dispute_risk == "READY"
        assert ResolutionTracker._detect_resolution(m, "polymarket") is True

    def test_resolved_yes(self):
        market = _make_market(active=False, yes_price=0.99)
        assert ResolutionTracker._detect_resolution(market, "polymarket") is True

    def test_resolved_no(self):
        market = _make_market(active=False, yes_price=0.01)
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

    def test_boundary_yes_099(self):
        market = _make_market(active=False, yes_price=0.99)
        assert ResolutionTracker._detect_resolution(market, "polymarket") is True

    def test_boundary_no_001(self):
        market = _make_market(active=False, yes_price=0.01)
        assert ResolutionTracker._detect_resolution(market, "polymarket") is False

    def test_just_below_threshold_returns_none(self):
        """0.95 used to trigger resolution but is now too loose — must be inactive
        AND tightly converged (>=0.99 / <=0.01) before we declare a winner."""
        market = _make_market(active=False, yes_price=0.95)
        assert ResolutionTracker._detect_resolution(market, "polymarket") is None

    def test_active_high_price_returns_none(self):
        """Even at 95%/5%, an active market is NOT resolved — the prior
        ordering let high-confidence still-trading markets be settled
        prematurely and have their portfolio rows deleted."""
        yes_market = _make_market(active=True, yes_price=0.97)
        no_market = _make_market(active=True, yes_price=0.03)
        assert ResolutionTracker._detect_resolution(yes_market, "polymarket") is None
        assert ResolutionTracker._detect_resolution(no_market, "polymarket") is None


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
        count = asyncio.run(tracker.check_resolutions())

        assert count == 1
        calibration.record_resolution.assert_awaited_once_with("mkt-1", True)

    def test_resolves_no_market(self, resolved_no_market):
        rows = [{"market_id": "mkt-2", "exchange": "polymarket"}]
        db = _make_db(rows=rows)
        calibration = AsyncMock()
        discoveries = {"polymarket": _make_discovery(resolved_no_market)}

        tracker = ResolutionTracker(db=db, calibration=calibration, discoveries=discoveries)
        count = asyncio.run(tracker.check_resolutions())

        assert count == 1
        calibration.record_resolution.assert_awaited_once_with("mkt-2", False)

    def test_skips_active_market(self, active_market):
        rows = [{"market_id": "mkt-3", "exchange": "polymarket"}]
        db = _make_db(rows=rows)
        calibration = AsyncMock()
        discoveries = {"polymarket": _make_discovery(active_market)}

        tracker = ResolutionTracker(db=db, calibration=calibration, discoveries=discoveries)
        count = asyncio.run(tracker.check_resolutions())

        assert count == 0
        calibration.record_resolution.assert_not_awaited()

    def test_handles_missing_discovery(self):
        rows = [{"market_id": "mkt-1", "exchange": "unknown_exchange"}]
        db = _make_db(rows=rows)
        calibration = AsyncMock()
        discoveries = {}  # No discoveries at all

        tracker = ResolutionTracker(db=db, calibration=calibration, discoveries=discoveries)
        count = asyncio.run(tracker.check_resolutions())

        assert count == 0

    def test_no_pending_predictions(self):
        db = _make_db(rows=[])
        calibration = AsyncMock()
        discoveries = {"polymarket": _make_discovery(None)}

        tracker = ResolutionTracker(db=db, calibration=calibration, discoveries=discoveries)
        count = asyncio.run(tracker.check_resolutions())

        assert count == 0

    def test_market_not_found_skipped(self):
        rows = [{"market_id": "mkt-gone", "exchange": "polymarket"}]
        db = _make_db(rows=rows)
        calibration = AsyncMock()
        discoveries = {"polymarket": _make_discovery(None)}  # get_market returns None

        tracker = ResolutionTracker(db=db, calibration=calibration, discoveries=discoveries)
        count = asyncio.run(tracker.check_resolutions())

        assert count == 0

    def test_multi_exchange_resolution(self, resolved_yes_market, resolved_no_market):
        """Markets from different exchanges are resolved correctly."""
        resolved_no_market_kalshi = _make_market(
            market_id="kalshi-mkt", active=False, yes_price=0.01, exchange="kalshi",
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
        count = asyncio.run(tracker.check_resolutions())

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

        asyncio.run(
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
        asyncio.run(
            tracker._settle_position("mkt-1", outcome=True)
        )

        # Should not try to delete anything
        delete_calls = [
            c for c in db.execute.await_args_list
            if "DELETE FROM portfolio" in str(c)
        ]
        assert len(delete_calls) == 0

    def test_settle_zeroes_cost_basis_despite_token_case(self):
        """cost_basis stores the venue's mixed-case label ("No") while the
        portfolio/ledger use the upper-cased TokenType ("NO"). Settlement must
        zero cost_basis.size and accrue realized_pnl regardless of case — a
        case-sensitive match left the row un-zeroed, resurrecting the position
        and never booking realized P&L into cost_basis."""
        from auramaur.db.database import Database

        async def run():
            db = Database(":memory:")
            await db.connect()
            try:
                # Held NO, mixed case in cost_basis; market resolved YES => loss.
                await db.execute(
                    """INSERT INTO cost_basis (market_id, token, token_id, size,
                                               avg_cost, total_cost, realized_pnl, is_paper)
                       VALUES ('mkt-1', 'No', 'tok', 20.0, 0.50, 10.0, 0, 0)"""
                )
                await db.execute(
                    """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
                                              current_price, token, is_paper, updated_at)
                       VALUES ('mkt-1', 'polymarket', 'BUY', 20.0, 0.50, 0, 'NO', 0, datetime('now'))"""
                )
                await db.commit()

                tracker = ResolutionTracker(db=db, calibration=AsyncMock(), discoveries={})
                await tracker._settle_position("mkt-1", outcome=True)  # YES wins, NO held => -10

                cb = await db.fetchone(
                    "SELECT size, realized_pnl FROM cost_basis WHERE market_id='mkt-1' AND is_paper=0")
                assert cb["size"] == 0, "cost_basis.size must be zeroed on settlement"
                assert abs(cb["realized_pnl"] - (-10.0)) < 1e-9, cb["realized_pnl"]

                pf = await db.fetchone(
                    "SELECT COUNT(*) AS n FROM portfolio WHERE market_id='mkt-1' AND is_paper=0")
                assert pf["n"] == 0, "portfolio row must be removed"
            finally:
                await db.close()

        asyncio.run(run())

    def test_settle_from_cost_basis_when_portfolio_row_absent(self):
        """A held leg with no portfolio row (raced/dropped mid-sync) still
        books off cost_basis and is removed — closing the gap where such a leg
        never settled and resurrected at $0."""
        from auramaur.db.database import Database

        async def run():
            db = Database(":memory:")
            await db.connect()
            try:
                await db.execute(
                    """INSERT INTO cost_basis (market_id, token, token_id, size,
                                               avg_cost, total_cost, realized_pnl, is_paper)
                       VALUES ('mkt-x', 'Yes', 'tok', 10.0, 0.1682, 1.682, 0, 0)"""
                )
                await db.commit()  # NOTE: no portfolio row at all
                tracker = ResolutionTracker(db=db, calibration=AsyncMock(), discoveries={})
                await tracker._settle_position("mkt-x", outcome=False)  # NO wins, YES held => loss

                cb = await db.fetchone(
                    "SELECT size, realized_pnl FROM cost_basis WHERE market_id='mkt-x' AND is_paper=0")
                assert cb["size"] == 0
                assert abs(cb["realized_pnl"] - (-1.682)) < 1e-9, cb["realized_pnl"]
                led = await db.fetchone(
                    "SELECT pnl FROM pnl_ledger WHERE source_ref='settle:mkt-x:YES:0'")
                assert led is not None and abs(led["pnl"] - (-1.682)) < 1e-9
            finally:
                await db.close()

        asyncio.run(run())

    def test_resettle_is_idempotent_no_double_count(self):
        """Settling an already-booked leg a second time must not double-count
        daily_stats / cost_basis.realized_pnl — it only cleans up residual size."""
        from auramaur.db.database import Database

        async def run():
            db = Database(":memory:")
            await db.connect()
            try:
                await db.execute(
                    """INSERT INTO cost_basis (market_id, token, token_id, size,
                                               avg_cost, total_cost, realized_pnl, is_paper)
                       VALUES ('mkt-y', 'No', 'tok', 20.0, 0.50, 10.0, 0, 0)"""
                )
                await db.commit()
                tracker = ResolutionTracker(db=db, calibration=AsyncMock(), discoveries={})
                await tracker._settle_position("mkt-y", outcome=True)  # YES wins, NO held => -10
                # simulate the row resurrecting (cost_basis size set back > 0)
                await db.execute("UPDATE cost_basis SET size = 20.0 WHERE market_id='mkt-y'")
                await db.commit()
                await tracker._settle_position("mkt-y", outcome=True)  # second pass

                cb = await db.fetchone(
                    "SELECT size, realized_pnl FROM cost_basis WHERE market_id='mkt-y' AND is_paper=0")
                assert cb["size"] == 0
                assert abs(cb["realized_pnl"] - (-10.0)) < 1e-9, "must not double-count"
                n = await db.fetchone("SELECT COUNT(*) AS n FROM pnl_ledger WHERE market_id='mkt-y'")
                assert n["n"] == 1, "ledger must hold exactly one settlement row"
                ds = await db.fetchone("SELECT total_pnl, trades_count FROM daily_stats")
                assert abs(ds["total_pnl"] - (-10.0)) < 1e-9 and ds["trades_count"] == 1
            finally:
                await db.close()

        asyncio.run(run())

    def test_settle_same_side_under_different_labels_books_once(self):
        """A non-binary market's held side carries a varying label across cycles
        — "YES" one pass, the literal outcome ("ARKANSAS RAZORBACKS") the next.
        Both settle the SAME economic side, so the source_ref must key on the
        side (canonical YES/NO), not the label, or the settlement double-books.
        """
        from auramaur.db.database import Database

        async def run():
            db = Database(":memory:")
            await db.connect()
            try:
                tracker = ResolutionTracker(db=db, calibration=AsyncMock(), discoveries={})

                # Pass 1: row labelled "YES" (the YES side), market resolves NO => loss.
                await db.execute(
                    """INSERT INTO cost_basis (market_id, token, token_id, size,
                                               avg_cost, total_cost, realized_pnl, is_paper)
                       VALUES ('game-1', 'YES', 'tok', 24.0, 0.23, 5.52, 0, 0)""")
                await db.commit()
                await tracker._settle_position("game-1", outcome=False, is_paper_scope=0)

                # Pass 2: SAME side resurfaces relabelled with the literal outcome.
                await db.execute(
                    """INSERT INTO cost_basis (market_id, token, token_id, size,
                                               avg_cost, total_cost, realized_pnl, is_paper)
                       VALUES ('game-1', 'ARKANSAS RAZORBACKS', 'tok', 24.0, 0.23, 5.52, 0, 0)""")
                await db.commit()
                await tracker._settle_position(
                    "game-1", outcome=False, token_scope="ARKANSAS RAZORBACKS",
                    is_paper_scope=0)

                # Exactly one settlement booked, under the canonical side ref.
                n = await db.fetchone(
                    "SELECT COUNT(*) AS n FROM pnl_ledger WHERE market_id='game-1'")
                assert n["n"] == 1, "label variant must not create a second ledger row"
                row = await db.fetchone(
                    "SELECT source_ref, pnl FROM pnl_ledger WHERE market_id='game-1'")
                assert row["source_ref"] == "settle:game-1:YES:0"
                assert abs(row["pnl"] - (0.0 - 0.23) * 24.0) < 1e-9
                ds = await db.fetchone("SELECT total_pnl, trades_count FROM daily_stats")
                assert ds["trades_count"] == 1, "daily_stats must not double-count"
            finally:
                await db.close()

        asyncio.run(run())

    def test_venue_override_corrects_stale_prior_settlement(self):
        """settle_via_venue is authoritative (data-api is_winner). When a side
        already settled at a STALE value (e.g. the winning side booked at $0),
        the venue sweep corrects the booked pnl in place and adjusts daily_stats
        — without creating a second row."""
        from unittest.mock import patch
        from types import SimpleNamespace
        from auramaur.db.database import Database

        async def run():
            db = Database(":memory:")
            await db.connect()
            try:
                # Stale prior settlement: the YES side booked as a LOSS at $0.
                await db.execute(
                    """INSERT INTO pnl_ledger (market_id, venue, category,
                         strategy_source, kind, token, qty, pnl, fees, is_paper, source_ref)
                       VALUES ('gm', 'polymarket', 'sports', 'llm', 'settlement',
                               'YES', 10.0, -5.0, 0.0, 0, 'settle:gm:YES:0')""")
                # The wallet still holds the tokens, so the syncer resurrected the
                # row — relabelled with the literal outcome.
                await db.execute(
                    """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
                         current_price, token, token_id, is_paper, updated_at)
                       VALUES ('gm', 'polymarket', 'BUY', 10.0, 0.50, 1.0,
                               'ARKANSAS RAZORBACKS', 'tok', 0, datetime('now'))""")
                await db.execute(
                    """INSERT INTO cost_basis (market_id, token, token_id, size,
                         avg_cost, total_cost, realized_pnl, is_paper)
                       VALUES ('gm', 'ARKANSAS RAZORBACKS', 'tok', 10.0, 0.50, 5.0, 0, 0)""")
                await db.commit()

                vp = SimpleNamespace(asset_id="tok", is_winner=True,
                                     status="redeemable", title="Arkansas vs Arizona")
                tracker = ResolutionTracker(db=db, calibration=AsyncMock(), discoveries={})
                with patch("auramaur.broker.redeemer.fetch_redeemable_positions",
                           AsyncMock(return_value=[vp])):
                    await tracker.settle_via_venue("0xproxy")

                # Exactly one settlement row for the side, corrected to the win.
                rows = await db.fetchall(
                    "SELECT pnl FROM pnl_ledger WHERE market_id='gm' AND kind='settlement'")
                assert len(rows) == 1
                assert abs(rows[0]["pnl"] - (1.0 - 0.50) * 10.0) < 1e-9  # +5.0 win
                ds = await db.fetchone("SELECT total_pnl FROM daily_stats")
                assert abs(ds["total_pnl"] - 10.0) < 1e-9   # delta +5 - (-5)
                pf = await db.fetchone("SELECT COUNT(*) AS n FROM portfolio WHERE market_id='gm'")
                assert pf["n"] == 0   # resurrected row cleaned up
            finally:
                await db.close()

        asyncio.run(run())

    def test_venue_consistent_prior_is_noop(self):
        """A prior settlement already matching the venue value is left untouched
        (no correction, no daily_stats churn)."""
        from unittest.mock import patch
        from types import SimpleNamespace
        from auramaur.db.database import Database

        async def run():
            db = Database(":memory:")
            await db.connect()
            try:
                await db.execute(
                    """INSERT INTO pnl_ledger (market_id, venue, category,
                         strategy_source, kind, token, qty, pnl, fees, is_paper, source_ref)
                       VALUES ('gm', 'polymarket', 'sports', 'llm', 'settlement',
                               'YES', 10.0, 5.0, 0.0, 0, 'settle:gm:YES:0')""")
                await db.execute(
                    """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
                         current_price, token, token_id, is_paper, updated_at)
                       VALUES ('gm', 'polymarket', 'BUY', 10.0, 0.50, 1.0,
                               'YES', 'tok', 0, datetime('now'))""")
                await db.commit()
                vp = SimpleNamespace(asset_id="tok", is_winner=True,
                                     status="redeemable", title="t")
                tracker = ResolutionTracker(db=db, calibration=AsyncMock(), discoveries={})
                with patch("auramaur.broker.redeemer.fetch_redeemable_positions",
                           AsyncMock(return_value=[vp])):
                    await tracker.settle_via_venue("0xproxy")
                row = await db.fetchone("SELECT pnl FROM pnl_ledger WHERE source_ref='settle:gm:YES:0'")
                assert abs(row["pnl"] - 5.0) < 1e-9      # unchanged
                ds = await db.fetchone("SELECT COUNT(*) AS n FROM daily_stats")
                assert ds["n"] == 0                      # no daily_stats churn
            finally:
                await db.close()

        asyncio.run(run())

    def test_venue_fresh_settles_teamname_label_as_held_payout(self):
        """A fresh venue settlement of a non-binary (team-name) held token uses
        the data-api payout directly — a winning team-name token books a WIN,
        not a loss from the old YES/NO round-trip that mis-mapped it to NO."""
        from unittest.mock import patch
        from types import SimpleNamespace
        from auramaur.db.database import Database

        async def run():
            db = Database(":memory:")
            await db.connect()
            try:
                await db.execute(
                    """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
                         current_price, token, token_id, is_paper, updated_at)
                       VALUES ('gm', 'polymarket', 'BUY', 10.0, 0.40, 1.0,
                               'BLAZERS', 'tok', 0, datetime('now'))""")
                await db.execute(
                    """INSERT INTO cost_basis (market_id, token, token_id, size,
                         avg_cost, total_cost, realized_pnl, is_paper)
                       VALUES ('gm', 'BLAZERS', 'tok', 10.0, 0.40, 4.0, 0, 0)""")
                await db.commit()
                vp = SimpleNamespace(asset_id="tok", is_winner=True,
                                     status="redeemable", title="t")
                tracker = ResolutionTracker(db=db, calibration=AsyncMock(), discoveries={})
                with patch("auramaur.broker.redeemer.fetch_redeemable_positions",
                           AsyncMock(return_value=[vp])):
                    await tracker.settle_via_venue("0xproxy")
                row = await db.fetchone("SELECT pnl FROM pnl_ledger WHERE source_ref='settle:gm:YES:0'")
                assert row is not None
                assert abs(row["pnl"] - (1.0 - 0.40) * 10.0) < 1e-9   # +6.0 win, not a loss
            finally:
                await db.close()

        asyncio.run(run())

    def test_detect_void(self):
        """closed + uma resolved + ~0.5 price = VOID (refund); pinned or
        still-open or non-poly is not a void."""

        def mk(**kw):
            d = dict(active=True, closed=True, yes_price=0.5)
            d.update(kw)
            m = _make_market(active=d["active"], yes_price=d["yes_price"], closed=d["closed"])
            m.uma_status = d.get("uma_status", "resolved")
            return m
        assert ResolutionTracker._detect_void(mk(), "polymarket") is True
        assert ResolutionTracker._detect_void(mk(yes_price=0.98), "polymarket") is False   # pinned winner
        assert ResolutionTracker._detect_void(mk(uma_status="proposed"), "polymarket") is False
        assert ResolutionTracker._detect_void(mk(closed=False), "polymarket") is False
        assert ResolutionTracker._detect_void(mk(), "kalshi") is False

    def test_void_settles_held_token_at_refund_price(self):
        """A void settles the held token at ~0.5 (refund), not 0/1 — booking the
        real pnl vs entry and freeing the position."""
        from auramaur.db.database import Database

        async def run():
            db = Database(":memory:")
            await db.connect()
            try:
                # held NO @ 0.51, refunded at 0.5 -> tiny loss
                await db.execute(
                    """INSERT INTO cost_basis (market_id, token, token_id, size,
                                               avg_cost, total_cost, realized_pnl, is_paper)
                       VALUES ('void-1', 'No', 'tok', 13.0, 0.51, 6.63, 0, 0)""")
                await db.execute(
                    """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
                                              current_price, token, is_paper, updated_at)
                       VALUES ('void-1', 'polymarket', 'BUY', 13.0, 0.51, 0.5, 'NO', 0, datetime('now'))""")
                await db.commit()
                tracker = ResolutionTracker(db=db, calibration=AsyncMock(), discoveries={})
                await tracker._settle_position("void-1", False, is_paper_scope=0,
                                               override_exit_price=0.5)
                led = await db.fetchone(
                    "SELECT pnl FROM pnl_ledger WHERE source_ref='settle:void-1:NO:0'")
                assert led is not None
                assert abs(led["pnl"] - (0.5 - 0.51) * 13.0) < 1e-9   # refund, not -100%
                pf = await db.fetchone("SELECT COUNT(*) AS n FROM portfolio WHERE market_id='void-1'")
                assert pf["n"] == 0
            finally:
                await db.close()
        asyncio.run(run())

    def test_cost_basis_fallback_respects_is_paper_scope(self):
        """When a market is held in BOTH modes and the portfolio row is absent,
        a mode-scoped settle must book only the targeted mode off cost_basis —
        not pick the wrong leg (the bug that booked a live loss as paper)."""
        from auramaur.db.database import Database

        async def run():
            db = Database(":memory:")
            await db.connect()
            try:
                for paper in (0, 1):
                    await db.execute(
                        """INSERT INTO cost_basis (market_id, token, token_id, size,
                                                   avg_cost, total_cost, realized_pnl, is_paper)
                           VALUES ('mkt-z', 'Yes', 'tok', 10.0, 0.20, 2.0, 0, ?)""",
                        (paper,),
                    )
                await db.commit()
                tracker = ResolutionTracker(db=db, calibration=AsyncMock(), discoveries={})
                await tracker._settle_position("mkt-z", outcome=False, is_paper_scope=0)  # live only

                live = await db.fetchone("SELECT size FROM cost_basis WHERE market_id='mkt-z' AND is_paper=0")
                paper = await db.fetchone("SELECT size FROM cost_basis WHERE market_id='mkt-z' AND is_paper=1")
                assert live["size"] == 0, "live leg must settle"
                assert paper["size"] == 10.0, "paper leg must be untouched"
                refs = [r["source_ref"] for r in await db.fetchall(
                    "SELECT source_ref FROM pnl_ledger WHERE market_id='mkt-z'")]
                assert refs == ["settle:mkt-z:YES:0"], refs
            finally:
                await db.close()

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Venue-truth sweep (Polymarket data-api) — settles positions the Gamma loop
# can't see (archived markets / stale active flags / phantom $0 marks)
# ---------------------------------------------------------------------------

def _make_venue_position(asset_id: str, *, is_winner: bool, title="Old match",
                         outcome="Yes", status="redeemable"):
    vp = MagicMock()
    vp.asset_id = asset_id
    vp.is_winner = is_winner
    vp.title = title
    vp.outcome = outcome
    vp.status = status
    return vp


def _make_sweep_db(portfolio_rows: list[dict]):
    db = AsyncMock()

    async def _fetchall(sql, params=None):
        if "FROM portfolio" in sql and "token_id" in sql:
            return portfolio_rows
        return []

    async def _fetchone(sql, params=None):
        # _settle_position re-fetches the scoped row
        if "FROM portfolio" in sql and portfolio_rows:
            tok = params[1] if params and len(params) > 1 else None
            for r in portfolio_rows:
                if tok is None or r.get("token") == tok:
                    return r
        return None

    db.fetchall = AsyncMock(side_effect=_fetchall)
    db.fetchone = AsyncMock(side_effect=_fetchone)
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    return _stub_transaction(db)


@pytest.mark.asyncio
async def test_venue_sweep_settles_matched_token(monkeypatch):
    """A live NO position whose token the venue reports as the winner
    settles as outcome NO (outcome_yes=False) with payout $1/share."""
    row = {"market_id": "m1", "token": "NO", "token_id": "tok-no-1",
           "size": 20.0, "avg_price": 0.62, "side": "BUY", "is_paper": 0}
    db = _make_sweep_db([row])
    tracker = ResolutionTracker(db=db, calibration=None, discoveries={},
                                proxy_address="0xproxy")
    vps = [_make_venue_position("tok-no-1", is_winner=True, outcome="No")]

    async def _fake_fetch(proxy, **kw):
        return vps
    monkeypatch.setattr("auramaur.broker.redeemer.fetch_redeemable_positions",
                        _fake_fetch)

    settlements = await tracker.settle_via_venue("0xproxy")
    assert len(settlements) == 1
    s = settlements[0]
    assert s["outcome_yes"] is False  # we hold NO and NO won
    assert s["pnl"] == pytest.approx((1.0 - 0.62) * 20.0)
    # The portfolio row was deleted token-scoped and the market deactivated.
    executed_sql = " ".join(str(c.args[0]) for c in db.execute.call_args_list)
    assert "DELETE FROM portfolio" in executed_sql
    assert "AND UPPER(token) = UPPER(?)" in executed_sql
    assert "UPDATE markets SET active = 0" in executed_sql


@pytest.mark.asyncio
async def test_venue_sweep_losing_yes_leg(monkeypatch):
    """A YES leg the venue says lost settles at $0 — realized loss equals
    cost, replacing the phantom -100% unrealized mark."""
    row = {"market_id": "m2", "token": "YES", "token_id": "tok-yes-2",
           "size": 10.0, "avg_price": 0.40, "side": "BUY", "is_paper": 0}
    db = _make_sweep_db([row])
    tracker = ResolutionTracker(db=db, calibration=None, discoveries={},
                                proxy_address="0xproxy")

    async def _fake_fetch(proxy, **kw):
        return [_make_venue_position("tok-yes-2", is_winner=False,
                                     status="pending_oracle")]
    monkeypatch.setattr("auramaur.broker.redeemer.fetch_redeemable_positions",
                        _fake_fetch)

    settlements = await tracker.settle_via_venue("0xproxy")
    assert len(settlements) == 1
    assert settlements[0]["outcome_yes"] is False
    assert settlements[0]["pnl"] == pytest.approx(-4.0)


@pytest.mark.asyncio
async def test_venue_sweep_dry_run_writes_nothing(monkeypatch):
    row = {"market_id": "m1", "token": "NO", "token_id": "tok-no-1",
           "size": 20.0, "avg_price": 0.62, "side": "BUY", "is_paper": 0}
    db = _make_sweep_db([row])
    tracker = ResolutionTracker(db=db, calibration=None, discoveries={},
                                proxy_address="0xproxy")

    async def _fake_fetch(proxy, **kw):
        return [_make_venue_position("tok-no-1", is_winner=True)]
    monkeypatch.setattr("auramaur.broker.redeemer.fetch_redeemable_positions",
                        _fake_fetch)

    settlements = await tracker.settle_via_venue("0xproxy", dry_run=True)
    assert len(settlements) == 1
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_venue_sweep_ignores_unmatched_and_no_proxy(monkeypatch):
    db = _make_sweep_db([{"market_id": "m1", "token": "YES",
                          "token_id": "tok-other", "size": 5.0,
                          "avg_price": 0.5, "side": "BUY", "is_paper": 0}])
    tracker = ResolutionTracker(db=db, calibration=None, discoveries={})

    async def _fake_fetch(proxy, **kw):
        return [_make_venue_position("tok-unknown", is_winner=True)]
    monkeypatch.setattr("auramaur.broker.redeemer.fetch_redeemable_positions",
                        _fake_fetch)

    assert await tracker.settle_via_venue("0xproxy") == []
    assert await tracker.settle_via_venue("") == []


@pytest.mark.asyncio
async def test_venue_sweep_drains_already_settled_token_without_double_book(monkeypatch):
    """A settled-but-unredeemed leg whose cost_basis size never zeroed must be
    DRAINED (size→0, market inactive) so it stops resurrecting every cycle — but
    WITHOUT double-booking: the ledger insert and daily_stats stay untouched when
    a matching prior already exists. (Previously the sweep skipped it entirely,
    which left the cost_basis leg lingering forever — the resurrection bug.)"""
    row = {"market_id": "m1", "token": "NO", "token_id": "tok-no-1",
           "size": 20.0, "avg_price": 0.62, "side": "BUY", "is_paper": 0}
    db = _make_sweep_db([row])

    async def _fetchone(sql, params=None):
        if "pnl_ledger" in sql:
            # NO @ 0.62 won (is_winner) => (1.0-0.62)*20 = 7.6; the prior matches,
            # so no correction and no re-book — only the drain proceeds.
            return {"pnl": (1.0 - 0.62) * 20.0}
        return row
    db.fetchone = AsyncMock(side_effect=_fetchone)

    tracker = ResolutionTracker(db=db, calibration=None, discoveries={},
                                proxy_address="0xproxy")

    async def _fake_fetch(proxy, **kw):
        return [_make_venue_position("tok-no-1", is_winner=True)]
    monkeypatch.setattr("auramaur.broker.redeemer.fetch_redeemable_positions",
                        _fake_fetch)

    settled = await tracker.settle_via_venue("0xproxy")

    # The leg is now processed (drained), not skipped — and not a correction.
    assert len(settled) == 1
    assert settled[0]["correction"] is False
    executed_sql = " ".join(str(c.args[0]) for c in db.execute.call_args_list)
    # Drain happened: cost_basis zeroed + the resurrected portfolio row removed.
    assert "UPDATE cost_basis" in executed_sql and "size = 0" in executed_sql
    assert "DELETE FROM portfolio" in executed_sql
    # No double-book: the matching prior means no re-insert and no daily_stats.
    assert "INSERT OR IGNORE INTO pnl_ledger" not in executed_sql
    assert "daily_stats" not in executed_sql
