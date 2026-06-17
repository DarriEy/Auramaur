"""Tests for the resolution tracker — auto-detection of market resolutions."""

from __future__ import annotations

import asyncio
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
    return db


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
async def test_venue_sweep_skips_already_settled_token(monkeypatch):
    """A settled-but-unredeemed leg resurrected by the syncer must NOT
    settle twice — only the ledger insert is idempotent; daily_stats and
    cost_basis would double-count."""
    row = {"market_id": "m1", "token": "NO", "token_id": "tok-no-1",
           "size": 20.0, "avg_price": 0.62, "side": "BUY", "is_paper": 0}
    db = _make_sweep_db([row])

    async def _fetchone(sql, params=None):
        if "pnl_ledger" in sql:
            assert params == ("settle:m1:NO:0",)
            return {"1": 1}  # settlement already recorded
        return row
    db.fetchone = AsyncMock(side_effect=_fetchone)

    tracker = ResolutionTracker(db=db, calibration=None, discoveries={},
                                proxy_address="0xproxy")

    async def _fake_fetch(proxy, **kw):
        return [_make_venue_position("tok-no-1", is_winner=True)]
    monkeypatch.setattr("auramaur.broker.redeemer.fetch_redeemable_positions",
                        _fake_fetch)

    assert await tracker.settle_via_venue("0xproxy") == []
    executed_sql = " ".join(str(c.args[0]) for c in db.execute.call_args_list)
    assert "DELETE FROM portfolio" not in executed_sql
