"""Tests for the Kalshi settlements sweep (venue feed → ledger).

Regression context (2026-06-12): Kalshi realized P&L was never recorded —
the resolution tracker keyed off ``market.status``, a field the Market
model didn't have, and the syncer dropped settled positions from portfolio
before booking. The venue's settlements feed is the authoritative source.

Payload semantics (verified against live API responses, 2026-07-19): the
record does NOT carry which side we held — ``yes_count_fp``/``no_count_fp``
both mirror the pair count, legacy ``revenue`` is always 0, and ``value`` is
the YES settlement price in cents. Payout must come from our own cost_basis
(token + size) or, absent that, from the venue's per-side costs when exactly
one side was ever bought. The fetch goes through the raw-JSON SDK variant:
the typed ``Settlement`` model predates the current contract and drops every
field the consumer needs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from auramaur.broker.kalshi_settlements import sweep_kalshi_settlements
from auramaur.exchange.models import Market
from auramaur.strategy.resolution_tracker import ResolutionTracker

SETTLED_AT = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
SETTLED_ISO = SETTLED_AT.isoformat()


def _settlement(ticker="KXTEST-1", result="yes", count="10.00",
                yes_cost="0", no_cost="0", fee="0",
                settled_time="2026-06-01T12:00:00Z"):
    """A raw-JSON settlement shaped like the live API (2026-07)."""
    return {
        "ticker": ticker, "event_ticker": ticker.rsplit("-", 1)[0],
        "market_result": result,
        "yes_count_fp": count, "no_count_fp": count,  # mirrored pair count
        "revenue": 0, "value": 100 if result == "yes" else 0,  # dead fields
        "yes_total_cost_dollars": yes_cost, "no_total_cost_dollars": no_cost,
        "fee_cost": fee, "settled_time": settled_time,
    }


class _RawApi:
    def get_settlements_without_preload_content(self, **kw):  # signature only
        raise AssertionError("must be invoked through _call_raw")


class _Client:
    """Mock KalshiClient exposing _portfolio_api + _call_raw like the real one."""

    def __init__(self, settlements, cursor=None):
        self._portfolio_api = _RawApi()
        self._pages = [{"settlements": settlements, "cursor": cursor}]

    async def _call_raw(self, fn, **kwargs):
        page = self._pages[0] if self._pages else {"settlements": [], "cursor": None}
        return json.dumps(page).encode()


def _db(*, ledger_has=(), cost_row=None):
    db = AsyncMock()

    async def _fetchone(sql, params=None):
        if "pnl_ledger" in sql:
            return {"1": 1} if params[0] in ledger_has else None
        if "cost_basis" in sql:
            return cost_row
        return None

    db.fetchone = AsyncMock(side_effect=_fetchone)
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_win_books_payout_minus_cost_from_own_records(monkeypatch):
    """Held side + size come from cost_basis; payout = size × $1 on a win;
    ledger row lands and cost_basis/portfolio close out."""
    recorded = []

    async def _fake_record(db, **kw):
        recorded.append(kw)

    monkeypatch.setattr(
        "auramaur.broker.kalshi_settlements.record_ledger_event", _fake_record)

    db = _db(cost_row={"token": "YES", "size": 10.0, "avg_cost": 0.62})
    out = await sweep_kalshi_settlements(
        db, _Client([_settlement(fee="0.35")]))

    assert len(out) == 1 and out[0]["booked"] is True
    assert out[0]["payout"] == pytest.approx(10.0)
    assert out[0]["pnl"] == pytest.approx(10.0 - 6.2 - 0.35)
    assert out[0]["basis"] == "own_records"
    assert recorded[0]["source_ref"] == f"kalshi-settle:KXTEST-1:{SETTLED_ISO}"
    assert recorded[0]["kind"] == "settlement"
    assert recorded[0]["realized_at"] == SETTLED_ISO
    sqls = " ".join(str(c.args[0]) for c in db.execute.call_args_list)
    assert "UPDATE cost_basis" in sqls and "DELETE FROM portfolio" in sqls


@pytest.mark.asyncio
async def test_loss_pays_zero_not_venue_revenue():
    """Held NO, market resolved YES → payout 0, pnl = −cost − fee. The dead
    venue ``revenue`` field must play no part."""
    db = _db(cost_row={"token": "NO", "size": 10.0, "avg_cost": 0.40})
    out = await sweep_kalshi_settlements(
        db, _Client([_settlement(result="yes", fee="0.10")]), dry_run=True)
    assert out[0]["payout"] == 0.0
    assert out[0]["pnl"] == pytest.approx(0.0 - 4.0 - 0.10)


@pytest.mark.asyncio
async def test_sweep_idempotent_on_source_ref():
    ref = f"kalshi-settle:KXTEST-1:{SETTLED_ISO}"
    db = _db(ledger_has={ref},
             cost_row={"token": "YES", "size": 10.0, "avg_cost": 0.62})
    out = await sweep_kalshi_settlements(db, _Client([_settlement()]))
    assert out == []
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_raw_iso_string_matches_typed_datetime_source_ref():
    """The raw path's Z-suffixed string must normalize to the same source_ref
    the typed-datetime path produced, or idempotency breaks across versions."""
    ref = f"kalshi-settle:KXTEST-1:{SETTLED_ISO}"  # from datetime.isoformat()
    db = _db(ledger_has={ref},
             cost_row={"token": "YES", "size": 10.0, "avg_cost": 0.62})
    out = await sweep_kalshi_settlements(
        db, _Client([_settlement(settled_time="2026-06-01T12:00:00Z")]))
    assert out == []  # dedup matched despite the differing wire format


@pytest.mark.asyncio
async def test_two_sided_history_without_own_records_is_not_guessed():
    """Both venue sides carry cost and we have no record → the held side is
    unknowable; report, never book (the real backfill case, 2026-07-19)."""
    db = _db(cost_row=None)
    out = await sweep_kalshi_settlements(
        db, _Client([_settlement(result="no", count="8.00",
                                 yes_cost="0.240000", no_cost="6.400000")]))
    assert out[0]["booked"] is False and out[0]["pnl"] is None
    assert "held side unknowable" in out[0]["reason"]
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_single_sided_venue_history_books_without_own_records():
    """Only one side was ever bought → the held side is unambiguous, and the
    venue's own cost prices the settlement."""
    db = _db(cost_row=None)
    out = await sweep_kalshi_settlements(
        db, _Client([_settlement(result="no", count="3.00",
                                 no_cost="2.910000", fee="0.03")]),
        dry_run=True)
    assert out[0]["basis"] == "venue_single_sided"
    assert out[0]["payout"] == pytest.approx(3.0)  # held NO, NO won
    assert out[0]["pnl"] == pytest.approx(3.0 - 2.91 - 0.03)


@pytest.mark.asyncio
async def test_schema_mismatch_still_reported():
    db = _db(cost_row=None)
    out = await sweep_kalshi_settlements(
        db, _Client([_settlement(result="", count="0")]))
    assert out[0]["booked"] is False
    assert out[0]["reason"] == "invalid settlement schema"


@pytest.mark.asyncio
async def test_fetch_survives_typed_model_absence():
    """The sweep must not regress to the typed SDK path: a client without
    _call_raw yields no settlements (and a clear log), not an exception."""
    class _Legacy:
        _portfolio_api = _RawApi()
        # no _call_raw

    out = await sweep_kalshi_settlements(_db(), _Legacy())
    assert out == []


# --- Resolution-tracker detection now actually works for Kalshi ---

def test_kalshi_market_carries_status_and_result():
    m = Market(id="KXTEST-1", exchange="kalshi", question="?",
               status="settled", result="no", outcome_yes_price=0.9)
    # Venue result wins over price inference (price says YES, venue says NO).
    assert ResolutionTracker._detect_resolution(m, "kalshi") is False


def test_kalshi_unsettled_market_not_resolved():
    m = Market(id="KXTEST-1", exchange="kalshi", question="?",
               status="open", outcome_yes_price=0.99)
    assert ResolutionTracker._detect_resolution(m, "kalshi") is None
