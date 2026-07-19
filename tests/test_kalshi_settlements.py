"""Tests for the Kalshi settlements sweep (venue feed → ledger).

Regression context (2026-06-12): Kalshi realized P&L was never recorded —
the resolution tracker keyed off ``market.status``, a field the Market
model didn't have, and the syncer dropped settled positions from portfolio
before booking. The venue's settlements feed is the authoritative source.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from auramaur.broker.kalshi_settlements import sweep_kalshi_settlements
from auramaur.exchange.models import Market
from auramaur.strategy.resolution_tracker import ResolutionTracker

SETTLED_AT = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def _settlement(ticker="KXTEST-1", result="yes", yes_count=10, no_count=0,
                revenue=1000, settled_time=SETTLED_AT):
    return SimpleNamespace(ticker=ticker, result=result, yes_count=yes_count,
                           no_count=no_count, revenue=revenue,
                           settled_time=settled_time)


def _client(settlements, cursor=None):
    """Mock KalshiClient exposing _portfolio_api + _call like the real one."""
    api = SimpleNamespace(get_settlements=lambda **kw: None)

    async def _call(fn, **kwargs):
        return SimpleNamespace(settlements=settlements, cursor=cursor)

    return SimpleNamespace(_portfolio_api=api, _call=_call)


def _db(*, ledger_has=(), cost_row=None, trades_net=None):
    db = AsyncMock()

    async def _fetchone(sql, params=None):
        if "pnl_ledger" in sql:
            return {"1": 1} if params[0] in ledger_has else None
        if "cost_basis" in sql:
            return cost_row
        if "FROM trades" in sql:
            return {"net": trades_net}
        return None

    db.fetchone = AsyncMock(side_effect=_fetchone)
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_sweep_books_settlement_with_cost_basis(monkeypatch):
    """A winning settlement books pnl = revenue − cost into the ledger and
    closes out cost_basis/portfolio (daily_stats untouched by design)."""
    recorded = []

    async def _fake_record(db, **kw):
        recorded.append(kw)

    monkeypatch.setattr(
        "auramaur.broker.kalshi_settlements.record_ledger_event", _fake_record)

    db = _db(cost_row={"size": 10.0, "avg_cost": 0.62})
    out = await sweep_kalshi_settlements(db, _client([_settlement()]))

    assert len(out) == 1 and out[0]["booked"] is True
    assert out[0]["pnl"] == pytest.approx(10.0 - 6.2)  # $10 revenue − $6.20 cost
    assert recorded[0]["source_ref"] == f"kalshi-settle:KXTEST-1:{SETTLED_AT.isoformat()}"
    assert recorded[0]["kind"] == "settlement"
    assert recorded[0]["realized_at"] == SETTLED_AT.isoformat()
    sqls = " ".join(str(c.args[0]) for c in db.execute.call_args_list)
    assert "UPDATE cost_basis" in sqls and "DELETE FROM portfolio" in sqls


@pytest.mark.asyncio
async def test_sweep_idempotent_on_source_ref():
    ref = f"kalshi-settle:KXTEST-1:{SETTLED_AT.isoformat()}"
    db = _db(ledger_has={ref}, cost_row={"size": 10.0, "avg_cost": 0.62})
    out = await sweep_kalshi_settlements(db, _client([_settlement()]))
    assert out == []
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_sweep_skips_unknown_cost_instead_of_guessing():
    """No cost basis on record → reported but NOT booked. A fabricated cost
    would poison the venue scorecard this sweep exists to build."""
    db = _db(cost_row=None, trades_net=None)
    out = await sweep_kalshi_settlements(db, _client([_settlement()]))
    assert len(out) == 1
    assert out[0]["booked"] is False and out[0]["pnl"] is None
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_sweep_falls_back_to_filled_trades_for_cost():
    db = _db(cost_row=None, trades_net=6.2)
    out = await sweep_kalshi_settlements(
        db, _client([_settlement(result="no", yes_count=0, no_count=10,
                                 revenue=0)]),
        dry_run=True)
    assert out[0]["pnl"] == pytest.approx(0.0 - 6.2)  # lost: $0 revenue
    assert out[0]["booked"] is False  # dry run


@pytest.mark.asyncio
async def test_sweep_parses_current_fixed_point_schema_and_fees():
    """Current Kalshi fields must not silently become zero-sized, fee-free rows."""
    settlement = SimpleNamespace(
        ticker="KXFP-1", market_result="yes", yes_count_fp="10.25",
        no_count_fp="0.00", yes_total_cost_dollars="6.1500",
        revenue=1025, fee_cost="0.35", settled_time=SETTLED_AT,
    )
    db = _db(cost_row=None, trades_net=None)
    out = await sweep_kalshi_settlements(db, _client([settlement]), dry_run=True)
    assert out[0]["qty"] == pytest.approx(10.25)
    assert out[0]["fees"] == pytest.approx(0.35)
    assert out[0]["pnl"] == pytest.approx(10.25 - 6.15 - 0.35)


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
