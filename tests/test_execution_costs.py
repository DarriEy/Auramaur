"""Measured execution costs: classification, idempotent ingest, aggregation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from auramaur.broker.execution_costs import ingest_fills, measure, venue_class
from auramaur.db.database import Database


def test_venue_class_splits_equities_by_region_only():
    """Region drives cost for equities (min ticket, conversion, stamp duty)
    and for nothing else — FX and options do not split that way."""
    assert venue_class("STK", "USD") == "us_equity"
    assert venue_class("STK", "EUR") == "eu_equity"
    assert venue_class("STK", "GBP") == "eu_equity"
    assert venue_class("STK", "JPY") == "asia_equity"
    assert venue_class("STK", "HKD") == "asia_equity"
    assert venue_class("CASH", "EUR") == "fx"        # not eu_equity
    assert venue_class("OPT", "USD") == "option"
    assert venue_class("FUT", "USD") == "future"
    assert venue_class("", "") == "other"


def _fill(exec_id, *, symbol="SPY", sec_type="STK", currency="USD",
          side="BOT", shares=10.0, price=100.0, commission=1.0,
          order_ref="probe-1"):
    return SimpleNamespace(
        contract=SimpleNamespace(symbol=symbol, secType=sec_type,
                                 currency=currency),
        execution=SimpleNamespace(
            execId=exec_id, acctNumber="U1", exchange="SMART", side=side,
            shares=shares, price=price, orderRef=order_ref),
        commissionReport=(SimpleNamespace(commission=commission, currency="USD")
                          if commission is not None else None),
        time="2026-07-29T16:00:00+00:00",
    )


@pytest.mark.asyncio
async def test_ingest_is_idempotent_on_exec_id(tmp_path):
    """IBKR returns the session's executions on every call, so re-ingesting
    must not double-count the same fill."""
    db = Database(str(tmp_path / "c.db"))
    await db.connect()
    try:
        ib = SimpleNamespace(reqExecutionsAsync=AsyncMock(
            return_value=[_fill("e1"), _fill("e2")]))
        assert await ingest_fills(db, ib, probe_label="p") == 2
        assert await ingest_fills(db, ib, probe_label="p") == 0
        row = await db.fetchone("SELECT COUNT(*) AS n FROM cost_observations")
        assert row["n"] == 2
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_zero_commission_is_stored_as_unknown(tmp_path):
    """The commission report arrives as a separate message; a fill read before
    it lands has none. Zero is a claim, and a wrong one."""
    db = Database(str(tmp_path / "c.db"))
    await db.connect()
    try:
        ib = SimpleNamespace(reqExecutionsAsync=AsyncMock(
            return_value=[_fill("e1", commission=0.0)]))
        await ingest_fills(db, ib)
        row = await db.fetchone(
            "SELECT commission FROM cost_observations WHERE exec_id='e1'")
        assert row["commission"] is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_measure_reports_commission_in_dollars_and_bps(tmp_path):
    """Both, because they answer different questions: dollars exposes the
    fixed minimum that dominates at small size, bps the marginal rate.
    Reporting only bps is how a $1 floor gets mistaken for a 0.1% rate."""
    db = Database(str(tmp_path / "c.db"))
    await db.connect()
    try:
        ib = SimpleNamespace(reqExecutionsAsync=AsyncMock(return_value=[
            # $1 commission on $1,000 notional == 10bps
            _fill("e1", shares=10, price=100.0, commission=1.0),
            # $1 commission on $200 notional == 50bps: same dollars, 5x bps
            _fill("e2", shares=2, price=100.0, commission=1.0),
        ]))
        await ingest_fills(db, ib)
        [us] = await measure(db, venue="us_equity")
        assert us.fills == 2
        assert us.commission_usd == pytest.approx(1.0)
        assert us.commission_bps == pytest.approx(30.0)   # mean of 10 and 50
        assert us.slippage_bps is None                     # no mids supplied
        assert us.mids_available == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_slippage_needs_the_mid_at_submission(tmp_path):
    """The mid cannot be recovered from the fill afterwards, so it is supplied
    at ingest or slippage is simply unknown."""
    db = Database(str(tmp_path / "c.db"))
    await db.connect()
    try:
        ib = SimpleNamespace(reqExecutionsAsync=AsyncMock(return_value=[
            _fill("e1", price=100.10, order_ref="probe-1")]))
        await ingest_fills(db, ib, mids={"probe-1": 100.00})
        [us] = await measure(db, venue="us_equity")
        assert us.mids_available == 1
        assert us.slippage_bps == pytest.approx(10.0)      # 10c on $100
    finally:
        await db.close()
