"""A closed exchange must not demote a proven instrument.

This is the failure that stopped a live-execution clock. `eligible_keys`
filters on status='eligible'; demote an instrument and the book's universe
empties, `run_once` returns before `_record_daily_mark`, and the 180-day clock
for `IBKRMultiAssetExecution.graduated()` silently stops accumulating.
"""

import pytest

from auramaur.db.database import Database
from auramaur.exchange.ibkr_instruments import BY_KEY
from auramaur.exchange.ibkr_registry import eligible_keys, record_validation


class _Contract:
    def __init__(self, con_id=42):
        self.conId = con_id
        self.exchange = "SMART"
        self.currency = "USD"
        self.multiplier = 1.0


async def _prove_eligible(db, spec):
    return await record_validation(db, spec, _Contract(), quote_source="ibkr_live",
                                   has_history=True)


@pytest.mark.asyncio
async def test_closed_venue_does_not_demote_a_proven_instrument():
    db = Database(":memory:")
    await db.connect()
    spec = BY_KEY["SPY"]

    assert await _prove_eligible(db, spec) == "eligible"
    assert "SPY" in await eligible_keys(db)

    # After hours: no live quote, but contract and history are fine.
    status = await record_validation(db, spec, _Contract(), quote_source="none",
                                     has_history=True, venue_closed=True)
    assert status == "eligible", "an after-hours probe demoted a proven instrument"
    assert "SPY" in await eligible_keys(db), "book universe would have emptied"
    await db.close()


@pytest.mark.asyncio
async def test_closed_venue_does_not_promote_an_unproven_instrument():
    """The guard preserves; it must never manufacture eligibility."""
    db = Database(":memory:")
    await db.connect()
    spec = BY_KEY["QQQ"]

    status = await record_validation(db, spec, _Contract(), quote_source="none",
                                     has_history=True, venue_closed=True)
    assert status == "qualified_no_live_data"
    assert "QQQ" not in await eligible_keys(db)
    await db.close()


@pytest.mark.asyncio
async def test_a_real_failure_still_demotes_even_when_closed():
    """Only the absence of a live quote is excused. A genuine error is not."""
    db = Database(":memory:")
    await db.connect()
    spec = BY_KEY["IWM"]
    assert await _prove_eligible(db, spec) == "eligible"

    status = await record_validation(db, spec, _Contract(), quote_source="none",
                                     has_history=True, error="no executable BBO",
                                     venue_closed=True)
    assert status == "quarantined"
    assert "IWM" not in await eligible_keys(db)
    await db.close()


@pytest.mark.asyncio
async def test_missing_history_still_demotes_when_closed():
    db = Database(":memory:")
    await db.connect()
    spec = BY_KEY["DIA"]
    assert await _prove_eligible(db, spec) == "eligible"

    status = await record_validation(db, spec, _Contract(), quote_source="none",
                                     has_history=False, venue_closed=True)
    assert status == "qualified_no_live_data"
    await db.close()


@pytest.mark.asyncio
async def test_open_venue_without_a_live_quote_still_demotes():
    """Venue open + no executable quote IS evidence, and must demote."""
    db = Database(":memory:")
    await db.connect()
    spec = BY_KEY["VTI"]
    assert await _prove_eligible(db, spec) == "eligible"

    status = await record_validation(db, spec, _Contract(), quote_source="ibkr_delayed",
                                     has_history=True)
    assert status == "qualified_no_live_data"
    await db.close()
