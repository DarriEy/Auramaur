"""Phase 2 of the txn migration (#353): the paper-book _fill batches.

The pattern per adopted writer: crash the SECOND statement of the batch
and assert zero rows from ANY statement (atomicity), assert the distinct
owner label, and prove the connection is usable afterwards (no strand —
the 2026-07-25 wedge class)."""

import sqlite3

import pytest

from auramaur.db.database import Database
from auramaur.exchange.models import OrderSide
from tests.test_ibkr_etf_paper import _pillar
from tests.txn_helpers import failing_on, span_owners, transaction_spy


@pytest.mark.asyncio
async def test_etf_fill_batch_is_atomic_under_mid_batch_crash():
    db = Database(":memory:")
    await db.connect()
    try:
        pillar = await _pillar(db)
        with failing_on(db, "INSERT INTO ibkr_etf_ledger"):
            with pytest.raises(sqlite3.OperationalError, match="injected"):
                await pillar._fill("SPY", OrderSide.BUY, 1, 100.0)
        for table in ("ibkr_etf_fills", "ibkr_etf_positions"):
            row = await db.fetchone(f"SELECT COUNT(*) AS n FROM {table}")
            assert row["n"] == 0, f"partial write survived in {table}"
        # No strand: the connection accepts unrelated writes immediately.
        await db.execute(
            "INSERT OR IGNORE INTO markets (id, question) VALUES ('m','q')")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_etf_fill_carries_its_owner_label():
    db = Database(":memory:")
    await db.connect()
    try:
        pillar = await _pillar(db)
        events: list = []
        with transaction_spy(db, events):
            await pillar._fill("SPY", OrderSide.BUY, 1, 100.0)
        assert "ibkr_etf_paper.fill" in span_owners(events)
        assert events.count(("ibkr_etf_paper.fill", "end")) == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_etf_sell_of_absent_position_writes_nothing():
    """The orphan fix the span buys for free: validation now precedes any
    write, so a bad sell leaves no fill/commission rows behind (the old
    sequence stranded both before raising)."""
    db = Database(":memory:")
    await db.connect()
    try:
        pillar = await _pillar(db)
        with pytest.raises(ValueError, match="absent"):
            await pillar._fill("SPY", OrderSide.SELL, 1, 100.0)
        row = await db.fetchone("SELECT COUNT(*) AS n FROM ibkr_etf_fills")
        assert row["n"] == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_multiasset_fill_batch_is_atomic_under_mid_batch_crash():
    from auramaur.exchange.ibkr_instruments import IBKRBook
    from config.settings import Settings

    from auramaur.strategy.ibkr_multiasset_paper import IBKRMultiAssetPaperBook
    from tests.test_ibkr_multiasset_paper import FakeMarketData

    db = Database(":memory:")
    await db.connect()
    try:
        settings = Settings()
        settings.ibkr.multiasset_paper_enabled = True
        settings.ibkr.multiasset_registry_required = False
        pillar = IBKRMultiAssetPaperBook(
            settings, FakeMarketData(), db, IBKRBook.GLOBAL_ETF)
        pillar.market_open = lambda now=None: True
        with failing_on(db, "INSERT INTO ibkr_paper_ledger"):
            # run_once swallows per-instrument errors by design; atomicity
            # is judged on the tables, not the exception.
            await pillar.run_once()
        for table in ("ibkr_paper_fills", "ibkr_paper_positions"):
            row = await db.fetchone(f"SELECT COUNT(*) AS n FROM {table}")
            assert row["n"] == 0, f"partial write survived in {table}"
        # No strand, and the writer works again once the fault clears.
        events: list = []
        with transaction_spy(db, events):
            await pillar.run_once()
        assert "ibkr_multiasset.fill" in span_owners(events)
        row = await db.fetchone("SELECT COUNT(*) AS n FROM ibkr_paper_fills")
        assert row["n"] >= 1
    finally:
        await db.close()
