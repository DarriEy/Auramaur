"""Phase 5 of docs/plans/db-contention-plan.md: record_fill is atomic and
retry-safe.

The fill INSERT + cost_basis upsert (+ daily_stats on sells) now run inside
one Database.transaction(). An SQLITE_BUSY mid-way can no longer leave a fill
row without its cost basis (or vice versa), and because the order_id dedupe
check runs inside the same transaction, a retry after a busy failure is
idempotent — the property that finally makes retry-on-busy safe.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from auramaur.broker.pnl import PnLTracker
from auramaur.db.database import Database
from auramaur.exchange.models import Fill, OrderSide, TokenType


def _paper_settings():
    from unittest.mock import MagicMock

    s = MagicMock()
    s.is_live = False
    return s


def _fill(order_id="ord-1", side=OrderSide.BUY, size=10.0, price=0.40):
    return Fill(
        order_id=order_id, market_id="mkt-1", token_id="tok-1",
        side=side, token=TokenType.YES, size=size, price=price,
        fee=0.0, is_paper=True,
        timestamp=datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc),
    )


async def _counts(db: Database) -> tuple[int, int]:
    fills = await db.fetchone("SELECT COUNT(*) AS n FROM fills")
    basis = await db.fetchone("SELECT COUNT(*) AS n FROM cost_basis")
    return int(fills["n"]), int(basis["n"])


@pytest.mark.asyncio
async def test_busy_failure_leaves_zero_partial_rows_and_retry_succeeds(tmp_path):
    db_path = str(tmp_path / "t.db")
    db = Database(db_path)
    await db.connect()
    # The production busy_timeout is 30s; shrink it so the blocked attempt
    # fails in milliseconds instead of stalling the test.
    await db.execute("PRAGMA busy_timeout=200")
    tracker = PnLTracker(db, _paper_settings())
    try:
        blocker = sqlite3.connect(db_path)
        blocker.execute("PRAGMA busy_timeout=100")
        blocker.execute("BEGIN IMMEDIATE")  # hold the write lock
        try:
            with pytest.raises(sqlite3.OperationalError):
                await tracker.record_fill(_fill())
        finally:
            blocker.rollback()
            blocker.close()

        # The failed attempt must be invisible: no fill, no cost basis.
        assert await _counts(db) == (0, 0)

        # Retry after the lock clears: lands exactly once.
        await tracker.record_fill(_fill())
        assert await _counts(db) == (1, 1)

        # And a replay of the same fill dedupes (idempotent retries).
        await tracker.record_fill(_fill())
        assert await _counts(db) == (1, 1)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_sell_realizes_atomically_with_ledger_after_commit(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    tracker = PnLTracker(db, _paper_settings())
    try:
        await tracker.record_fill(_fill(order_id="buy-1"))
        await tracker.record_fill(
            _fill(order_id="sell-1", side=OrderSide.SELL, size=10.0, price=0.55))

        basis = await db.fetchone(
            "SELECT size, realized_pnl FROM cost_basis WHERE market_id = 'mkt-1'")
        assert basis["size"] == pytest.approx(0.0)
        assert basis["realized_pnl"] == pytest.approx((0.55 - 0.40) * 10.0)

        # Ledger event landed AFTER the authoritative commit, keyed to the fill.
        ledger = await db.fetchone(
            "SELECT kind, pnl, source_ref FROM pnl_ledger WHERE market_id = 'mkt-1'")
        assert ledger is not None
        assert ledger["kind"] == "sell"
        assert ledger["pnl"] == pytest.approx(1.5)
        assert ledger["source_ref"].startswith("fill:")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_conflicting_replay_rejected_without_side_effects(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    tracker = PnLTracker(db, _paper_settings())
    try:
        await tracker.record_fill(_fill())
        with pytest.raises(ValueError, match="conflicting fill replay"):
            await tracker.record_fill(_fill(price=0.99))  # same order, new price
        assert await _counts(db) == (1, 1)
    finally:
        await db.close()
