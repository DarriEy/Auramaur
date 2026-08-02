"""A settlement whose quantity cannot be explained by the mode's own placed
BUYs must not move a named strategy's cell.

The 2026-07-23/24 all-paper window persisted instant-filled MM quote
inventory (no trades/fills ancestry) into ``portfolio``; each such row later
settled into the record of the earliest real entrant on the market — 280
tokens booked against a 13.7-token entry put -$138.10 into term_structure's
cell. The guard books over-bound settlements to ``phantom_unattributed``.
"""

from __future__ import annotations

import pytest

from auramaur.broker.ledger import PHANTOM_STRATEGY, record_ledger_event
from auramaur.db.database import Database


async def _seed_trade(db, *, market_id, order_id, strategy, size=13.7,
                      ts="2026-07-22 02:25:18", is_paper=1, side="BUY",
                      status="filled"):
    await db.execute(
        """INSERT INTO trades (market_id, timestamp, side, size, price,
           is_paper, order_id, status, exchange, strategy_source)
           VALUES (?, ?, ?, ?, 0.73, ?, ?, ?, 'polymarket', ?)""",
        (market_id, ts, side, size, is_paper, order_id, status, strategy),
    )
    await db.commit()


@pytest.mark.asyncio
async def test_unexplained_settlement_qty_is_quarantined(tmp_path):
    """The incident shape: 13.7 tokens bought, 280 settled."""
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    try:
        await _seed_trade(db, market_id="m1", order_id="o1",
                          strategy="term_structure")
        await record_ledger_event(
            db, market_id="m1", kind="settlement", token="YES", qty=280.0,
            pnl=-138.10, fees=0, is_paper=True, source_ref="s:1")
        row = await db.fetchone("SELECT strategy_source FROM pnl_ledger")
        assert row["strategy_source"] == PHANTOM_STRATEGY
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_explained_qty_keeps_normal_attribution(tmp_path):
    """Stacked entries still attribute normally: 3 x 10 bought, 28 settled
    sits inside the 2x + 5 bound."""
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    try:
        for i in range(3):
            await _seed_trade(db, market_id="m1", order_id=f"o{i}",
                              strategy="long_horizon", size=10.0,
                              ts=f"2026-07-22 02:25:{18 + i}")
        await record_ledger_event(
            db, market_id="m1", kind="settlement", token="YES", qty=28.0,
            pnl=5.0, fees=0, is_paper=True, source_ref="s:1")
        row = await db.fetchone("SELECT strategy_source FROM pnl_ledger")
        assert row["strategy_source"] == "long_horizon"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_cancelled_orders_do_not_explain_quantity(tmp_path):
    """A cancelled 300-token order bought nothing; only the filled 10 count."""
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    try:
        await _seed_trade(db, market_id="m1", order_id="o-c",
                          strategy="llm", size=300.0, status="cancelled")
        await _seed_trade(db, market_id="m1", order_id="o-f",
                          strategy="llm", size=10.0)
        await record_ledger_event(
            db, market_id="m1", kind="settlement", token="YES", qty=280.0,
            pnl=-100.0, fees=0, is_paper=True, source_ref="s:1")
        row = await db.fetchone("SELECT strategy_source FROM pnl_ledger")
        assert row["strategy_source"] == PHANTOM_STRATEGY
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_mode_scoped_bound(tmp_path):
    """A paper settlement is bounded by PAPER buys — a large live position on
    the same market does not explain paper quantity."""
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    try:
        await _seed_trade(db, market_id="m1", order_id="o-live",
                          strategy="market_maker", size=500.0, is_paper=0)
        await _seed_trade(db, market_id="m1", order_id="o-paper",
                          strategy="bias_harvest", size=10.0, is_paper=1,
                          ts="2026-07-23 00:00:00")
        await record_ledger_event(
            db, market_id="m1", kind="settlement", token="YES", qty=280.0,
            pnl=-100.0, fees=0, is_paper=True, source_ref="s:1")
        row = await db.fetchone("SELECT strategy_source FROM pnl_ledger")
        assert row["strategy_source"] == PHANTOM_STRATEGY
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_sell_realizations_are_not_bounded(tmp_path):
    """The guard is settlement-only; a sell realizes an actual fill whose
    quantity is already ground truth."""
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    try:
        await _seed_trade(db, market_id="m1", order_id="o1",
                          strategy="weather_temp", size=10.0)
        await record_ledger_event(
            db, market_id="m1", kind="sell", token="YES", qty=280.0,
            pnl=1.0, fees=0, is_paper=True, source_ref="s:1")
        row = await db.fetchone("SELECT strategy_source FROM pnl_ledger")
        assert row["strategy_source"] == "weather_temp"
    finally:
        await db.close()
