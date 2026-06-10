"""Tests for the unified realized-P&L ledger (pnl_ledger).

The ledger is the single source of truth for realized money: one row per
realization event (sell fill or settlement) carrying venue, entry strategy
and category. These tests lock in:

  1. SELL fills write a ledger row with the realized P&L and resolved context.
  2. cost_basis reads/writes are token-scoped — a NO sell can no longer
     realize against the YES side's basis (post-#78 PKs).
  3. Settlements write a ledger row and only zero the settled token's basis.
  4. The backfill is idempotent and dedupes against forward-written rows
     (same source_ref scheme).
  5. Kraken pair fills (no markets row) get venue/category fallbacks.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from auramaur.broker.ledger import backfill_ledger
from auramaur.broker.pnl import PnLTracker
from auramaur.db.database import Database
from auramaur.exchange.models import Fill, Market, OrderSide, TokenType
from auramaur.strategy.resolution_tracker import ResolutionTracker
from config.settings import Settings


def _fill(market_id: str, side: OrderSide, size: float, price: float,
          token: TokenType = TokenType.YES, fee: float = 0.0,
          is_paper: bool = False) -> Fill:
    return Fill(
        order_id=f"o-{market_id}-{side.value}-{price}",
        market_id=market_id,
        side=side,
        token=token,
        size=size,
        price=price,
        fee=fee,
        is_paper=is_paper,
        timestamp=datetime.now(timezone.utc),
    )


async def _seed_market(db: Database, mid: str, exchange: str = "polymarket",
                       category: str = "tech") -> None:
    await db.execute(
        "INSERT INTO markets (id, exchange, question, category, active, last_updated) "
        "VALUES (?, ?, 'q', ?, 1, datetime('now'))",
        (mid, exchange, category),
    )
    await db.execute(
        "INSERT INTO trades (market_id, timestamp, side, size, price, is_paper, "
        "status, strategy_source) VALUES (?, datetime('now'), 'BUY', 1, 0.5, 0, "
        "'filled', 'llm')",
        (mid,),
    )
    await db.commit()


def test_sell_fill_writes_ledger_row():
    async def run():
        db = Database(":memory:")
        await db.connect()
        await _seed_market(db, "m1")
        tracker = PnLTracker(db, Settings())

        await tracker.record_fill(_fill("m1", OrderSide.BUY, 10, 0.40))
        await tracker.record_fill(_fill("m1", OrderSide.SELL, 10, 0.55, fee=0.10))

        rows = await db.fetchall("SELECT * FROM pnl_ledger")
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["kind"] == "sell"
        assert row["venue"] == "polymarket"
        assert row["category"] == "tech"
        assert row["strategy_source"] == "llm"
        assert row["token"] == "YES"
        assert row["is_paper"] == 0
        # (0.55 - 0.40) * 10 - 0.10 = 1.40
        assert abs(row["pnl"] - 1.40) < 1e-9
        await db.close()

    asyncio.run(run())


def test_sell_realizes_against_token_scoped_basis():
    """YES and NO bases coexist; a NO sell must use the NO basis."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        await _seed_market(db, "m2")
        tracker = PnLTracker(db, Settings())

        await tracker.record_fill(_fill("m2", OrderSide.BUY, 10, 0.80, TokenType.YES))
        await tracker.record_fill(_fill("m2", OrderSide.BUY, 10, 0.20, TokenType.NO))
        await tracker.record_fill(_fill("m2", OrderSide.SELL, 10, 0.30, TokenType.NO))

        row = await db.fetchone(
            "SELECT pnl FROM pnl_ledger WHERE token = 'NO' AND kind = 'sell'"
        )
        # Against the NO basis: (0.30 - 0.20) * 10 = 1.0.
        # Against the YES basis (the old un-scoped read): (0.30 - 0.80) * 10 = -5.0.
        assert row is not None
        assert abs(row["pnl"] - 1.0) < 1e-9

        # YES basis untouched by the NO sell.
        yes = await db.fetchone(
            "SELECT size, avg_cost FROM cost_basis WHERE market_id = 'm2' AND token = 'YES'"
        )
        assert abs(yes["size"] - 10) < 1e-9
        assert abs(yes["avg_cost"] - 0.80) < 1e-9
        await db.close()

    asyncio.run(run())


def test_settlement_writes_ledger_row_token_scoped():
    async def run():
        db = Database(":memory:")
        await db.connect()
        mid = "m3"
        await _seed_market(db, mid, exchange="kalshi", category="politics_intl")
        # Held live NO position entered at 0.30; market resolves NO -> $1.
        await db.execute(
            "INSERT INTO portfolio (market_id, exchange, side, size, avg_price, "
            "current_price, token, is_paper) "
            "VALUES (?, 'kalshi', 'BUY', 10, 0.30, 0.01, 'NO', 0)",
            (mid,),
        )
        # Both tokens have cost_basis rows; only NO may be zeroed.
        for token, size, cost in (("NO", 10, 0.30), ("YES", 5, 0.50)):
            await db.execute(
                "INSERT INTO cost_basis (market_id, token, size, avg_cost, "
                "total_cost, realized_pnl, is_paper) VALUES (?, ?, ?, ?, ?, 0, 0)",
                (mid, token, size, cost, size * cost),
            )
        await db.commit()

        market = Market(
            id=mid, exchange="kalshi", question="q",
            active=False, outcome_yes_price=0.01, outcome_no_price=0.99,
        )
        disc = AsyncMock()
        disc.get_market = AsyncMock(return_value=market)
        tracker = ResolutionTracker(
            db=db, calibration=AsyncMock(), discoveries={"kalshi": disc},
        )
        count = await tracker.check_resolutions()
        assert count == 1

        row = await db.fetchone("SELECT * FROM pnl_ledger WHERE kind = 'settlement'")
        assert row is not None
        # BUY NO @0.30 resolving NO: (1.0 - 0.30) * 10 = 7.0
        assert abs(row["pnl"] - 7.0) < 1e-9
        assert row["token"] == "NO"
        assert row["venue"] == "kalshi"
        assert row["category"] == "politics_intl"

        # The YES row's basis must be untouched by the NO settlement.
        yes = await db.fetchone(
            "SELECT size, realized_pnl FROM cost_basis WHERE market_id = ? AND token = 'YES'",
            (mid,),
        )
        assert abs(yes["size"] - 5) < 1e-9
        assert abs(yes["realized_pnl"]) < 1e-9
        await db.close()

    asyncio.run(run())


def test_backfill_idempotent_and_dedupes_forward_rows():
    async def run():
        db = Database(":memory:")
        await db.connect()
        await _seed_market(db, "m4")
        tracker = PnLTracker(db, Settings())

        # Forward-written sell (ledger row exists with source_ref fill:<id>).
        await tracker.record_fill(_fill("m4", OrderSide.BUY, 10, 0.40))
        await tracker.record_fill(_fill("m4", OrderSide.SELL, 4, 0.50))

        # Residual 6 tokens resolve YES.
        await db.execute(
            "INSERT INTO calibration (market_id, predicted_prob, actual_outcome, "
            "resolved_at, category) VALUES ('m4', 0.6, 1, datetime('now'), 'tech')"
        )
        await db.commit()

        await backfill_ledger(db)
        await backfill_ledger(db)  # idempotent

        rows = await db.fetchall("SELECT kind, pnl FROM pnl_ledger ORDER BY kind")
        kinds = [r["kind"] for r in rows]
        assert kinds == ["sell", "settlement"]
        # sell: (0.50-0.40)*4 = 0.40 ; settlement: (1.0-0.40)*6 = 3.60
        total = sum(r["pnl"] for r in rows)
        assert abs(total - 4.0) < 1e-9
        await db.close()

    asyncio.run(run())


def test_kraken_pair_gets_venue_fallback():
    async def run():
        db = Database(":memory:")
        await db.connect()
        tracker = PnLTracker(db, Settings())

        await tracker.record_fill(_fill("XBTUSDC", OrderSide.BUY, 0.001, 70000.0))
        await tracker.record_fill(_fill("XBTUSDC", OrderSide.SELL, 0.001, 71000.0, fee=0.5))

        row = await db.fetchone("SELECT * FROM pnl_ledger")
        assert row is not None
        assert row["venue"] == "kraken"
        assert row["category"] == "kraken_spot"
        assert row["strategy_source"] == "kraken_directional"
        await db.close()

    asyncio.run(run())
