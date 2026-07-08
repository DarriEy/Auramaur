"""Ledger attribution is token-scoped: credit the strategy that opened THIS
side of the market, not whoever touched the market first on any side.

Market-level earliest-entrant attribution booked one strategy's win on the
OPPOSITE token into another strategy's cell whenever two strategies traded
the same market (40+ such markets existed). The token comes from the fill
paired to the trade by order_id; the market-level lookup remains as the
fallback for realizations without a paired fill.
"""

from __future__ import annotations

import pytest

from auramaur.broker.ledger import record_ledger_event
from auramaur.db.database import Database


async def _seed_trade_and_fill(db, *, market_id, order_id, token, strategy,
                               ts, is_paper=1, side="BUY"):
    await db.execute(
        """INSERT INTO trades (market_id, timestamp, side, size, price,
           is_paper, order_id, status, exchange, strategy_source)
           VALUES (?, ?, ?, 10, 0.5, ?, ?, 'filled', 'polymarket', ?)""",
        (market_id, ts, side, is_paper, order_id, strategy),
    )
    await db.execute(
        """INSERT INTO fills (order_id, market_id, token_id, side, token,
           size, price, fee, is_paper, timestamp)
           VALUES (?, ?, 'tok', ?, ?, 10, 0.5, 0, ?, ?)""",
        (order_id, market_id, side, token, is_paper, ts),
    )
    await db.commit()


@pytest.mark.asyncio
async def test_opposite_tokens_attribute_to_their_own_strategies(tmp_path):
    """The incident shape: arm buys YES, the llm engine buys NO minutes later
    on the same market. Each side's realization credits its own strategy."""
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    try:
        await _seed_trade_and_fill(
            db, market_id="m1", order_id="o-yes", token="YES",
            strategy="agent_trader_haiku", ts="2026-07-06 05:25:33")
        await _seed_trade_and_fill(
            db, market_id="m1", order_id="o-no", token="NO",
            strategy="llm", ts="2026-07-06 05:28:01")

        await record_ledger_event(
            db, market_id="m1", kind="settlement", token="YES", qty=10,
            pnl=-5.0, fees=0, is_paper=True, source_ref="s:yes")
        await record_ledger_event(
            db, market_id="m1", kind="settlement", token="NO", qty=10,
            pnl=5.0, fees=0, is_paper=True, source_ref="s:no")

        rows = {r["token"]: r["strategy_source"] for r in await db.fetchall(
            "SELECT token, strategy_source FROM pnl_ledger")}
        assert rows["YES"] == "agent_trader_haiku"
        assert rows["NO"] == "llm"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_mode_scoped_lookup(tmp_path):
    """A paper realization must not inherit the strategy of a LIVE fill on
    the same (market, token)."""
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    try:
        await _seed_trade_and_fill(
            db, market_id="m1", order_id="o-live", token="YES",
            strategy="market_maker", ts="2026-07-01 00:00:00", is_paper=0)
        await _seed_trade_and_fill(
            db, market_id="m1", order_id="o-paper", token="YES",
            strategy="bias_harvest", ts="2026-07-02 00:00:00", is_paper=1)

        await record_ledger_event(
            db, market_id="m1", kind="settlement", token="YES", qty=10,
            pnl=1.0, fees=0, is_paper=True, source_ref="s:p")
        row = await db.fetchone("SELECT strategy_source FROM pnl_ledger")
        assert row["strategy_source"] == "bias_harvest"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_falls_back_to_market_level_without_paired_fill(tmp_path):
    """A trade with no matching fill (legacy rows) still resolves via the
    market-level earliest-entrant lookup."""
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    try:
        await db.execute(
            """INSERT INTO trades (market_id, timestamp, side, size, price,
               is_paper, order_id, status, exchange, strategy_source)
               VALUES ('m1', '2026-07-01 00:00:00', 'BUY', 10, 0.5, 1,
                       'legacy-1', 'filled', 'polymarket', 'long_horizon')""")
        await db.commit()
        await record_ledger_event(
            db, market_id="m1", kind="settlement", token="YES", qty=10,
            pnl=2.0, fees=0, is_paper=True, source_ref="s:1")
        row = await db.fetchone("SELECT strategy_source FROM pnl_ledger")
        assert row["strategy_source"] == "long_horizon"
    finally:
        await db.close()
