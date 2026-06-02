"""Strategy attribution now sums cost_basis.realized_pnl (not trades.pnl)."""

import asyncio
from types import SimpleNamespace

from auramaur.db.database import Database
from auramaur.monitoring.attribution import PerformanceAttributor


async def _seed(db):
    # Live (is_paper=0) cost_basis: m1 win, m2 loss, m3 win.
    cb = [("m1", 5.0), ("m2", -3.0), ("m3", 2.0)]
    for mid, pnl in cb:
        await db.execute(
            "INSERT INTO cost_basis (market_id, token, size, avg_cost, total_cost, realized_pnl, is_paper) "
            "VALUES (?,?,0,0,0,?,0)", (mid, "YES", pnl))
    # m1 + m2 attributed via trades.strategy_source; m3 only via signals (fallback).
    await db.execute("INSERT INTO trades (market_id, side, size, price, is_paper, strategy_source, timestamp) "
                     "VALUES ('m1','BUY',1,1,0,'llm','2026-06-01T01:00')")
    await db.execute("INSERT INTO trades (market_id, side, size, price, is_paper, strategy_source, timestamp) "
                     "VALUES ('m2','BUY',1,1,0,'news_speed','2026-06-01T01:00')")
    await db.execute("INSERT INTO markets (id, question, last_updated) VALUES ('m3','q3',datetime('now'))")
    await db.execute("INSERT INTO signals (market_id, claude_prob, claude_confidence, market_prob, edge, strategy_source, timestamp) "
                     "VALUES ('m3',0.5,'MEDIUM',0.4,5,'technical','2026-06-01T01:00')")
    # m4: oracle-resolved (resolution_pnl), attributed to llm. Tests the union.
    await db.execute("INSERT INTO resolution_pnl (market_id, category, pnl, resolved_at) "
                     "VALUES ('m4','crypto',7.0,'2026-06-01T02:00')")
    await db.execute("INSERT INTO trades (market_id, side, size, price, is_paper, strategy_source, timestamp) "
                     "VALUES ('m4','BUY',1,1,0,'llm','2026-06-01T01:30')")
    await db.commit()


def test_strategy_summary_uses_cost_basis():
    async def run():
        db = Database(":memory:")
        await db.connect()
        await _seed(db)
        attr = PerformanceAttributor(db=db)
        rows = {r["strategy_source"]: r for r in await attr.get_strategy_summary(is_live=True)}

        # llm = m1 (cost_basis +5) + m4 (resolution_pnl +7) = +12 across 2 markets.
        assert abs(rows["llm"]["total_pnl"] - 12.0) < 1e-9
        assert rows["llm"]["wins"] == 2 and rows["llm"]["trade_count"] == 2
        assert abs(rows["news_speed"]["total_pnl"] - (-3.0)) < 1e-9
        assert rows["news_speed"]["wins"] == 0
        # m3 attributed to 'technical' via the signals fallback (no trades row).
        assert abs(rows["technical"]["total_pnl"] - 2.0) < 1e-9
        await db.close()

    asyncio.run(run())


def test_strategy_summary_scopes_by_mode():
    async def run():
        db = Database(":memory:")
        await db.connect()
        await _seed(db)  # all is_paper=0 (live)
        attr = PerformanceAttributor(db=db)
        # Paper mode sees none of the live rows.
        assert await attr.get_strategy_summary(is_live=False) == []
        await db.close()

    asyncio.run(run())


if __name__ == "__main__":
    test_strategy_summary_uses_cost_basis()
    test_strategy_summary_scopes_by_mode()
    print("ok")
