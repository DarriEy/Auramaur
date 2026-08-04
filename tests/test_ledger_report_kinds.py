"""Issue #299 item 6: commission ledger rows counted as losing events.

A cost row is bookkeeping, not a trading outcome — it must weigh on the
P&L totals (costs are real money) while never appearing in the win/loss
denominators, where every commission reads as a lost trade."""

import pytest

from auramaur.db.database import Database
from auramaur.monitoring.ledger_report import gather_ledger_report


@pytest.mark.asyncio
async def test_commission_rows_hit_pnl_but_never_the_win_rate(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    try:
        rows = [
            ("m1", "sell", 10.0, 0),         # winning realization
            ("m2", "settlement", -4.0, 0),   # losing realization
            ("ibkr:global_etf:XLV", "commission", -2.0, 0),  # cost row
        ]
        for market_id, kind, pnl, paper in rows:
            await db.execute(
                """INSERT INTO pnl_ledger (market_id, venue, category,
                     strategy_source, kind, token, qty, pnl, fees, is_paper,
                     source_ref)
                   VALUES (?, 'ibkr', 'x', 's', ?, 'YES', 1, ?, 0, ?, ?)""",
                (market_id, kind, pnl, paper, f"{market_id}:{kind}"))
        await db.commit()

        report = await gather_ledger_report(db, is_paper=False)
        total = report["total"]
        assert total["n"] == 2          # realizations only
        assert total["wins"] == 1       # the commission is not a loss event
        assert abs(total["pnl"] - 4.0) < 1e-9  # 10 - 4 - 2: cost still counts
    finally:
        await db.close()
