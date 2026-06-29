"""The correlation score is MODE-SCOPED: a paper entry correlates only against
the paper book, a live entry only against the live book. A paper entry adds no
real exposure, so the live book must not crowd out paper exploration (the bug
that choked long_horizon in categories with a big live book). Live behavior is
unchanged."""

from __future__ import annotations

import asyncio

from auramaur.db.database import Database
from auramaur.risk.portfolio import PortfolioTracker


async def _pos(db, mid, category, is_paper):
    await db.execute(
        "INSERT INTO portfolio (market_id, exchange, side, size, avg_price, "
        "current_price, category, is_paper) "
        "VALUES (?, 'polymarket', 'BUY', 5, 0.5, 0.5, ?, ?)",
        (mid, category, 1 if is_paper else 0),
    )
    await db.commit()


def test_correlation_is_mode_scoped():
    async def run():
        db = Database(":memory:")
        await db.connect()
        # 5 LIVE + 1 PAPER same-category ("tech") open positions.
        for i in range(5):
            await _pos(db, f"live{i}", "tech", is_paper=False)
        await _pos(db, "paper0", "tech", is_paper=True)
        # The entry market's category lives in the markets table (not yet held).
        await db.execute(
            "INSERT INTO markets (id, exchange, question, category, active, last_updated) "
            "VALUES ('new', 'polymarket', 'q-new', 'tech', 1, datetime('now'))")
        await db.commit()

        pt = PortfolioTracker(db=db)
        w = PortfolioTracker.CATEGORY_WEIGHT  # 0.3

        # Paper entry: only the 1 paper position counts.
        paper_score = await pt.get_correlated_markets("new", is_paper=True)
        assert abs(paper_score - 1 * w) < 1e-9

        # Live entry: only the 5 live positions count (unchanged behavior).
        live_score = await pt.get_correlated_markets("new", is_paper=False)
        assert abs(live_score - 5 * w) < 1e-9

        # Legacy unscoped (None): both modes — the pre-fix behavior.
        both_score = await pt.get_correlated_markets("new")
        assert abs(both_score - 6 * w) < 1e-9

        # The key property: a paper entry is NOT choked by the live book.
        assert paper_score < live_score
        await db.close()

    asyncio.run(run())
