"""S0 smoke test for the Hermes agent-trader MCP bridge.

Proves the pipe end-to-end without Hermes: a fresh ``agent.db`` self-bootstraps,
and ``get_portfolio`` reads it through Auramaur's real accounting (PortfolioTracker
+ PnLTracker) — scoped to paper, isolated from auramaur.db.
"""

import pytest

# The bridge is an OPTIONAL extra (pyproject [agent]); skip cleanly where it
# isn't installed. CI installs it (--extra agent) so this runs for real there.
pytest.importorskip("fastmcp")

from auramaur.agentmcp.server import _read_portfolio  # noqa: E402
from auramaur.db.database import Database  # noqa: E402


@pytest.mark.asyncio
async def test_fresh_agent_db_bootstraps_and_reads_empty(tmp_path):
    """A brand-new agent.db is schema-complete on first connect and snapshots
    as a clean, empty paper book — the minimum proof the pipe is wired."""
    db_path = str(tmp_path / "agent.db")

    snap = await _read_portfolio(db_path)

    assert snap["db_path"] == db_path
    assert snap["is_paper"] is True
    assert snap["open_positions"] == 0
    assert snap["realized_pnl"] == 0.0
    assert snap["unrealized_pnl_marked"] == 0.0
    assert snap["positions"] == []


@pytest.mark.asyncio
async def test_get_portfolio_reflects_seeded_paper_book(tmp_path):
    """A seeded paper position + a realized-ledger row surface in the snapshot,
    with marked unrealized = (last - avg) * size and realized summed from the
    ledger — the same readers the live bot reports through."""
    db_path = str(tmp_path / "agent.db")
    db = Database(db_path)
    await db.connect()
    try:
        await db.execute(
            """INSERT INTO portfolio
               (market_id, exchange, side, size, avg_price, current_price,
                category, token, token_id, is_paper)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
            ("mkt-1", "polymarket", "BUY", 20.0, 0.40, 0.55, "crypto", "YES", "tok-1"),
        )
        await db.execute(
            """INSERT INTO pnl_ledger
               (market_id, venue, category, strategy_source, kind, token,
                qty, pnl, fees, is_paper, source_ref)
               VALUES (?, '', '', 'agent_hermes', 'sell', 'YES', 10, ?, 0, 1, ?)""",
            ("mkt-0", 3.25, "seed-ref-1"),
        )
        await db.commit()
    finally:
        await db.close()

    snap = await _read_portfolio(db_path)

    assert snap["open_positions"] == 1
    pos = snap["positions"][0]
    assert pos["market_id"] == "mkt-1"
    assert pos["size"] == 20.0
    assert pos["avg_price"] == 0.40
    # (0.55 - 0.40) * 20 = 3.0
    assert pos["unrealized"] == pytest.approx(3.0)
    assert snap["unrealized_pnl_marked"] == pytest.approx(3.0)
    assert snap["realized_pnl"] == pytest.approx(3.25)
