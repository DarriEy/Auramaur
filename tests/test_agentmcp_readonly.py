"""Phase-4 read-only policy tests: out-of-process consumers of the BOT's live
database must open it ``mode=ro``, transiently, with a busy_timeout — and can
never write, by construction (kernel-enforced) rather than by convention.

Mirrors tests/test_web_api.py::test_readonly_database_cannot_write for the
agent MCP side.
"""

import aiosqlite
import pytest

from auramaur.agentmcp.compare import _ReadOnlyBotView, build_comparison
from auramaur.agentmcp.market_data import MarketData
from auramaur.db.database import Database


async def _seed_bot_db(path: str) -> None:
    """Full bot schema + one paper ledger row, via the real Database class."""
    db = Database(path)
    await db.connect()
    try:
        await db.execute(
            """INSERT INTO pnl_ledger
               (market_id, venue, category, strategy_source, kind, token,
                qty, pnl, fees, is_paper, source_ref)
               VALUES ('m1', '', 'crypto', '', 'sell', 'YES', 1, 5.0, 0, 1, 'r1')""",
        )
        await db.commit()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_compare_bot_opener_cannot_write(tmp_path):
    """The connection compare.py builds for the BOT's DB rejects writes at the
    SQLite layer — never again the full read-write Database with its DDL."""
    path = str(tmp_path / "auramaur.db")
    await _seed_bot_db(path)

    view = _ReadOnlyBotView(path)
    await view.connect()
    try:
        row = await view.fetchone("SELECT COUNT(*) AS c FROM pnl_ledger")
        assert row["c"] == 1
        with pytest.raises(aiosqlite.OperationalError, match="readonly|attempt to write"):
            await view.db.execute(
                "INSERT INTO pnl_ledger (market_id, venue, category, "
                "strategy_source, kind, token, qty, pnl, fees, is_paper, "
                "source_ref) VALUES ('x', '', '', '', 'sell', 'YES', 1, 0, 0, 1, 'x')")
    finally:
        await view.close()


@pytest.mark.asyncio
async def test_compare_bot_opener_sets_busy_timeout_and_query_only(tmp_path):
    path = str(tmp_path / "auramaur.db")
    await _seed_bot_db(path)

    view = _ReadOnlyBotView(path)
    await view.connect()
    try:
        busy = await view.fetchone("PRAGMA busy_timeout")
        assert busy[0] >= 5000
        qonly = await view.fetchone("PRAGMA query_only")
        assert qonly[0] == 1
    finally:
        await view.close()


@pytest.mark.asyncio
async def test_build_comparison_still_reads_bot_ledger(tmp_path):
    """End to end through the ro opener: same data shape as before."""
    agent = str(tmp_path / "agent.db")
    bot = str(tmp_path / "auramaur.db")
    await _seed_bot_db(bot)

    data = await build_comparison(agent, bot)
    assert data["bot_paper"]["events"] == 1
    assert data["bot_paper"]["realized"] == pytest.approx(5.0)
    assert data["bot_live"]["events"] == 0
    assert data["verdict"]["agent_has_history"] is False


@pytest.mark.asyncio
async def test_market_data_connection_sets_busy_timeout(tmp_path):
    path = str(tmp_path / "auramaur.db")
    await _seed_bot_db(path)

    conn = await MarketData(path)._connect()
    try:
        row = await (await conn.execute("PRAGMA busy_timeout")).fetchone()
        assert row[0] >= 5000
    finally:
        await conn.close()
