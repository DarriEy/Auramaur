"""S3 tests: the agent-vs-bot comparison scorecard."""

import pytest

from auramaur.agentmcp.compare import build_comparison, render_comparison
from auramaur.db.database import Database


async def _seed_ledger(path: str, rows: list[tuple]) -> None:
    """rows = (market_id, category, pnl, is_paper, source_ref)."""
    db = Database(path)
    await db.connect()
    try:
        for mid, cat, pnl, is_paper, ref in rows:
            await db.execute(
                """INSERT INTO pnl_ledger
                   (market_id, venue, category, strategy_source, kind, token,
                    qty, pnl, fees, is_paper, source_ref)
                   VALUES (?, '', ?, '', 'sell', 'YES', 1, ?, 0, ?, ?)""",
                (mid, cat, pnl, is_paper, ref),
            )
        await db.commit()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_empty_agent_book_reports_no_history(tmp_path):
    agent = str(tmp_path / "agent.db")
    bot = str(tmp_path / "auramaur.db")
    await _seed_ledger(bot, [("m1", "crypto", 5.0, 1, "b1")])

    data = await build_comparison(agent, bot)
    assert data["agent_paper"]["events"] == 0
    assert data["verdict"]["agent_has_history"] is False
    # Render must not blow up on an empty agent book.
    assert render_comparison(data) is not None


@pytest.mark.asyncio
async def test_agent_leads_on_realized(tmp_path):
    agent = str(tmp_path / "agent.db")
    bot = str(tmp_path / "auramaur.db")
    # Agent: +10 over 2 events (1 win). Bot paper: +4 over 4 events.
    await _seed_ledger(agent, [
        ("a1", "crypto", 12.0, 1, "a1"),
        ("a2", "tech", -2.0, 1, "a2"),
    ])
    await _seed_ledger(bot, [
        ("b1", "crypto", 1.0, 1, "b1"),
        ("b2", "crypto", 1.0, 1, "b2"),
        ("b3", "tech", 1.0, 1, "b3"),
        ("b4", "tech", 1.0, 1, "b4"),
        ("b5", "crypto", 99.0, 0, "b5"),   # live row — context only
    ])

    data = await build_comparison(agent, bot)
    assert data["agent_paper"]["realized"] == pytest.approx(10.0)
    assert data["bot_paper"]["realized"] == pytest.approx(4.0)
    assert data["bot_live"]["realized"] == pytest.approx(99.0)
    assert data["verdict"]["realized_leader"] == "agent"
    assert data["verdict"]["realized_gap"] == pytest.approx(6.0)
    # $/event: agent 5.0 vs bot 1.0 -> agent leads.
    assert data["verdict"]["per_event_leader"] == "agent"
    # win%: agent 50% (1/2) vs bot 100% (4/4) -> bot leads.
    assert data["verdict"]["win_pct_leader"] == "bot"


@pytest.mark.asyncio
async def test_live_rows_excluded_from_paper_ab(tmp_path):
    """A big bot live win must NOT contaminate the paper head-to-head."""
    agent = str(tmp_path / "agent.db")
    bot = str(tmp_path / "auramaur.db")
    await _seed_ledger(agent, [("a1", "crypto", 3.0, 1, "a1")])
    await _seed_ledger(bot, [("b1", "crypto", 1000.0, 0, "b1")])  # live only

    data = await build_comparison(agent, bot)
    assert data["bot_paper"]["events"] == 0
    assert data["verdict"]["realized_leader"] == "agent"
