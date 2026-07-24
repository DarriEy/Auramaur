"""Tests for the llm_costs_daily view — the unified cost/quota ledger.

daily_stats.api_cost_estimate was never wired; the view joins the four real
tracking tables so a day's inference bill is one query (2026-07-24 audit)."""

import asyncio

from auramaur.db.database import Database


def test_llm_costs_daily_unifies_all_four_sources():
    async def run():
        db = Database(":memory:")
        await db.connect()
        await db.execute(
            "INSERT INTO agent_trader_costs (day, model_alias, calls, usd) "
            "VALUES ('2026-07-24', 'gpro', 5, 0.10)")
        await db.execute(
            "INSERT INTO ibkr_etf_openai_attempts (model_alias, model, status, cost_usd, started_at) "
            "VALUES ('luna', 'gpt-5.6-luna', 'ok', 0.02, '2026-07-24T10:00:00')")
        await db.execute(
            "INSERT INTO llm_call_counter (day, claude_calls) VALUES ('2026-07-24', 150)")
        await db.execute(
            "INSERT INTO local_llm_calls (purpose, model, status, created_at) "
            "VALUES ('distill', 'qwen3:8b', 'ok', '2026-07-24T10:00:00')")
        await db.commit()

        rows = await db.fetchall(
            "SELECT source, calls, cost_usd FROM llm_costs_daily "
            "WHERE day = '2026-07-24' ORDER BY source")
        by_source = {r["source"]: (r["calls"], round(r["cost_usd"], 4)) for r in rows}
        assert by_source["gemini:gpro"] == (5, 0.10)
        assert by_source["openai:luna"] == (1, 0.02)
        assert by_source["claude_cli(quota)"] == (150, 0.0)
        assert by_source["local:qwen3:8b"] == (1, 0.0)

        total = await db.fetchone(
            "SELECT ROUND(SUM(cost_usd), 4) AS usd FROM llm_costs_daily WHERE day='2026-07-24'")
        assert total["usd"] == 0.12
    asyncio.run(run())


def test_migration_v40_creates_view_on_old_schema(tmp_path):
    """A v39 database gains the view through the migration path."""
    async def run():
        path = str(tmp_path / "old.db")
        db = Database(path)
        await db.connect()
        # Simulate a pre-v40 file: drop the view and stamp version 39.
        await db.execute("DROP VIEW IF EXISTS llm_costs_daily")
        await db.execute("UPDATE schema_version SET version = 39")
        await db.commit()
        await db.close()

        db2 = Database(path)
        await db2.connect()  # runs migrations
        row = await db2.fetchone(
            "SELECT name FROM sqlite_master WHERE type='view' AND name='llm_costs_daily'")
        assert row is not None
        await db2.close()
    asyncio.run(run())
