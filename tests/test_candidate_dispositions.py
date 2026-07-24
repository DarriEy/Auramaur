"""Every discovered candidate gets one persisted terminal disposition."""
import asyncio

import pytest

from auramaur.db.database import Database
from auramaur.monitoring.candidate_dispositions import CandidateDispositionCycle
from auramaur.strategy.engine_cycle import CycleOrchestrationMixin


def test_candidate_cycle_persists_terminal_rows_and_counts():
    async def run():
        db = Database(":memory:")
        await db.connect()
        cycle = CandidateDispositionCycle(db, "polymarket", cycle_id="cycle-1")
        cycle.offer("m1")
        cycle.offer("m2")
        cycle.offer("m3")
        cycle.mark("m1", "filtered", "category", "sports")
        cycle.mark("m2", "risk-blocked", "risk", "daily loss limit")
        cycle.mark("m3", "executed", "execution", "paper")
        counts = await cycle.flush()
        assert counts == {"filtered": 1, "risk-blocked": 1, "executed": 1}
        rows = await db.fetchall(
            "SELECT market_id, disposition, stage, reason FROM candidate_dispositions "
            "WHERE cycle_id='cycle-1' ORDER BY market_id")
        assert [(r["market_id"], r["disposition"]) for r in rows] == [
            ("m1", "filtered"), ("m2", "risk-blocked"), ("m3", "executed")]
        summary = await db.fetchone(
            "SELECT * FROM candidate_cycle_summaries WHERE cycle_id='cycle-1'")
        assert summary["discovered"] == 3
        assert summary["filtered"] == summary["risk_blocked"] == summary["executed"] == 1
        await db.close()
    asyncio.run(run())


def test_candidate_cycle_rejects_non_spec_disposition():
    cycle = CandidateDispositionCycle(object(), "test")
    with pytest.raises(ValueError):
        cycle.mark("m1", "skipped", "unknown")


def test_engine_funnel_closes_every_discovered_market():
    class Engine(CycleOrchestrationMixin):
        exchange_name = "polymarket"

    class Market:
        def __init__(self, market_id):
            self.id = market_id

    class Decision:
        approved = False
        reason = "category exposure"

    async def run():
        db = Database(":memory:")
        await db.connect()
        engine = Engine()
        engine.db = db
        markets = [Market(f"m{i}") for i in range(1, 6)]
        await engine._persist_cycle_dispositions(
            markets, {m.id for m in markets}, {"m2", "m3", "m4", "m5"},
            {"m3", "m4", "m5"}, {"m4", "m5"},
            [{"market": markets[3], "decision": Decision(), "order": None},
             {"market": markets[4], "decision": None, "order": None,
              "execution_error": "venue timeout"}],
        )
        rows = await db.fetchall(
            "SELECT market_id, disposition FROM candidate_dispositions ORDER BY market_id")
        assert [(r["market_id"], r["disposition"]) for r in rows] == [
            ("m1", "filtered"), ("m2", "filtered"), ("m3", "throttled"),
            ("m4", "risk-blocked"), ("m5", "failed")]
        await db.close()
    asyncio.run(run())


def test_candidate_cycle_prunes_expired_detail_and_summary_rows():
    async def run():
        db = Database(":memory:")
        await db.connect()
        await db.execute("""INSERT INTO candidate_dispositions
            (cycle_id, market_id, disposition, stage, observed_at)
            VALUES ('old-cycle', 'old-market', 'filtered', 'test',
                    datetime('now', '-31 days'))""")
        await db.execute("""INSERT INTO candidate_cycle_summaries
            (cycle_id, discovered, completed_at)
            VALUES ('old-summary', 1, datetime('now', '-91 days'))""")
        cycle = CandidateDispositionCycle(
            db, "polymarket", cycle_id="fresh-cycle",
            retention_days=30, summary_retention_days=90)
        cycle.offer("fresh-market")
        await cycle.flush()
        assert await db.fetchone(
            "SELECT 1 FROM candidate_dispositions WHERE cycle_id='old-cycle'") is None
        assert await db.fetchone(
            "SELECT 1 FROM candidate_cycle_summaries WHERE cycle_id='old-summary'") is None
        assert await db.fetchone(
            "SELECT 1 FROM candidate_dispositions WHERE cycle_id='fresh-cycle'") is not None
        await db.close()
    asyncio.run(run())
