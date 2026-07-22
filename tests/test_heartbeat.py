"""Strategy heartbeats — every pillar cycle leaves a persistent trace, so a
silent strategy is distinguishable from a dead task (the 2026-07-22 class of
bug: four pillars indistinguishable from dead in one sweep)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from auramaur.db.database import Database
from auramaur.monitoring.heartbeat import beat, run_pillar_once


class _Pillar:
    name = "test_pillar"

    def __init__(self, entered=0, error=None, detail=None):
        self._entered = entered
        self._error = error
        if detail is not None:
            self.last_cycle_detail = detail

    async def run_once(self):
        if self._error:
            raise self._error
        return self._entered


def test_beat_upserts_and_counts_cycles():
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            await beat(db, "s1", status="ok", entries=2, interval_seconds=300)
            await beat(db, "s1", status="ok", entries=0, interval_seconds=300)
            row = await db.fetchone(
                "SELECT * FROM strategy_heartbeats WHERE strategy = 's1'")
            assert row["cycles"] == 2
            assert row["entries"] == 0
            assert row["status"] == "ok"
            assert row["interval_seconds"] == 300
        finally:
            await db.close()
    asyncio.run(run())


def test_run_pillar_once_records_success_with_detail():
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            pillar = _Pillar(entered=3, detail={"scanned": 10, "in_band": 4})
            n = await run_pillar_once(db, pillar, interval_seconds=600)
            assert n == 3
            row = await db.fetchone(
                "SELECT * FROM strategy_heartbeats WHERE strategy = 'test_pillar'")
            assert row["status"] == "ok"
            assert row["entries"] == 3
            assert '"scanned": 10' in row["detail"]
        finally:
            await db.close()
    asyncio.run(run())


def test_run_pillar_once_records_error_and_reraises():
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            pillar = _Pillar(error=RuntimeError("venue exploded"))
            raised = False
            try:
                await run_pillar_once(db, pillar)
            except RuntimeError:
                raised = True
            assert raised, "the task loop's own error handling must still run"
            row = await db.fetchone(
                "SELECT * FROM strategy_heartbeats WHERE strategy = 'test_pillar'")
            assert row["status"] == "error"
            assert "venue exploded" in row["detail"]
        finally:
            await db.close()
    asyncio.run(run())


def test_beat_never_raises_on_broken_db():
    async def run():
        broken = MagicMock()
        broken.transaction = MagicMock(side_effect=RuntimeError("db gone"))
        await beat(broken, "s1")  # must not raise
    asyncio.run(run())
