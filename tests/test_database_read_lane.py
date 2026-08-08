from __future__ import annotations

import asyncio

import pytest

from auramaur.db.database import Database


@pytest.mark.asyncio
async def test_wal_reader_does_not_queue_behind_open_write_transaction(tmp_path):
    db = Database(str(tmp_path / "read-lane.db"))
    await db.connect()
    await db.execute("CREATE TABLE sample (value INTEGER)")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def writer():
        async with db.transaction(owner="slow_writer"):
            await db.execute("INSERT INTO sample VALUES (1)")
            entered.set()
            await release.wait()

    task = asyncio.create_task(writer())
    try:
        await entered.wait()

        # A separate WAL reader sees the last committed snapshot immediately;
        # it must not wait on the writer's application-level serializer.
        row = await asyncio.wait_for(
            db.fetchone("SELECT COUNT(*) AS count FROM sample"),
            timeout=0.2,
        )
        assert row["count"] == 0

        release.set()
        await task
        row = await db.fetchone("SELECT COUNT(*) AS count FROM sample")
        assert row["count"] == 1
    finally:
        release.set()
        await task
        await db.close()


@pytest.mark.asyncio
async def test_transaction_internal_reads_stay_on_writer_connection(tmp_path):
    db = Database(str(tmp_path / "read-your-writes.db"))
    await db.connect()
    await db.execute("CREATE TABLE sample (value INTEGER)")
    try:
        async with db.transaction(owner="atomic"):
            await db.execute("INSERT INTO sample VALUES (1)")
            row = await db.fetchone("SELECT COUNT(*) AS count FROM sample")
            assert row["count"] == 1
    finally:
        await db.close()
