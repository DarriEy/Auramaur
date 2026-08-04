"""The scaffolding itself must be trustworthy: an injector that misses
its statement, or a spy that survives a raised span in a broken state,
would make every downstream atomicity test lie."""

import sqlite3

import pytest

from auramaur.db.database import Database
from tests.txn_helpers import failing_on, span_owners, transaction_spy


@pytest.mark.asyncio
async def test_failing_on_hits_only_the_target_statement(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    try:
        with failing_on(db, "INSERT INTO signals"):
            await db.execute(
                "INSERT OR IGNORE INTO markets (id, question) VALUES ('m','q')")
            with pytest.raises(sqlite3.OperationalError, match="injected"):
                await db.execute(
                    "INSERT INTO signals (market_id, claude_prob,"
                    " claude_confidence, market_prob, edge)"
                    " VALUES ('m', 0.5, 'LOW', 0.5, 0)")
        # Restored: the same statement succeeds outside the context.
        await db.execute(
            "INSERT INTO signals (market_id, claude_prob, claude_confidence,"
            " market_prob, edge) VALUES ('m', 0.5, 'LOW', 0.5, 0)")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_spy_records_owner_and_survives_a_raising_span(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    try:
        events: list = []
        with transaction_spy(db, events):
            async with db.transaction(owner="test.batch"):
                await db.execute(
                    "INSERT OR IGNORE INTO markets (id, question)"
                    " VALUES ('m','q')")
            with pytest.raises(RuntimeError):
                async with db.transaction(owner="test.broken"):
                    raise RuntimeError("boom")
        assert span_owners(events) == ["test.batch", "test.broken"]
        assert events.count(("test.broken", "end")) == 1
        # The connection is usable after the raising span (no strand).
        await db.execute(
            "INSERT OR IGNORE INTO markets (id, question) VALUES ('n','q')")
    finally:
        await db.close()
