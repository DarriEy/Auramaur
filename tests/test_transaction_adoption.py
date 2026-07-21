"""Phase 3 transaction() adoption — the lineage observer as single writer.

The LineageObserver used to open a SECOND full read-write Database on the
bot's file (contention plan M2). It now runs its multi-statement batches on
the ONE shared connection inside ``Database.transaction()`` — which is what
makes a mid-batch failure atomic: either every row of a batch lands, or none
do. It also no longer owns the connection, so ``close()`` must drain its
queue but leave the shared database open for its owner.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from auramaur.db.database import Database
from auramaur.lineage_observer import LineageObserver


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ingestion_payload(run_id: str = "run-1") -> dict:
    now = _now()
    item = {
        "run_id": run_id, "item_id": "doc-1", "source": "primary",
        "title": "A fact", "url": "https://example.test/a",
        "content_hash": "hash-1", "excerpt": "body", "published_at": now,
        "observed_at": now, "timestamp_quality": "source",
        "relevance_score": 0.9, "rank_position": 1, "market_id": "m1",
        "information_mode": "production",
    }
    return dict(
        run_id=run_id, query="q", category="tech", market_id="m1",
        started_at=now, observed_at=now,
        fetch_rows=[(run_id, "primary", "ok", 1, 12, "", now, "production")],
        raw_items=1, items=[item], active_sources=[],
    )


def _forecast_payload() -> dict:
    return dict(
        market_id="m1", exchange="polymarket", category="tech",
        raw_probability=0.6, calibrated_probability=0.58,
        market_yes_price=0.4, market_no_price=0.6, observed_at=_now(),
        evidence_run_ids=["run-1"], model="test-model",
        strategy_source="llm", config={"a": 1},
    )


@pytest.mark.asyncio
async def test_observer_batches_land_on_the_shared_connection(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    observer = await LineageObserver.create(db)
    try:
        assert observer.db is db  # single in-process writer, no second connection
        observer.ingestion(**_ingestion_payload())
        observer.forecast(**_forecast_payload())
        await observer.flush()

        run = await db.fetchone("SELECT * FROM ingestion_runs")
        fetch = await db.fetchone("SELECT * FROM source_fetches")
        obs = await db.fetchone("SELECT * FROM evidence_observations")
        snap = await db.fetchone("SELECT * FROM forecast_snapshots")
        assert run["status"] == "ok" and run["unique_items"] == 1
        assert fetch["run_id"] == run["id"] and fetch["status"] == "ok"
        assert obs["run_id"] == run["id"] and obs["content_hash"] == "hash-1"
        assert snap["market_id"] == "m1" and snap["raw_probability"] == 0.6
    finally:
        await observer.close()
        await db.close()


@pytest.mark.asyncio
async def test_mid_batch_failure_leaves_zero_partial_rows(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    observer = await LineageObserver.create(db)
    try:
        # The batch's FIRST insert (ingestion_runs) succeeds; the
        # source_fetches executemany that follows fails. transaction() must
        # roll the whole batch back — no half-written run may land.
        await db.execute("DROP TABLE source_fetches")
        await db.commit()

        observer.ingestion(**_ingestion_payload())
        await observer.flush()

        for table in ("ingestion_runs", "evidence_observations"):
            row = await db.fetchone(f"SELECT COUNT(*) AS n FROM {table}")
            assert row["n"] == 0, f"partial rows stranded in {table}"

        # The worker survives the rollback and later batches still land.
        observer.forecast(**_forecast_payload())
        await observer.flush()
        row = await db.fetchone("SELECT COUNT(*) AS n FROM forecast_snapshots")
        assert row["n"] == 1
    finally:
        await observer.close()
        await db.close()


@pytest.mark.asyncio
async def test_close_is_idempotent_and_leaves_shared_db_open(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    observer = await LineageObserver.create(db)

    observer.forecast(**_forecast_payload())
    # Bot shutdown reaches close() twice: once via the component registry
    # (before the db closes) and again via Aggregator.close().
    await observer.close()
    await observer.close()

    # The queued write was drained, and the SHARED connection is still open —
    # closing it is the owner's job, not the observer's.
    row = await db.fetchone("SELECT COUNT(*) AS n FROM forecast_snapshots")
    assert row["n"] == 1
    await db.close()
