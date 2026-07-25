from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from auramaur.data_edge import (
    DataDelivery,
    _last_healthy_recorded,
    record_delivery,
    record_market_snapshot,
    snapshot_id,
)
from auramaur.data_sources.aggregator import Aggregator
from auramaur.db.database import Database
from auramaur.monitoring.readiness import check_strategy_data_delivery


@pytest.fixture
async def db(tmp_path):
    instance = Database(str(tmp_path / "edge.db"))
    await instance.connect()
    yield instance
    await instance.close()


@pytest.mark.asyncio
async def test_record_delivery_persists_point_in_time_metadata(db):
    source_at = datetime.now(timezone.utc) - timedelta(seconds=4)
    await record_delivery(db, DataDelivery(
        strategy="market_maker", component="order_book", status="ok",
        provider="clob", snapshot_id="snap", source_at=source_at,
        item_count=2, required_fields=("best_bid", "best_ask"),
    ))
    row = await db.fetchone("SELECT * FROM strategy_data_deliveries")
    assert row["strategy"] == "market_maker"
    assert row["snapshot_id"] == "snap"
    assert 3 <= row["age_seconds"] <= 6


@pytest.mark.asyncio
async def test_aggregator_outer_deadline_marks_timeout(db):
    class Slow:
        source_name = "slow"
        categories = None

        async def fetch(self, query, limit=20):
            await asyncio.sleep(0.2)
            return []

        async def close(self):
            return None

    class Observer:
        def __init__(self):
            self.db = db
            self.rows = None

        def ingestion(self, **kwargs):
            self.rows = kwargs["fetch_rows"]

        async def close(self):
            return None

    observer = Observer()
    out = await Aggregator(
        [Slow()], observer, source_timeout_seconds=0.01).gather("q")
    assert out == []
    assert observer.rows[0][2] == "timeout"


@pytest.mark.asyncio
async def test_strategy_readiness_fails_stale_observed_requirement(db):
    """A fresh record actively reporting 'stale' on a fail-closed contract
    (term_structure's per-cycle market snapshot) is a hard failure."""
    await db.execute(
        "INSERT INTO strategy_heartbeats "
        "(strategy,last_beat_at,interval_seconds) VALUES ('term_structure',?,30)",
        (datetime.now(timezone.utc).isoformat(),))
    await record_delivery(db, DataDelivery(
        strategy="term_structure", component="market_snapshot", status="stale",
        source_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        item_count=1,
    ))
    result = await check_strategy_data_delivery(
        db, since=datetime.now(timezone.utc) - timedelta(hours=1))
    assert result.status == "FAIL"
    assert "term_structure:market_snapshot" in result.detail


@pytest.mark.asyncio
async def test_strategy_readiness_parses_sqlite_heartbeat_timestamp(db):
    await db.execute(
        "INSERT INTO strategy_heartbeats "
        "(strategy,last_beat_at,interval_seconds) "
        "VALUES ('term_structure',datetime('now'),30)")
    await record_delivery(db, DataDelivery(
        strategy="term_structure", component="market_snapshot", status="stale",
        source_at=datetime.now(timezone.utc) - timedelta(minutes=10), item_count=1))
    result = await check_strategy_data_delivery(
        db, since=datetime.now(timezone.utc) - timedelta(hours=1))
    assert result.status == "FAIL"


@pytest.mark.asyncio
async def test_strategy_readiness_fails_old_required_record(db):
    """Fail-closed contracts are for components recorded unconditionally every
    cycle: an expired record there means the pipeline stopped delivering."""
    await db.execute(
        "INSERT INTO strategy_heartbeats "
        "(strategy,last_beat_at,interval_seconds) VALUES ('term_structure',?,30)",
        (datetime.now(timezone.utc).isoformat(),))
    old = datetime.now(timezone.utc) - timedelta(hours=3)
    await db.execute(
        "INSERT INTO strategy_data_deliveries "
        "(delivery_id,strategy,component,status,observed_at,item_count) "
        "VALUES ('d1','term_structure','market_snapshot','ok',?,10)",
        (old.isoformat(),))
    result = await check_strategy_data_delivery(
        db, since=datetime.now(timezone.utc) - timedelta(hours=1))
    assert result.status == "FAIL"
    assert "term_structure:market_snapshot" in result.detail


@pytest.mark.asyncio
async def test_strategy_readiness_tolerates_aged_conditional_record(db):
    """Fail-open contracts (FRED cache hits, per-market books) record only
    when an event materializes; an aged 'ok' record is missing telemetry, not
    a delivery failure — it must not turn the criterion red."""
    await db.execute(
        "INSERT INTO strategy_heartbeats "
        "(strategy,last_beat_at,interval_seconds) VALUES ('settlement_arb',?,30)",
        (datetime.now(timezone.utc).isoformat(),))
    old = datetime.now(timezone.utc) - timedelta(hours=3)
    await db.execute(
        "INSERT INTO strategy_data_deliveries "
        "(delivery_id,strategy,component,status,observed_at,item_count) "
        "VALUES ('d1','settlement_arb','fred_observations','ok',?,5)",
        (old.isoformat(),))
    await record_delivery(db, DataDelivery(
        strategy="settlement_arb", component="market_snapshot", status="ok",
        source_at=datetime.now(timezone.utc), item_count=10))
    result = await check_strategy_data_delivery(
        db, since=datetime.now(timezone.utc) - timedelta(hours=1))
    assert result.status == "INSUFFICIENT_DATA"
    assert "settlement_arb:fred_observations" in result.detail


@pytest.mark.asyncio
async def test_strategy_readiness_rejects_fresh_partial_delivery(db):
    await db.execute(
        "INSERT INTO strategy_heartbeats "
        "(strategy,last_beat_at,interval_seconds) VALUES ('term_structure',?,30)",
        (datetime.now(timezone.utc).isoformat(),))
    await record_delivery(db, DataDelivery(
        strategy="term_structure", component="market_snapshot", status="partial",
        source_at=datetime.now(timezone.utc), item_count=0))
    result = await check_strategy_data_delivery(
        db, since=datetime.now(timezone.utc) - timedelta(hours=1))
    assert result.status == "FAIL"


@pytest.mark.asyncio
async def test_strategy_readiness_rejects_missing_source_timestamp(db):
    await db.execute(
        "INSERT INTO strategy_heartbeats "
        "(strategy,last_beat_at,interval_seconds) VALUES ('term_structure',?,30)",
        (datetime.now(timezone.utc).isoformat(),))
    await record_delivery(db, DataDelivery(
        strategy="term_structure", component="market_snapshot", status="ok",
        item_count=2))
    result = await check_strategy_data_delivery(
        db, since=datetime.now(timezone.utc) - timedelta(hours=1))
    assert result.status == "FAIL"
    assert "missing source time" in result.detail


@pytest.mark.asyncio
async def test_strategy_readiness_rejects_unknown_strategy_contract(db):
    await db.execute(
        "INSERT INTO strategy_heartbeats "
        "(strategy,last_beat_at,interval_seconds) VALUES ('new_book',?,30)",
        (datetime.now(timezone.utc).isoformat(),))
    result = await check_strategy_data_delivery(
        db, since=datetime.now(timezone.utc) - timedelta(hours=1))
    assert result.status == "FAIL"
    assert "unknown data contract" in result.detail


@pytest.mark.asyncio
async def test_healthy_book_pulses_are_throttled_but_failure_is_kept(db):
    _last_healthy_recorded.clear()
    healthy = DataDelivery(
        strategy="market_maker", component="order_book", status="ok",
        item_count=2)
    await record_delivery(db, healthy)
    await record_delivery(db, healthy)
    await record_delivery(db, healthy.model_copy(update={"status": "timeout"}))
    row = await db.fetchone(
        "SELECT COUNT(*) n FROM strategy_data_deliveries "
        "WHERE strategy='market_maker' AND component='order_book'")
    assert row["n"] == 2


def test_snapshot_id_is_stable_and_order_sensitive():
    assert snapshot_id("a", 1) == snapshot_id("a", 1)
    assert snapshot_id("a", 1) != snapshot_id(1, "a")


class _Mkt:
    def __init__(self, yes=None, no=None):
        self.outcome_yes_price = yes
        self.outcome_no_price = no


@pytest.mark.asyncio
async def test_market_snapshot_partial_only_when_nothing_priced(db):
    """A few unpriced markets are routine (strategies skip them): the snapshot
    is 'ok' while at least one market is priced, 'partial' only when rows
    arrived but none are usable."""
    await record_market_snapshot(db, "term_structure",
                                 [_Mkt(0.4, 0.6), _Mkt()], provider="t1")
    await record_market_snapshot(db, "term_structure",
                                 [_Mkt(), _Mkt()], provider="t2")
    rows = await db.fetchall(
        "SELECT provider,status,detail FROM strategy_data_deliveries "
        "WHERE component='market_snapshot' ORDER BY provider")
    assert [(r["provider"], r["status"]) for r in rows] == [
        ("t1", "ok"), ("t2", "partial")]
    assert '"missing_market_count": 1' in rows[0]["detail"]
