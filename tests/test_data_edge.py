from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from auramaur.data_edge import (
    DataDelivery,
    _last_healthy_recorded,
    record_delivery,
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
    await db.execute(
        "INSERT INTO strategy_heartbeats "
        "(strategy,last_beat_at,interval_seconds) VALUES ('market_maker',?,30)",
        (datetime.now(timezone.utc).isoformat(),))
    await record_delivery(db, DataDelivery(
        strategy="market_maker", component="order_book", status="stale",
        source_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        item_count=1,
    ))
    result = await check_strategy_data_delivery(
        db, since=datetime.now(timezone.utc) - timedelta(hours=1))
    assert result.status == "FAIL"
    assert "market_maker:order_book" in result.detail


@pytest.mark.asyncio
async def test_strategy_readiness_parses_sqlite_heartbeat_timestamp(db):
    await db.execute(
        "INSERT INTO strategy_heartbeats "
        "(strategy,last_beat_at,interval_seconds) "
        "VALUES ('market_maker',datetime('now'),30)")
    await record_delivery(db, DataDelivery(
        strategy="market_maker", component="order_book", status="stale",
        source_at=datetime.now(timezone.utc) - timedelta(minutes=10), item_count=1))
    result = await check_strategy_data_delivery(
        db, since=datetime.now(timezone.utc) - timedelta(hours=1))
    assert result.status == "FAIL"


@pytest.mark.asyncio
async def test_strategy_readiness_treats_old_healthy_record_as_unobserved(db):
    """Conditional emitters (FRED cache hits, weather priced on demand) stop
    recording between events; an aged 'ok' record is missing telemetry, not a
    delivery failure."""
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
    await db.execute(
        "INSERT INTO strategy_data_deliveries "
        "(delivery_id,strategy,component,status,observed_at,item_count) "
        "VALUES ('d2','settlement_arb','market_snapshot','ok',?,10)",
        (datetime.now(timezone.utc).isoformat(),))
    result = await check_strategy_data_delivery(
        db, since=datetime.now(timezone.utc) - timedelta(hours=1))
    assert result.status == "INSUFFICIENT_DATA"
    assert "settlement_arb:fred_observations" in result.detail


@pytest.mark.asyncio
async def test_strategy_readiness_accepts_fresh_partial_delivery(db):
    await db.execute(
        "INSERT INTO strategy_heartbeats "
        "(strategy,last_beat_at,interval_seconds) VALUES ('market_maker',?,30)",
        (datetime.now(timezone.utc).isoformat(),))
    await record_delivery(db, DataDelivery(
        strategy="market_maker", component="order_book", status="partial",
        source_at=datetime.now(timezone.utc), item_count=0))
    result = await check_strategy_data_delivery(
        db, since=datetime.now(timezone.utc) - timedelta(hours=1))
    assert result.status == "PASS"


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
