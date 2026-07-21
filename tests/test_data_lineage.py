import sqlite3
from types import SimpleNamespace
from uuid import uuid4

import pytest

from auramaur.data_sources.aggregator import Aggregator
from auramaur.data_sources.base import NewsItem
from auramaur.db.database import Database
from auramaur.db.models import SCHEMA_VERSION
from auramaur.data_quality import audit_data_contracts
from auramaur.exchange.models import Market
from auramaur.lineage_observer import LineageObserver
from auramaur.nlp.analyzer import AnalysisResult
from auramaur.nlp.calibration import CalibrationTracker
from auramaur.risk.checks import CheckResult
from auramaur.risk.manager import RiskDecision
from auramaur.strategy.engine import TradingEngine
from config.settings import Settings


class Source:
    source_name = "primary"
    categories = None

    async def fetch(self, query, limit=20):
        return [NewsItem(id="doc-1", source="primary", title="A fact", content="body")]

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_existing_v21_database_runs_all_current_migrations(tmp_path):
    path = tmp_path / "v21.db"
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
    raw.execute("INSERT INTO schema_version VALUES (21)")
    raw.commit()
    raw.close()
    db = Database(str(path))
    await db.connect()
    version = await db.fetchone("SELECT version FROM schema_version")
    table = await db.fetchone(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='forecast_snapshots'"
    )
    assert version["version"] == SCHEMA_VERSION
    assert table["name"] == "forecast_snapshots"
    await db.close()


@pytest.mark.asyncio
async def test_gather_persists_point_in_time_lineage(tmp_path):
    db = Database(str(tmp_path / "lineage.db"))
    await db.connect()
    observer = await LineageObserver.create(db)
    items = await Aggregator([Source()], observer=observer).gather(
        "question", market_id="m1",
    )
    await observer.flush()
    assert items[0].ingestion_run_id
    run = await db.fetchone("SELECT * FROM ingestion_runs")
    obs = await db.fetchone("SELECT * FROM evidence_observations")
    fetch = await db.fetchone("SELECT * FROM source_fetches")
    assert run["status"] == "ok" and run["market_id"] == "m1"
    assert obs["content_hash"] and obs["run_id"] == run["id"]
    assert fetch["status"] == "ok" and fetch["item_count"] == 1
    await observer.close()
    await db.close()


@pytest.mark.asyncio
async def test_lineage_failure_does_not_drop_evidence(tmp_path):
    db = Database(str(tmp_path / "broken.db"))
    await db.connect()
    observer = await LineageObserver.create(db)
    await db.execute("DROP TABLE ingestion_runs")
    await db.commit()
    items = await Aggregator([Source()], observer=observer).gather(
        "question", market_id="m1",
    )
    await observer.flush()
    assert [item.title for item in items] == ["A fact"]
    await observer.close()
    await db.close()


@pytest.mark.asyncio
async def test_data_contract_audit_reports_stuck_ingestion():
    db = Database(":memory:")
    await db.connect()
    await db.execute(
        "INSERT INTO ingestion_runs (id,query,status,started_at) "
        "VALUES ('stuck','q','running',datetime('now','-2 hours'))"
    )
    await db.commit()
    violations = await audit_data_contracts(db)
    assert [(v.contract, v.count) for v in violations] == [("incomplete_ingestion", 1)]
    await db.close()


class Analyzer:
    _model = "integration-model"

    async def analyze(self, market, evidence, cache):
        assert evidence and any(item.ingestion_run_id for item in evidence)
        return AnalysisResult(
            probability=0.72, confidence="HIGH", reasoning="primary evidence",
            key_factors=["fact"],
        )


class Risk:
    async def evaluate(self, signal, market, price_history=None, available_cash=None):
        return RiskDecision(
            approved=False, checks=[CheckResult(name="paper", passed=True)],
            position_size=0, reason="integration observation only", force_paper=True,
        )


class Exchange:
    async def get_balance(self):
        return 1000.0


@pytest.mark.asyncio
async def test_paper_cycle_lineage_through_resolution_and_report(tmp_path, capsys):
    db_path = tmp_path / "cycle.db"
    db = Database(str(db_path))
    await db.connect()
    observer = await LineageObserver.create(db)
    # One in-process writer: the observer shares the bot's connection.
    assert observer.db is db
    settings = Settings()
    calibration = CalibrationTracker(db, min_samples=30, lineage_observer=observer)
    engine = TradingEngine(
        settings=settings, db=db, discovery=SimpleNamespace(),
        aggregator=Aggregator([Source()], observer=observer), analyzer=Analyzer(), cache=None,
        risk_manager=Risk(), exchange=Exchange(), calibration=calibration,
    )
    engine.exchange_name = "polymarket"
    market_id = f"lineage-{uuid4().hex}"
    result = await engine.analyze_market(
        Market(
            id=market_id, exchange="polymarket",
            question="Will the integration fact happen by December 2026?",
            description="Resolves YES when the integration fact is officially confirmed.",
            category="tech", outcome_yes_price=0.40, outcome_no_price=0.60,
            volume=10000, liquidity=5000,
        ),
        place_order=False,
    )
    assert result is not None and result["order"] is None
    await observer.flush()
    for table in ("ingestion_runs", "source_fetches", "evidence_observations",
                  "forecast_snapshots", "calibration"):
        assert (await db.fetchone(f"SELECT COUNT(*) n FROM {table}"))["n"] > 0

    await calibration.record_resolution(market_id, True)
    await observer.flush()
    snapshot = await db.fetchone(
        "SELECT * FROM forecast_snapshots WHERE market_id=?", (market_id,),
    )
    assert snapshot["actual_outcome"] == 1
    assert snapshot["market_yes_price"] == 0.40
    assert snapshot["evidence_run_ids"] != "[]"
    await observer.close()
    await db.close()

    import scripts.research.edge_vs_market as report
    old_db = report.DB
    report.DB = str(db_path)
    try:
        report.main()
    finally:
        report.DB = old_db
    assert "Insufficient data: 1 resolved point-in-time forecasts" in capsys.readouterr().out
