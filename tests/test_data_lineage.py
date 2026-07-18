from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from auramaur.data_quality import audit_data_contracts, rollup_and_prune_prices
from auramaur.data_sources.aggregator import Aggregator
from auramaur.data_sources.base import NewsItem
from auramaur.db.database import Database
from auramaur.exchange.models import Market
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
async def test_gather_persists_point_in_time_lineage():
    db = Database(":memory:")
    await db.connect()
    items = await Aggregator([Source()], db=db).gather("question", market_id="m1")
    assert items[0].ingestion_run_id
    run = await db.fetchone("SELECT * FROM ingestion_runs")
    obs = await db.fetchone("SELECT * FROM evidence_observations")
    fetch = await db.fetchone("SELECT * FROM source_fetches")
    assert run["status"] == "ok" and run["market_id"] == "m1"
    assert obs["content_hash"] and obs["run_id"] == run["id"]
    assert fetch["status"] == "ok" and fetch["item_count"] == 1
    await db.close()


@pytest.mark.asyncio
async def test_lineage_failure_does_not_drop_evidence():
    db = Database(":memory:")
    await db.connect()
    await db.execute("DROP TABLE ingestion_runs")
    await db.commit()
    items = await Aggregator([Source()], db=db).gather("question", market_id="m1")
    assert [item.title for item in items] == ["A fact"]
    await db.close()


@pytest.mark.asyncio
async def test_contracts_and_hourly_rollup():
    db = Database(":memory:")
    await db.connect()
    old = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
    for minute, price in enumerate((0.2, 0.4, 0.3)):
        stamp = datetime.strptime(old, "%Y-%m-%d %H:%M:%S") + timedelta(minutes=minute)
        await db.execute(
            "INSERT INTO price_history (market_id,exchange,price,recorded_at) VALUES ('m','kalshi',?,?)",
            (price, stamp.strftime("%Y-%m-%d %H:%M:%S")),
        )
    await db.commit()
    assert await audit_data_contracts(db) == []
    await rollup_and_prune_prices(db, raw_retention_days=8, rollup_after_days=7)
    row = await db.fetchone("SELECT * FROM price_history_hourly")
    assert (row["exchange"], row["open"], row["high"], row["low"], row["close"], row["samples"]) == (
        "kalshi", 0.2, 0.4, 0.2, 0.3, 3,
    )
    assert (await db.fetchone("SELECT COUNT(*) n FROM price_history"))["n"] == 0
    await db.close()


@pytest.mark.asyncio
async def test_rollup_rejects_gap_that_would_lose_raw_data():
    db = Database(":memory:")
    await db.connect()
    with pytest.raises(ValueError, match="must be <="):
        await rollup_and_prune_prices(db, raw_retention_days=8, rollup_after_days=20)
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
    settings = Settings()
    calibration = CalibrationTracker(db, min_samples=30)
    engine = TradingEngine(
        settings=settings, db=db, discovery=SimpleNamespace(),
        aggregator=Aggregator([Source()], db=db), analyzer=Analyzer(), cache=None,
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
    for table in ("ingestion_runs", "source_fetches", "evidence_observations",
                  "forecast_snapshots", "calibration"):
        assert (await db.fetchone(f"SELECT COUNT(*) n FROM {table}"))["n"] > 0

    await calibration.record_resolution(market_id, True)
    snapshot = await db.fetchone(
        "SELECT * FROM forecast_snapshots WHERE market_id=?", (market_id,),
    )
    assert snapshot["actual_outcome"] == 1
    assert snapshot["market_yes_price"] == 0.40
    assert snapshot["evidence_run_ids"] != "[]"
    await db.close()

    import scripts.research.edge_vs_market as report
    old_db = report.DB
    report.DB = str(db_path)
    try:
        report.main()
    finally:
        report.DB = old_db
    assert "Insufficient data: 1 resolved point-in-time forecasts" in capsys.readouterr().out
