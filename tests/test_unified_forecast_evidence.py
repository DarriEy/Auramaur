from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from auramaur.db.database import Database
from auramaur.evaluation.domain import EpisodeSnapshot, EvaluationForecast, EvaluationRun, RunStatus
from auramaur.evaluation.evidence import (
    ForecastScoreMaterializer, MarketOutcome, OutcomeRepository, event_key,
)
from auramaur.evaluation.store import EvaluationStore
from auramaur.exchange.models import Market
from auramaur.strategy.resolution_tracker import ResolutionTracker

NOW = datetime(2026, 7, 21, tzinfo=timezone.utc)


async def _seed_streams(db):
    await db.execute(
        """INSERT INTO forecast_snapshots
           (market_id,exchange,raw_probability,calibrated_probability,
            market_yes_price,observed_at,model,strategy_source)
           VALUES ('m1','polymarket',.7,.65,.55,?,'prod','strategic')""",
        ((NOW - timedelta(days=2)).isoformat(),),
    )
    await db.commit()
    store = EvaluationStore(db)
    episode = EpisodeSnapshot(
        venue="polymarket", market_id="m1", event_family="family-1",
        observed_at=NOW - timedelta(days=2), evidence_cutoff=NOW - timedelta(days=2),
        market_prob_yes=.55, question="Will it happen?",
    )
    await store.put_episode(episode)
    await store.put_run(EvaluationRun(
        run_id="r1", arm_name="local_single", model="qwen3:8b",
        exploration_policy="single", prompt_version="v1", output_schema_version="v1",
        status=RunStatus.SUCCEEDED, started_at=NOW, completed_at=NOW,
    ))
    await store.put_forecast(EvaluationForecast(
        forecast_id="ef1", run_id="r1", episode_hash=episode.episode_hash,
        prob_yes=.8, action="YES",
    ))
    await db.execute(
        """INSERT INTO information_strategies
           (id,source,category,horizon,event_type) VALUES ('s1','official','tech','week','event')""")
    await db.execute(
        """INSERT INTO information_trials
           (id,strategy_id,market_id,observed_at,assignment,assignment_hash,market_price)
           VALUES ('t1','s1','m1',?,'treatment','hash',.55)""",
        ((NOW - timedelta(days=2)).isoformat(),),
    )
    await db.execute(
        "INSERT INTO paired_forecasts (trial_id,arm,probability) VALUES ('t1','treatment',.75)")
    await db.commit()
    return episode


async def test_canonical_outcomes_are_venue_qualified_and_immutable(tmp_path):
    db = Database(str(tmp_path / "outcomes.db"))
    await db.connect()
    try:
        repo = OutcomeRepository(db)
        poly = MarketOutcome("polymarket", "same-id", 1, NOW)
        assert event_key("PolyMarket", "same-id") == "polymarket:same-id"
        assert await repo.record(poly) is True
        assert await repo.record(poly) is False
        assert await repo.record(MarketOutcome("kalshi", "same-id", 0, NOW)) is True
        with pytest.raises(ValueError, match="conflicting"):
            await repo.record(MarketOutcome("polymarket", "same-id", 0, NOW))
    finally:
        await db.close()


async def test_view_normalizes_streams_and_materializes_event_weighted_scores(tmp_path):
    db = Database(str(tmp_path / "unified.db"))
    await db.connect()
    try:
        await _seed_streams(db)
        await OutcomeRepository(db).record(MarketOutcome("polymarket", "m1", 1, NOW))
        rows = await db.fetchall(
            "SELECT stream,probability_kind FROM unified_forecast_evidence "
            "WHERE outcome=1 ORDER BY stream,probability_kind")
        assert [(r["stream"], r["probability_kind"]) for r in rows] == [
            ("information_trial", "trial"),
            ("intelligence_eval", "single"),
            ("production", "calibrated"),
            ("production", "raw"),
        ]
        materializer = ForecastScoreMaterializer(db)
        assert await materializer.refresh() == 4
        assert await materializer.refresh() == 4
        facts = await db.fetchall("SELECT * FROM forecast_score_facts")
        assert len(facts) == 4
        assert all(row["horizon_bucket"] == "1-7d" for row in facts)
        summary = await materializer.event_weighted_summary()
        assert sum(row["events"] for row in summary) == 4
        local = next(row for row in summary if row["stream"] == "intelligence_eval")
        assert local["brier_delta"] > 0
    finally:
        await db.close()


class _Discovery:
    async def get_market(self, market_id):
        return Market(
            id=market_id, exchange="polymarket", question="Resolved evaluation market",
            active=False, closed=True, outcome_yes_price=1, outcome_no_price=0,
        )


async def test_resolution_tracker_discovers_evaluation_only_market(tmp_path):
    db = Database(str(tmp_path / "tracker.db"))
    await db.connect()
    try:
        episode = EpisodeSnapshot(
            venue="polymarket", market_id="eval-only", observed_at=NOW,
            evidence_cutoff=NOW, market_prob_yes=.6, question="Evaluation only?",
        )
        await EvaluationStore(db).put_episode(episode)
        calibration = AsyncMock()
        tracker = ResolutionTracker(
            db, calibration, {"polymarket": _Discovery()}, proxy_address="")
        assert await tracker.check_resolutions() == 1
        row = await db.fetchone(
            "SELECT outcome,source FROM market_outcomes WHERE event_key='polymarket:eval-only'")
        assert row["outcome"] == 1 and row["source"] == "resolution_tracker"
        calibration.record_resolution.assert_awaited_once_with("eval-only", True)
    finally:
        await db.close()


async def test_resolution_tracker_discovers_lineage_only_market(tmp_path):
    db = Database(str(tmp_path / "lineage-tracker.db"))
    await db.connect()
    try:
        await db.execute(
            """INSERT INTO forecast_snapshots
               (market_id,exchange,raw_probability,market_yes_price,observed_at)
               VALUES ('lineage-only','polymarket',.7,.6,?)""", (NOW.isoformat(),))
        await db.commit()
        tracker = ResolutionTracker(db, None, {"polymarket": _Discovery()})
        assert await tracker.check_resolutions() == 1
        row = await db.fetchone(
            "SELECT outcome FROM market_outcomes WHERE event_key='polymarket:lineage-only'")
        assert row["outcome"] == 1
        scored = await db.fetchone(
            "SELECT brier FROM forecast_score_facts WHERE stream='production'")
        assert scored is not None
    finally:
        await db.close()


async def test_v36_migration_backfills_latest_legacy_outcome(tmp_path):
    path = tmp_path / "v36.db"
    db = Database(str(path))
    await db.connect()
    await db.execute(
        "INSERT INTO markets (id,exchange,question,last_updated) VALUES ('m1','kalshi','Q',?)",
        (NOW.isoformat(),),
    )
    await db.execute(
        """INSERT INTO calibration
           (market_id,predicted_prob,actual_outcome,resolved_at)
           VALUES ('m1',.4,0,'2026-01-01T00:00:00+00:00'),
                  ('m1',.6,1,'2026-01-02T00:00:00+00:00')""")
    await db.execute("DELETE FROM market_outcomes")
    await db.execute("UPDATE schema_version SET version=36")
    await db.commit()
    await db.close()

    migrated = Database(str(path))
    await migrated.connect()
    try:
        row = await migrated.fetchone("SELECT * FROM market_outcomes")
        assert row["event_key"] == "kalshi:m1"
        assert row["outcome"] == 1
        assert row["source"] == "calibration_backfill"
    finally:
        await migrated.close()
