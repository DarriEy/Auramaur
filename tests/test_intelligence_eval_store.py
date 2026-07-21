from datetime import datetime, timedelta, timezone

import pytest

from auramaur.db.database import Database
from auramaur.evaluation.domain import (
    EpisodeSnapshot, EvaluationForecast, EvaluationOutcome, EvaluationRun, RunStatus,
)
from auramaur.evaluation.store import EvaluationStore


def episode(**updates):
    values = dict(
        venue="polymarket", market_id="m1", event_family="event-1",
        observed_at=datetime(2026, 7, 21, tzinfo=timezone.utc),
        market_prob_yes=0.6, question="Will it happen?", rules="Venue rules",
        yes_bid=0.59, yes_ask=0.61, bid_depth=10, ask_depth=12,
        evidence_cutoff=datetime(2026, 7, 21, tzinfo=timezone.utc),
        evidence_ids=("news-1",), context={"b": 2, "a": 1},
    )
    values.update(updates)
    return EpisodeSnapshot(**values)


def test_episode_hash_is_canonical_and_requires_aware_time():
    first = episode()
    offset = timezone(timedelta(hours=-6))
    second = episode(
        observed_at=first.observed_at.astimezone(offset),
        evidence_cutoff=first.evidence_cutoff.astimezone(offset),
        context={"a": 1, "b": 2},
    )
    assert first.episode_hash == second.episode_hash
    with pytest.raises(ValueError):
        episode(observed_at=datetime(2026, 7, 21))


async def test_store_round_trip_idempotence_settlement_and_score(tmp_path):
    db = Database(str(tmp_path / "eval.db"))
    await db.connect()
    try:
        store = EvaluationStore(db)
        item = episode()
        assert await store.put_episode(item) == item.episode_hash
        await store.put_episode(item)
        row = await db.fetchone("SELECT COUNT(*) AS n FROM evaluation_episodes")
        assert row["n"] == 1
        restored = await store.get_episode(item.episode_hash)
        assert restored is not None and restored.event_family == "event-1"

        run = EvaluationRun(
            run_id="r1", arm_name="local-single", model="qwen3:8b",
            exploration_policy="single", prompt_version="v1",
            output_schema_version="v1", status=RunStatus.SUCCEEDED,
            started_at=item.observed_at, completed_at=item.observed_at,
        )
        await store.put_run(run)
        forecast = EvaluationForecast(
            forecast_id="f1", run_id="r1", episode_hash=item.episode_hash,
            prob_yes=0.8, action="YES", thesis="Concrete mechanism",
            evidence_ids=("news-1",),
        )
        await store.put_forecast(forecast)
        resolved = EvaluationOutcome(
            episode_hash=item.episode_hash, outcome=1,
            resolved_at=item.observed_at + timedelta(days=1), source="venue",
        )
        await store.settle(resolved)
        await store.settle(resolved)
        score = await store.score("f1")
        assert score is not None
        assert score.brier == pytest.approx(0.04)
        assert score.brier_delta > 0
        with pytest.raises(ValueError, match="conflicting"):
            await store.settle(resolved.model_copy(update={"outcome": 0}))
    finally:
        await db.close()


def test_perfect_market_has_undefined_skill():
    from auramaur.evaluation.scoring import score_forecast
    assert score_forecast(0.8, 1.0, 1).brier_skill is None
