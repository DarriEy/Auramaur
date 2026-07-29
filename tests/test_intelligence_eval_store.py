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


# ---------------------------------------------------------------------------
# Scorecard trustworthiness (2026-07-29)
# ---------------------------------------------------------------------------


async def _fact(db, **kw):
    """Insert one forecast_score_fact directly — these guard the read model."""
    values = dict(
        forecast_key="fk1", event_key="polymarket:m1", event_family="fam-1",
        stream="intelligence_eval", arm="local_single",
        probability_kind="single", observed_at="2026-07-21T00:00:00+00:00",
        horizon_bucket="0-1d", outcome=1, brier=0.09, log_loss=0.3,
        market_brier=0.16, brier_delta=0.07, brier_skill=0.4,
        score_version="binary-proper-v1", prompt_version="forecast-v2",
        abstained=0,
    )
    values.update(kw)
    cols = ",".join(values)
    await db.execute(
        f"INSERT INTO forecast_score_facts ({cols}) "
        f"VALUES ({','.join('?' * len(values))})", tuple(values.values()))
    await db.commit()


@pytest.mark.asyncio
async def test_newer_prompt_version_is_not_hidden_by_the_older_one(tmp_path):
    """v1 and v2 share every other partition key and v1 is always earlier, so
    without prompt_version in the window the dedup keeps v1 and v2 vanishes
    from the scorecard entirely — silently."""
    from auramaur.evaluation.evidence import ForecastScoreMaterializer

    db = Database(str(tmp_path / "facts.db"))
    await db.connect()
    try:
        await _fact(db, forecast_key="v1", prompt_version="forecast-v1",
                    observed_at="2026-07-20T00:00:00+00:00", brier=0.25)
        await _fact(db, forecast_key="v2", prompt_version="forecast-v2",
                    observed_at="2026-07-21T00:00:00+00:00", brier=0.09)
        rows = await ForecastScoreMaterializer(db).event_weighted_summary()
        versions = {r["prompt_version"] for r in rows}
        assert versions == {"forecast-v1", "forecast-v2"}
        by_version = {r["prompt_version"]: r for r in rows}
        assert by_version["forecast-v2"]["brier"] == pytest.approx(0.09)
        assert by_version["forecast-v1"]["brier"] == pytest.approx(0.25)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_abstentions_are_counted_but_not_scored(tmp_path):
    """"No opinion" is not a prediction. It must not be averaged into a Brier,
    and an all-abstained arm must report NULL rather than a manufactured 0."""
    from auramaur.evaluation.evidence import ForecastScoreMaterializer

    db = Database(str(tmp_path / "abstain.db"))
    await db.connect()
    try:
        await _fact(db, forecast_key="a", event_family="fam-1",
                    abstained=0, brier=0.09)
        await _fact(db, forecast_key="b", event_family="fam-2",
                    abstained=1, brier=0.25)
        rows = await ForecastScoreMaterializer(db).event_weighted_summary()
        row = next(r for r in rows if r["arm"] == "local_single")
        assert row["events"] == 2          # both observations counted
        assert row["abstentions"] == 1
        assert row["scored"] == 1
        assert row["brier"] == pytest.approx(0.09)   # NOT (0.09+0.25)/2

        # An arm that only ever abstained has no Brier at all.
        await _fact(db, forecast_key="c", arm="all_abstain",
                    event_family="fam-3", abstained=1, brier=0.25)
        rows = await ForecastScoreMaterializer(db).event_weighted_summary()
        silent = next(r for r in rows if r["arm"] == "all_abstain")
        assert silent["abstentions"] == 1
        assert silent["brier"] is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_lookahead_rows_are_not_materialised(tmp_path):
    """A market that resolved before it was observed was never a prospective
    forecast. _horizon_bucket clamps the negative lag to zero, which used to
    file these under '0-1d' instead of rejecting them."""
    from auramaur.evaluation.evidence import _resolved_before_observed

    assert _resolved_before_observed(
        "2026-07-21T00:00:00+00:00", "2026-07-20T00:00:00+00:00") is True
    assert _resolved_before_observed(
        "2026-07-20T00:00:00+00:00", "2026-07-21T00:00:00+00:00") is False
    # Unparseable timestamps drop the guard, not the row.
    assert _resolved_before_observed("nonsense", "2026-07-21") is False
    assert _resolved_before_observed(None, None) is False
