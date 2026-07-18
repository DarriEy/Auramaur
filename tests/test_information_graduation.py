from datetime import datetime, timezone

import pytest

from auramaur.data_sources.aggregator import Aggregator
from auramaur.data_sources.base import NewsItem
from auramaur.db.database import Database
from auramaur.information_graduation import InformationGraduation
from auramaur.nlp.calibration import CalibrationTracker


class ShadowSource:
    source_name = "candidate"
    categories = None
    information_mode = "shadow"

    async def fetch(self, query, limit=20):
        return [NewsItem(id="shadow-1", source="candidate", title="Official fact")]

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_shadow_evidence_is_persisted_but_withheld():
    db = Database(":memory:")
    await db.connect()
    ladder = InformationGraduation(db)
    result = await Aggregator(
        [ShadowSource()], db=db, information_graduation=ladder,
    ).gather("fact", market_id="m", market_price=.4)
    assert result == []
    row = await db.fetchone("SELECT information_mode,rank_position FROM evidence_observations")
    assert row["information_mode"] == "shadow" and row["rank_position"] is None
    trial = await db.fetchone("SELECT assignment,market_price FROM information_trials")
    assert trial["assignment"] in {"control", "treatment"} and trial["market_price"] == .4
    await db.close()


@pytest.mark.asyncio
async def test_positive_paired_trial_reaches_probation():
    db = Database(":memory:")
    await db.connect()
    ladder = InformationGraduation(db, min_resolved=1, min_paired=1, min_success_rate=.5)
    sid = await ladder.register("candidate", "weather", "0-6h", "alert")
    now = datetime.now(timezone.utc)
    trial, assignment = await ladder.assign(sid, "m1", now, .5)
    assert assignment in {"control", "treatment"}
    # Treatment is closer to the YES resolution and earns more simulated P&L.
    await ladder.record_forecast(trial, "control", .55, net_paper_pnl=0)
    await ladder.record_forecast(trial, "treatment", .80, net_paper_pnl=1)
    await db.execute(
        "INSERT INTO source_fetches (run_id,source,status,observed_at) "
        "VALUES ('r','candidate','ok',datetime('now'))")
    await ladder.resolve(trial, True)
    decision = await ladder.evaluate(sid)
    assert decision.status == "probation" and decision.influence_multiplier == .25
    contribution = await db.fetchone("SELECT * FROM source_contributions")
    assert contribution["incremental_brier"] > 0
    await db.close()


@pytest.mark.asyncio
async def test_assignment_is_immutable_and_deterministic():
    db = Database(":memory:")
    await db.connect()
    ladder = InformationGraduation(db)
    sid = await ladder.register("candidate", "crypto", "1-3d")
    now = datetime.now(timezone.utc)
    first = await ladder.assign(sid, "m", now, .4)
    second = await ladder.assign(sid, "m", now, .9)
    assert first == second
    row = await db.fetchone("SELECT market_price FROM information_trials")
    assert row["market_price"] == .4
    await db.close()


@pytest.mark.asyncio
async def test_market_resolution_reconciles_information_trials():
    db = Database(":memory:")
    await db.connect()
    ladder = InformationGraduation(db)
    sid = await ladder.register("candidate", "weather", "0-6h")
    trial, _ = await ladder.assign(
        sid, "resolved-market", datetime.now(timezone.utc), .5,
    )
    await ladder.record_forecast(trial, "control", .4)
    await ladder.record_forecast(trial, "treatment", .8)
    await CalibrationTracker(db).record_prediction("resolved-market", .7, "weather")

    await CalibrationTracker(db).record_resolution("resolved-market", True)

    resolved = await db.fetchone(
        "SELECT resolved_outcome FROM information_trials WHERE id=?", (trial,),
    )
    contribution = await db.fetchone(
        "SELECT incremental_brier FROM source_contributions WHERE trial_id=?", (trial,),
    )
    assert resolved["resolved_outcome"] == 1
    assert contribution["incremental_brier"] > 0
    await db.close()
