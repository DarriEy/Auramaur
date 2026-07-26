from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from auramaur.db.database import Database
from auramaur.experiments.models import (
    ExperimentDefinition,
    MarketSnapshot,
    PortfolioSnapshot,
    PricePoint,
    TargetPosition,
)
from auramaur.experiments.registry import ExperimentRegistry
from auramaur.experiments.runtimes import LinearExecutionModel, ReplayRuntime
from auramaur.research.experiment_repository import ExperimentRepository
from auramaur.research.experiment_reporting import (
    CriterionStatus,
    ExperimentReportBuilder,
    ReportStatus,
)

NOW = datetime(2026, 7, 25, tzinfo=timezone.utc)


class HoldTen:
    async def evaluate(self, snapshot, portfolio):
        del portfolio
        price = snapshot.price("m1:YES")
        return [TargetPosition(
            instrument_id="m1:YES",
            target_quantity=10,
            reference_price=price,
            rationale="fixture",
            max_notional=10,
        )]


def _registered(**updates):
    values = dict(
        key="persisted-replay",
        strategy_source="persisted_replay",
        hypothesis="The fixture has positive net expectancy.",
        mechanism="test fixture",
        implementation_version="1",
        parameters={"quantity": 10},
        venues=frozenset({"paper"}),
        primary_metric="net_pnl_after_costs",
        baseline="no_position",
        min_observations=30,
        holdout_days=14,
        max_drawdown=0.10,
        cost_model="linear-v1",
        rejection_criteria=("holdout_pnl_lte_zero",),
    )
    values.update(updates)
    definition = ExperimentDefinition(**values)
    return ExperimentRegistry().register(definition, HoldTen())


def _snapshot(price: float, seconds: int, sequence: int) -> MarketSnapshot:
    return MarketSnapshot(
        observed_at=NOW + timedelta(seconds=seconds),
        sequence=sequence,
        market_id="m1",
        venue="paper",
        prices=(PricePoint(instrument_id="m1:YES", price=price),),
        data_version="fixture-data-v1",
    )


async def _report(registered):
    return await ReplayRuntime(LinearExecutionModel(
        version="linear-v1", fee_bps=10, slippage_bps=10
    )).run(
        registered,
        (_snapshot(0.40, 0, 0), _snapshot(0.50, 1, 1)),
        PortfolioSnapshot(
            observed_at=NOW - timedelta(seconds=1), cash=1000, equity=1000
        ),
    )


@pytest.mark.asyncio
async def test_registration_uses_database_time_and_preserves_complete_contract(tmp_path):
    db = Database(str(tmp_path / "experiments.db"))
    await db.connect()
    registered = _registered()
    record = await ExperimentRepository(db).register(registered)
    row = await db.fetchone(
        """SELECT config_json,
                  julianday(holdout_starts_at)-julianday(registered_at) AS days
           FROM strategy_experiments WHERE strategy_version=?""",
        (registered.lineage_id,),
    )
    assert record.lineage_id == registered.lineage_id
    assert row["config_json"] == registered.definition.registration_json
    assert json.loads(row["config_json"])["rejection_criteria"] == [
        "holdout_pnl_lte_zero"
    ]
    assert row["days"] == pytest.approx(14)
    await db.close()


@pytest.mark.asyncio
async def test_replay_is_idempotent_research_evidence_and_never_graduation_input(tmp_path):
    db = Database(str(tmp_path / "experiments.db"))
    await db.connect()
    registered = _registered()
    report = await _report(registered)
    repository = ExperimentRepository(db)
    assert await repository.record_replay(registered, report) == 2
    assert await repository.record_replay(registered, report) == 0

    evaluations = await db.fetchall(
        "SELECT * FROM strategy_evaluations ORDER BY observed_at"
    )
    assert len(evaluations) == 2
    assert all(row["is_paper"] == 1 for row in evaluations)
    payload = json.loads(evaluations[0]["payload_json"])
    assert payload["lineage_id"] == registered.lineage_id
    assert payload["runtime"] == "replay"
    assert payload["data_version"] == "fixture-data-v1"
    assert payload["execution_model_version"] == "linear-v1"

    # These are the tables read by prospective graduation. Replay touches none.
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM decision_snapshots"))["n"] == 0
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM pnl_ledger"))["n"] == 0
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM fills"))["n"] == 0
    await db.close()


@pytest.mark.asyncio
async def test_lineage_and_execution_model_mismatches_fail_closed(tmp_path):
    db = Database(str(tmp_path / "experiments.db"))
    await db.connect()
    registered = _registered()
    report = await _report(registered)
    repository = ExperimentRepository(db)
    with pytest.raises(ValueError, match="lineage"):
        await repository.record_replay(
            registered, replace(report, lineage_id="0" * 64)
        )
    with pytest.raises(ValueError, match="execution-model"):
        await repository.record_replay(
            registered, replace(report, execution_model_version="other")
        )
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM strategy_evaluations"))["n"] == 0
    await db.close()


@pytest.mark.asyncio
async def test_existing_registration_collision_fails_closed(tmp_path):
    db = Database(str(tmp_path / "experiments.db"))
    await db.connect()
    registered = _registered()
    await db.execute(
        """INSERT INTO strategy_experiments
               (strategy_version,strategy_source,config_json,holdout_starts_at)
           VALUES (?, 'wrong', '{}', datetime('now'))""",
        (registered.lineage_id,),
    )
    with pytest.raises(ValueError, match="conflicts"):
        await ExperimentRepository(db).register(registered)
    await db.close()


@pytest.mark.asyncio
async def test_same_second_lineage_registration_rolls_back_as_ambiguous(tmp_path):
    db = Database(str(tmp_path / "experiments.db"))
    await db.connect()
    registered = _registered()
    await db.execute(
        """INSERT INTO strategy_experiments
               (strategy_version,strategy_source,config_json,registered_at,
                holdout_starts_at)
           VALUES ('older', ?, '{}', datetime('now'), datetime('now'))""",
        (registered.definition.strategy_source,),
    )
    with pytest.raises(ValueError, match="timestamp collides"):
        await ExperimentRepository(db).register(registered)
    row = await db.fetchone(
        "SELECT COUNT(*) AS n FROM strategy_experiments WHERE strategy_version=?",
        (registered.lineage_id,),
    )
    assert row["n"] == 0
    await db.close()


@pytest.mark.asyncio
async def test_standard_report_separates_replay_from_prospective_evidence(tmp_path):
    db = Database(str(tmp_path / "experiments.db"))
    await db.connect()
    registered = _registered(min_observations=2)
    report = await _report(registered)
    await ExperimentRepository(db).record_replay(registered, report)
    for holdout in (0, 1):
        await db.execute(
            """INSERT INTO decision_snapshots
                   (market_id,strategy_source,side,fair_probability,
                    reference_price,strategy_version,is_holdout)
               VALUES (?,?,'BUY',.6,.5,?,?)""",
            (f"prospective-{holdout}", registered.definition.strategy_source,
             registered.lineage_id, holdout),
        )

    built = await ExperimentReportBuilder(db).build(registered.lineage_id)
    assert built.execution_model_version == "linear-v1"
    assert built.data_versions == ("fixture-data-v1",)
    assert built.replay.observations == 2
    assert built.replay.independent_markets == 1
    # Observation thresholds count chronological events; independent markets
    # remain a separately reported robustness metric.
    assert built.replay.minimum_observations_met is True
    assert built.replay.total_net_pnl == pytest.approx(
        built.replay.final_equity - built.replay.initial_equity
    )
    scenarios = {item.cost_multiplier: item.estimated_net_pnl
                 for item in built.replay.cost_sensitivity}
    assert scenarios[1.0] == pytest.approx(built.replay.total_net_pnl)
    assert scenarios[0.0] > scenarios[1.0] > scenarios[2.0]
    assert built.prospective.warmup_decisions == 1
    assert built.prospective.holdout_decisions == 1
    assert built.criteria[0].status is CriterionStatus.NOT_EVALUABLE
    assert built.status is ReportStatus.INSUFFICIENT
    await db.close()


@pytest.mark.asyncio
async def test_report_evaluates_supported_drawdown_rejection_rule(tmp_path):
    db = Database(str(tmp_path / "experiments.db"))
    await db.connect()
    registered = _registered(
        max_drawdown=0.000001,
        rejection_criteria=("max_drawdown_exceeds_limit",),
        min_observations=1,
    )
    await ExperimentRepository(db).record_replay(registered, await _report(registered))
    built = await ExperimentReportBuilder(db).build(registered.lineage_id)
    assert built.criteria[0].status is CriterionStatus.FAIL
    assert built.status is ReportStatus.REPLAY_FAIL
    await db.close()


@pytest.mark.asyncio
async def test_report_rejects_tampered_replay_payload(tmp_path):
    db = Database(str(tmp_path / "experiments.db"))
    await db.connect()
    registered = _registered()
    await ExperimentRepository(db).record_replay(registered, await _report(registered))
    await db.execute(
        "UPDATE strategy_evaluations SET payload_json='{}' WHERE id=(SELECT MIN(id) FROM strategy_evaluations)"
    )
    with pytest.raises(ValueError, match="invalid replay payload"):
        await ExperimentReportBuilder(db).build(registered.lineage_id)
    await db.close()


@pytest.mark.asyncio
async def test_report_never_blends_data_versions_into_one_equity_curve(tmp_path):
    db = Database(str(tmp_path / "experiments.db"))
    await db.connect()
    registered = _registered()
    report = await _report(registered)
    repository = ExperimentRepository(db)
    await repository.record_replay(registered, report)
    second_results = tuple(
        result.model_copy(update={
            "data_version": "fixture-data-v2",
            "observed_at": result.observed_at + timedelta(days=1),
        })
        for result in report.results
    )
    await repository.record_replay(
        registered, replace(report, results=second_results)
    )
    builder = ExperimentReportBuilder(db)
    with pytest.raises(ValueError, match="select data_version"):
        await builder.build(registered.lineage_id)
    selected = await builder.build(
        registered.lineage_id, data_version="fixture-data-v2"
    )
    assert selected.data_versions == ("fixture-data-v2",)
    assert selected.replay.observations == 2
    await db.close()
