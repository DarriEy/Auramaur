"""Standardized, evidence-separated reports for registered experiments."""

from __future__ import annotations

import json
import math
from enum import Enum

from pydantic import BaseModel, ConfigDict

from auramaur.experiments.models import ExperimentDefinition


class CriterionStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_EVALUABLE = "not_evaluable"


class ReportStatus(str, Enum):
    REPLAY_PASS = "replay_pass"
    REPLAY_FAIL = "replay_fail"
    INSUFFICIENT = "insufficient_evidence"


class CriterionResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    criterion: str
    status: CriterionStatus
    reason: str


class CostScenario(BaseModel):
    model_config = ConfigDict(frozen=True)
    cost_multiplier: float
    estimated_net_pnl: float


class ReplayMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)
    observations: int
    independent_markets: int
    initial_equity: float
    final_equity: float
    total_net_pnl: float
    gross_pnl_before_costs: float
    total_costs: float
    turnover: float
    max_drawdown: float
    event_hit_rate: float | None
    event_profit_factor: float | None
    average_gross_exposure: float
    minimum_observations_met: bool
    cost_sensitivity: tuple[CostScenario, ...]


class ProspectiveCounts(BaseModel):
    model_config = ConfigDict(frozen=True)
    warmup_decisions: int
    holdout_decisions: int


class ExperimentReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    lineage_id: str
    strategy_source: str
    registered_at: str
    holdout_starts_at: str
    execution_model_version: str
    data_versions: tuple[str, ...]
    replay: ReplayMetrics
    prospective: ProspectiveCounts
    criteria: tuple[CriterionResult, ...]
    status: ReportStatus


class ExperimentReportBuilder:
    def __init__(self, db) -> None:
        self.db = db

    async def build(
        self,
        lineage_id: str,
        *,
        execution_model_version: str | None = None,
        data_version: str | None = None,
    ) -> ExperimentReport:
        registration = await self.db.fetchone(
            """SELECT strategy_source,config_json,registered_at,holdout_starts_at
               FROM strategy_experiments WHERE strategy_version=?""",
            (lineage_id,),
        )
        if registration is None:
            raise KeyError(f"unknown experiment lineage: {lineage_id}")
        definition = ExperimentDefinition.model_validate(
            json.loads(registration["config_json"])
        )
        if definition.lineage_id != lineage_id:
            raise ValueError("stored registration does not match its lineage")
        rows = await self.db.fetchall(
            """SELECT id,market_id,score,payload_json,observed_at
               FROM strategy_evaluations
               WHERE strategy_source=? AND mechanism LIKE ?
               ORDER BY observed_at,id""",
            (definition.strategy_source, f"experiment_replay:{lineage_id}:%"),
        )
        observations = [_parse_replay_row(row, lineage_id) for row in rows]
        versions = sorted({item["execution_model_version"] for item in observations})
        if execution_model_version is None:
            if len(versions) != 1:
                raise ValueError(
                    "select execution_model_version when a lineage has zero or multiple models"
                )
            execution_model_version = versions[0]
        model_rows = [
            item for item in observations
            if item["execution_model_version"] == execution_model_version
        ]
        if not model_rows:
            raise KeyError(
                f"no replay observations for execution model {execution_model_version}"
            )
        data_versions = sorted({item["data_version"] for item in model_rows})
        if data_version is None:
            if len(data_versions) != 1:
                raise ValueError(
                    "select data_version when an execution model has multiple datasets"
                )
            data_version = data_versions[0]
        selected = [item for item in model_rows if item["data_version"] == data_version]
        if not selected:
            raise KeyError(f"no replay observations for data version {data_version}")
        replay = _replay_metrics(selected, definition.min_observations)
        counts = await self.db.fetchone(
            """SELECT
                   SUM(CASE WHEN is_holdout=0 THEN 1 ELSE 0 END) AS warmup,
                   SUM(CASE WHEN is_holdout=1 THEN 1 ELSE 0 END) AS holdout
               FROM decision_snapshots
               WHERE strategy_version=? AND strategy_source=?""",
            (lineage_id, definition.strategy_source),
        )
        prospective = ProspectiveCounts(
            warmup_decisions=int(counts["warmup"] or 0),
            holdout_decisions=int(counts["holdout"] or 0),
        )
        criteria = tuple(
            _evaluate_criterion(rule, definition, replay, prospective)
            for rule in definition.rejection_criteria
        )
        if any(item.status is CriterionStatus.FAIL for item in criteria):
            status = ReportStatus.REPLAY_FAIL
        elif (
            not replay.minimum_observations_met
            or any(item.status is CriterionStatus.NOT_EVALUABLE for item in criteria)
        ):
            status = ReportStatus.INSUFFICIENT
        else:
            status = ReportStatus.REPLAY_PASS
        return ExperimentReport(
            lineage_id=lineage_id,
            strategy_source=definition.strategy_source,
            registered_at=str(registration["registered_at"]),
            holdout_starts_at=str(registration["holdout_starts_at"]),
            execution_model_version=execution_model_version,
            data_versions=(data_version,),
            replay=replay,
            prospective=prospective,
            criteria=criteria,
            status=status,
        )


def _parse_replay_row(row, lineage_id: str) -> dict:
    payload = json.loads(row["payload_json"])
    if (
        payload.get("schema_version") != 1
        or payload.get("lineage_id") != lineage_id
        or payload.get("runtime") != "replay"
    ):
        raise ValueError(f"invalid replay payload in evaluation row {row['id']}")
    net_pnl = payload.get("net_pnl")
    costs = payload.get("costs")
    metadata = payload.get("metadata") or {}
    required = (
        payload.get("data_version"),
        payload.get("execution_model_version"),
        metadata.get("equity"),
        metadata.get("turnover"),
    )
    if any(value is None for value in required) or net_pnl is None or costs is None:
        raise ValueError(f"incomplete replay payload in evaluation row {row['id']}")
    numeric = (net_pnl, costs, metadata["equity"], metadata["turnover"])
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in numeric):
        raise ValueError(f"non-finite replay payload in evaluation row {row['id']}")
    if not math.isclose(float(row["score"]), float(net_pnl), abs_tol=1e-12):
        raise ValueError(f"replay score disagrees with payload in evaluation row {row['id']}")
    return {
        **payload,
        "market_id": row["market_id"],
        "observed_at": row["observed_at"],
    }


def _replay_metrics(rows: list[dict], min_observations: int) -> ReplayMetrics:
    pnls = [float(row["net_pnl"]) for row in rows]
    costs = [float(row["costs"]) for row in rows]
    equities = [float(row["metadata"]["equity"]) for row in rows]
    initial = equities[0] - pnls[0]
    curve = [initial, *equities]
    peak = curve[0]
    max_drawdown = 0.0
    for equity in curve:
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    gains = sum(value for value in pnls if value > 0)
    losses = -sum(value for value in pnls if value < 0)
    gross = sum(pnls) + sum(costs)
    targets = [target for row in rows for target in row.get("targets", [])]
    exposure = sum(
        abs(float(target["target_quantity"]) * float(target["reference_price"]))
        for target in targets
    ) / len(rows)
    independent = len({row["market_id"] for row in rows})
    return ReplayMetrics(
        observations=len(rows),
        independent_markets=independent,
        initial_equity=initial,
        final_equity=equities[-1],
        total_net_pnl=sum(pnls),
        gross_pnl_before_costs=gross,
        total_costs=sum(costs),
        turnover=sum(float(row["metadata"]["turnover"]) for row in rows),
        max_drawdown=max_drawdown,
        event_hit_rate=sum(value > 0 for value in pnls) / len(pnls) if pnls else None,
        event_profit_factor=(
            gains / losses if losses else None
        ),
        average_gross_exposure=exposure,
        minimum_observations_met=len(rows) >= min_observations,
        cost_sensitivity=tuple(
            CostScenario(
                cost_multiplier=multiplier,
                estimated_net_pnl=gross - multiplier * sum(costs),
            )
            for multiplier in (0.0, 1.0, 2.0)
        ),
    )


def _evaluate_criterion(
    criterion: str,
    definition: ExperimentDefinition,
    replay: ReplayMetrics,
    prospective: ProspectiveCounts,
) -> CriterionResult:
    if criterion in {"max_drawdown_exceeds_limit", "max_drawdown_exceeds_15_percent"}:
        passed = replay.max_drawdown <= definition.max_drawdown
        return CriterionResult(
            criterion=criterion,
            status=CriterionStatus.PASS if passed else CriterionStatus.FAIL,
            reason=(
                f"replay drawdown {replay.max_drawdown:.2%}; "
                f"preregistered limit {definition.max_drawdown:.2%}"
            ),
        )
    if criterion in {"holdout_pnl_lte_zero", "non_positive_out_of_sample_net_pnl"}:
        return CriterionResult(
            criterion=criterion,
            status=CriterionStatus.NOT_EVALUABLE,
            reason=(
                f"requires resolved prospective holdout P&L; "
                f"{prospective.holdout_decisions} holdout decisions are registered"
            ),
        )
    return CriterionResult(
        criterion=criterion,
        status=CriterionStatus.NOT_EVALUABLE,
        reason="no registered evaluator exists for this preregistered rule",
    )
