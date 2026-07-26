"""Lineage- and cohort-aware experiment scoreboard."""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict

from auramaur.research.experiment_reporting import ExperimentReportBuilder


class ScoreboardRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    lineage_id: str
    strategy_source: str
    registered_at: str
    execution_model_version: str | None
    data_version: str | None
    replay_observations: int
    independent_markets: int
    prospective_warmup: int
    prospective_holdout: int
    net_pnl: float | None
    max_drawdown: float | None
    status: str


class ExperimentScoreboard:
    def __init__(self, db) -> None:
        self.db = db

    async def build(self) -> tuple[ScoreboardRow, ...]:
        registrations = await self.db.fetchall(
            """SELECT strategy_version,strategy_source,registered_at
               FROM strategy_experiments ORDER BY registered_at,strategy_version"""
        )
        rows: list[ScoreboardRow] = []
        reporter = ExperimentReportBuilder(self.db)
        for registration in registrations:
            lineage = str(registration["strategy_version"])
            evaluations = await self.db.fetchall(
                """SELECT payload_json FROM strategy_evaluations
                   WHERE strategy_source=? AND mechanism LIKE ? AND is_paper=1""",
                (registration["strategy_source"], f"experiment_replay:{lineage}:%"),
            )
            cohorts: set[tuple[str, str]] = set()
            for evaluation in evaluations:
                payload = json.loads(evaluation["payload_json"])
                if payload.get("lineage_id") != lineage or payload.get("runtime") != "replay":
                    continue
                model = payload.get("execution_model_version")
                data = payload.get("data_version")
                if model and data:
                    cohorts.add((str(model), str(data)))
            if not cohorts:
                counts = await self._prospective_counts(lineage, registration["strategy_source"])
                specialized = await self.db.fetchall(
                    """SELECT market_id FROM strategy_evaluations
                       WHERE strategy_source=? AND mechanism LIKE ? AND is_paper=1""",
                    (
                        registration["strategy_source"],
                        f"experiment_specialized_replay:{lineage}:%",
                    ),
                )
                rows.append(ScoreboardRow(
                    lineage_id=lineage,
                    strategy_source=str(registration["strategy_source"]),
                    registered_at=str(registration["registered_at"]),
                    execution_model_version=None,
                    data_version=None,
                    replay_observations=len(specialized),
                    independent_markets=len({row["market_id"] for row in specialized}),
                    prospective_warmup=counts[0],
                    prospective_holdout=counts[1],
                    net_pnl=None,
                    max_drawdown=None,
                    status="proposal_replay" if specialized else "no_replay",
                ))
                continue
            for model, data in sorted(cohorts):
                report = await reporter.build(
                    lineage, execution_model_version=model, data_version=data
                )
                rows.append(ScoreboardRow(
                    lineage_id=lineage,
                    strategy_source=report.strategy_source,
                    registered_at=report.registered_at,
                    execution_model_version=model,
                    data_version=data,
                    replay_observations=report.replay.observations,
                    independent_markets=report.replay.independent_markets,
                    prospective_warmup=report.prospective.warmup_decisions,
                    prospective_holdout=report.prospective.holdout_decisions,
                    net_pnl=report.replay.total_net_pnl,
                    max_drawdown=report.replay.max_drawdown,
                    status=report.status.value,
                ))
        return tuple(rows)

    async def _prospective_counts(self, lineage: str, source: str) -> tuple[int, int]:
        row = await self.db.fetchone(
            """SELECT SUM(CASE WHEN is_holdout=0 THEN 1 ELSE 0 END) warmup,
                      SUM(CASE WHEN is_holdout=1 THEN 1 ELSE 0 END) holdout
               FROM decision_snapshots
               WHERE strategy_version=? AND strategy_source=?""",
            (lineage, source),
        )
        return int(row["warmup"] or 0), int(row["holdout"] or 0)
