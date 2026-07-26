"""Persistence adapter for registered experiments and replay observations.

Registration uses the existing prospective anchor. Replay is deliberately
stored only as a paper research evaluation: it never writes decision snapshots,
fills, outcomes, P&L ledger rows, or any other input consumed by graduation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from auramaur.experiments.registry import RegisteredExperiment
from auramaur.experiments.runtimes import ReplayReport


@dataclass(frozen=True)
class ProspectiveRegistration:
    lineage_id: str
    strategy_source: str
    registered_at: str
    holdout_starts_at: str


class ExperimentRepository:
    """Write experiment artifacts without promoting replay into capital evidence."""

    def __init__(self, db) -> None:
        self.db = db

    async def register(
        self, registered: RegisteredExperiment
    ) -> ProspectiveRegistration:
        definition = registered.definition
        async with self.db.transaction(owner="experiment_register"):
            await self.db.execute(
                """INSERT OR IGNORE INTO strategy_experiments
                       (strategy_version, strategy_source, config_json,
                        holdout_starts_at)
                   VALUES (?, ?, ?, datetime('now', ?))""",
                (
                    registered.lineage_id,
                    definition.strategy_source,
                    definition.registration_json,
                    f"+{definition.holdout_days} days",
                ),
            )
            row = await self.db.fetchone(
                """SELECT strategy_source, config_json, registered_at,
                          holdout_starts_at
                   FROM strategy_experiments WHERE strategy_version=?""",
                (registered.lineage_id,),
            )
            if row is None:
                raise RuntimeError("experiment registration was not persisted")
            if (
                row["strategy_source"] != definition.strategy_source
                or row["config_json"] != definition.registration_json
            ):
                raise ValueError("lineage conflicts with an existing registration")
            tied = await self.db.fetchone(
                """SELECT COUNT(*) AS n FROM strategy_experiments
                   WHERE strategy_source=? AND registered_at=?""",
                (definition.strategy_source, row["registered_at"]),
            )
            if int(tied["n"]) > 1:
                raise ValueError(
                    "registration timestamp collides with another lineage; retry later"
                )
        return ProspectiveRegistration(
            registered.lineage_id,
            definition.strategy_source,
            str(row["registered_at"]),
            str(row["holdout_starts_at"]),
        )

    async def record_replay(
        self,
        registered: RegisteredExperiment,
        report: ReplayReport,
    ) -> int:
        """Persist replay artifacts as non-graduating paper research rows."""
        if report.lineage_id != registered.lineage_id:
            raise ValueError("replay report lineage does not match registration")
        await self.register(registered)
        inserted = 0
        async with self.db.transaction(owner="experiment_replay"):
            for result in report.results:
                if result.lineage_id != registered.lineage_id:
                    raise ValueError("replay result lineage does not match registration")
                if result.runtime != "replay":
                    raise ValueError("only replay results can be recorded here")
                if result.execution_model_version != report.execution_model_version:
                    raise ValueError("replay execution-model versions do not match")
                observed_at = _sqlite_timestamp(result.observed_at)
                mechanism = (
                    f"experiment_replay:{registered.lineage_id}:"
                    f"{report.execution_model_version}:seq={result.event_sequence}"
                )
                payload = json.dumps({
                    "schema_version": 1,
                    "lineage_id": registered.lineage_id,
                    "runtime": "replay",
                    "data_version": result.data_version,
                    "execution_model_version": result.execution_model_version,
                    "event_sequence": result.event_sequence,
                    "net_pnl": result.net_pnl,
                    "costs": result.costs,
                    "targets": [target.model_dump(mode="json") for target in result.targets],
                    "metadata": result.metadata.decoded(),
                }, sort_keys=True, separators=(",", ":"), allow_nan=False)
                cursor = await self.db.execute(
                    """INSERT OR IGNORE INTO strategy_evaluations
                           (strategy_source, market_id, mechanism, score,
                            expected_edge, payload_json, is_paper, observed_at)
                       VALUES (?, ?, ?, ?, 0, ?, 1, ?)""",
                    (
                        registered.definition.strategy_source,
                        result.market_id,
                        mechanism,
                        float(result.net_pnl or 0.0),
                        payload,
                        observed_at,
                    ),
                )
                if cursor.rowcount == 1:
                    inserted += 1
                    continue
                existing = await self.db.fetchone(
                    """SELECT payload_json, score, is_paper
                       FROM strategy_evaluations
                       WHERE strategy_source=? AND market_id=?
                         AND mechanism=? AND observed_at=?""",
                    (
                        registered.definition.strategy_source,
                        result.market_id,
                        mechanism,
                        observed_at,
                    ),
                )
                if (
                    existing is None
                    or existing["payload_json"] != payload
                    or float(existing["score"]) != float(result.net_pnl or 0.0)
                    or int(existing["is_paper"]) != 1
                ):
                    raise ValueError("replay observation conflicts with an existing row")
        return inserted


def _sqlite_timestamp(value: datetime) -> str:
    """UTC ISO timestamp whose lexical order matches chronological order."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("replay timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")
