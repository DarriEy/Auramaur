"""Read-only contracts for the point-in-time evidence and forecast plane."""

from __future__ import annotations

from dataclasses import dataclass

from auramaur.db.database import Database


@dataclass(frozen=True)
class ContractViolation:
    contract: str
    count: int
    detail: str


async def audit_data_contracts(db: Database) -> list[ContractViolation]:
    checks = [
        ("probability_range",
         "SELECT COUNT(*) n FROM calibration WHERE predicted_prob NOT BETWEEN 0 AND 1 OR predicted_prob IS NULL",
         "calibration probability outside [0,1]"),
        ("orphan_forecast_evidence",
         """SELECT COUNT(*) n FROM forecast_snapshots f
            JOIN json_each(f.evidence_run_ids) e
            LEFT JOIN ingestion_runs r ON r.id=e.value WHERE r.id IS NULL""",
         "forecast names an ingestion run that does not exist"),
        ("future_evidence",
         "SELECT COUNT(*) n FROM evidence_observations WHERE datetime(published_at) > datetime(observed_at, '+1 hour')",
         "publication time is implausibly after observation"),
        ("incomplete_ingestion",
         "SELECT COUNT(*) n FROM ingestion_runs WHERE completed_at IS NULL OR status='running'",
         "persisted ingestion row is incomplete"),
        ("unlinked_strategic_forecast",
         """SELECT COUNT(*) n FROM forecast_snapshots
            WHERE strategy_source='strategic' AND evidence_run_ids='[]'""",
         "strategic forecast has no point-in-time evidence lineage"),
        ("weak_timestamp_ranked_first",
         """SELECT COUNT(*) n FROM evidence_observations
            WHERE rank_position=1 AND timestamp_quality IN ('unknown','provider_seen')""",
         "top-ranked evidence has weak publication-time semantics"),
        ("invalid_delivery_age",
         """SELECT COUNT(*) n FROM strategy_data_deliveries
            WHERE age_seconds < 0 OR latency_ms < 0 OR item_count < 0""",
         "strategy delivery contains invalid age, latency, or coverage"),
        ("failed_latest_delivery",
         """SELECT COUNT(*) n FROM strategy_data_deliveries d
            WHERE d.id=(SELECT d2.id FROM strategy_data_deliveries d2
                        WHERE d2.strategy=d.strategy AND d2.component=d.component
                        ORDER BY d2.observed_at DESC,d2.id DESC LIMIT 1)
              AND d.status NOT IN ('ok','empty')""",
         "latest strategy-facing dataset delivery is unhealthy"),
    ]
    violations = []
    for name, sql, detail in checks:
        row = await db.fetchone(sql)
        count = int(row["n"] if row else 0)
        if count:
            violations.append(ContractViolation(name, count, detail))
    return violations
