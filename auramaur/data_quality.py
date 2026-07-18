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
        ("forecast_lineage",
         "SELECT COUNT(*) n FROM forecast_snapshots WHERE evidence_run_ids='[]' AND strategy_source='llm'",
         "LLM forecast has no evidence run"),
        ("future_evidence",
         "SELECT COUNT(*) n FROM evidence_observations WHERE datetime(published_at) > datetime(observed_at, '+1 hour')",
         "publication time is implausibly after observation"),
        ("stuck_ingestion",
         "SELECT COUNT(*) n FROM ingestion_runs WHERE status='running' AND datetime(started_at) < datetime('now','-1 hour')",
         "ingestion run never completed"),
    ]
    violations = []
    for name, sql, detail in checks:
        row = await db.fetchone(sql)
        count = int(row["n"] if row else 0)
        if count:
            violations.append(ContractViolation(name, count, detail))
    return violations
