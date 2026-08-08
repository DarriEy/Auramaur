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
        # Both lineage checks bound themselves to rows written after delivery
        # telemetry began (the v44 deploy): earlier rows predate the contract
        # and would otherwise count as violations forever. MIN(observed_at)
        # over an empty deliveries table is NULL, which flags nothing.
        ("unlinked_strategic_forecast",
         """SELECT COUNT(*) n FROM forecast_snapshots
            WHERE strategy_source='strategic' AND evidence_run_ids='[]'
              AND datetime(observed_at) >=
                  (SELECT MIN(datetime(observed_at)) FROM strategy_data_deliveries)""",
         "strategic forecast has no point-in-time evidence lineage"),
        # 'provider_seen' is a deliberate stamp on official sources (BLS/BEA/
        # EIA) and a legitimate first rank; only 'unknown' is a contract gap.
        ("weak_timestamp_ranked_first",
         """SELECT COUNT(*) n FROM evidence_observations
            WHERE rank_position=1 AND timestamp_quality='unknown'
              AND datetime(observed_at) >=
                  (SELECT MIN(datetime(observed_at)) FROM strategy_data_deliveries)""",
         "top-ranked evidence has unknown publication-time semantics"),
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
        ("governed_trade_missing_decision",
         """SELECT COUNT(*) n FROM trades
            WHERE signal_id IS NOT NULL AND decision_id IS NULL
              AND status IN ('pending','filled','partial')""",
         "gateway-governed trade has no immutable decision link"),
        ("orphan_trade_decision",
         """SELECT COUNT(*) n FROM trades t
            LEFT JOIN decision_snapshots d ON d.id=t.decision_id
            WHERE t.decision_id IS NOT NULL AND d.id IS NULL""",
         "trade names a decision snapshot that does not exist"),
        ("orphan_fill_decision",
         """SELECT COUNT(*) n FROM fills f
            LEFT JOIN decision_snapshots d ON d.id=f.decision_id
            WHERE f.decision_id IS NOT NULL AND d.id IS NULL""",
         "fill names a decision snapshot that does not exist"),
        ("trade_decision_mismatch",
         """SELECT COUNT(*) n FROM trades t
            JOIN decision_snapshots d ON d.id=t.decision_id
            WHERE t.market_id != d.market_id
               OR t.is_paper != d.is_paper
               OR COALESCE(t.strategy_source,'') != d.strategy_source""",
         "trade and decision disagree on market, book, or strategy"),
        ("fill_decision_mismatch",
         """SELECT COUNT(*) n FROM fills f
            JOIN decision_snapshots d ON d.id=f.decision_id
            WHERE f.market_id != d.market_id OR f.is_paper != d.is_paper""",
         "fill and decision disagree on market or paper/live book"),
        ("fill_trade_lineage_mismatch",
         """SELECT COUNT(*) n FROM fills f
            WHERE EXISTS (
                SELECT 1 FROM trades t
                 WHERE t.order_id=f.order_id AND t.market_id=f.market_id
                   AND t.is_paper=f.is_paper AND t.decision_id IS NOT NULL)
              AND NOT EXISTS (
                SELECT 1 FROM trades t
                 WHERE t.order_id=f.order_id AND t.market_id=f.market_id
                   AND t.is_paper=f.is_paper
                   AND t.decision_id=f.decision_id)""",
         "fill does not carry the decision linked by its mirrored trade"),
        ("orphan_ledger_fill",
         """SELECT COUNT(*) n FROM pnl_ledger l
            LEFT JOIN fills f
              ON f.id=CAST(SUBSTR(l.source_ref,6) AS INTEGER)
            WHERE l.kind='sell' AND l.source_ref LIKE 'fill:%'
              AND f.id IS NULL""",
         "realized P&L names a fill that does not exist"),
        ("ledger_fill_mismatch",
         """SELECT COUNT(*) n FROM pnl_ledger l
            JOIN fills f ON f.id=CAST(SUBSTR(l.source_ref,6) AS INTEGER)
            WHERE l.kind='sell' AND l.source_ref LIKE 'fill:%'
              AND (l.market_id != f.market_id
                   OR l.is_paper != f.is_paper
                   OR UPPER(l.token) != UPPER(f.token))""",
         "realized P&L and its source fill disagree on position identity"),
    ]
    violations = []
    for name, sql, detail in checks:
        row = await db.fetchone(sql)
        count = int(row["n"] if row else 0)
        if count:
            violations.append(ContractViolation(name, count, detail))
    return violations
