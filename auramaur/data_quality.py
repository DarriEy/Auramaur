"""Data contracts, retention, and lossless-enough research rollups."""

from __future__ import annotations

from dataclasses import dataclass

from auramaur.db.database import Database


@dataclass(frozen=True)
class ContractViolation:
    contract: str
    count: int
    detail: str


async def audit_data_contracts(db: Database) -> list[ContractViolation]:
    """Return measurable contract violations without mutating data."""
    checks = [
        ("probability_range", "SELECT COUNT(*) n FROM calibration WHERE predicted_prob NOT BETWEEN 0 AND 1 OR predicted_prob IS NULL", "calibration probability outside [0,1]"),
        ("price_range", "SELECT COUNT(*) n FROM price_history WHERE price NOT BETWEEN 0 AND 1 OR price IS NULL", "binary-market price outside [0,1]"),
        ("forecast_lineage", "SELECT COUNT(*) n FROM forecast_snapshots WHERE evidence_run_ids='[]' AND strategy_source='llm'", "LLM forecast has no evidence run"),
        ("future_evidence", "SELECT COUNT(*) n FROM evidence_observations WHERE published_at > datetime(observed_at, '+1 hour')", "publication time is implausibly after observation"),
        ("stuck_ingestion", "SELECT COUNT(*) n FROM ingestion_runs WHERE status='running' AND started_at < datetime('now','-1 hour')", "ingestion run never completed"),
    ]
    violations = []
    for name, sql, detail in checks:
        row = await db.fetchone(sql)
        count = int(row["n"] if row else 0)
        if count:
            violations.append(ContractViolation(name, count, detail))
    return violations


async def rollup_and_prune_prices(
    db: Database, *, raw_retention_days: int = 30, rollup_after_days: int = 7,
) -> tuple[int, int]:
    """Create hourly OHLC rollups, then prune raw rows beyond retention.

    This is deliberately explicit rather than automatic during connection or a
    trading cycle; operators can schedule it outside latency-sensitive hours.
    """
    if rollup_after_days > raw_retention_days:
        raise ValueError("rollup_after_days must be <= raw_retention_days")
    before = db.db.total_changes
    try:
        await db.execute("SAVEPOINT price_retention")
        await db.execute(
            """INSERT OR IGNORE INTO price_history_hourly
           (exchange,market_id,hour,open,high,low,close,samples)
           WITH ranked AS (
             SELECT exchange,market_id,price,recorded_at,
                    strftime('%Y-%m-%d %H:00:00',recorded_at) hour,
                    ROW_NUMBER() OVER (PARTITION BY exchange,market_id,
                      strftime('%Y-%m-%d %H',recorded_at) ORDER BY recorded_at,id) first_rn,
                    ROW_NUMBER() OVER (PARTITION BY exchange,market_id,
                      strftime('%Y-%m-%d %H',recorded_at) ORDER BY recorded_at DESC,id DESC) last_rn
             FROM price_history p WHERE recorded_at < datetime('now', ?)
               AND NOT EXISTS (
                 SELECT 1 FROM price_history_hourly h
                 WHERE h.exchange=p.exchange AND h.market_id=p.market_id
                   AND h.hour=strftime('%Y-%m-%d %H:00:00',p.recorded_at)
               )
           )
           SELECT exchange,market_id,hour,
                  MAX(CASE WHEN first_rn=1 THEN price END),MAX(price),MIN(price),
                  MAX(CASE WHEN last_rn=1 THEN price END),COUNT(*)
           FROM ranked GROUP BY exchange,market_id,hour""",
            (f"-{rollup_after_days} days",),
        )
        rolled = db.db.total_changes - before
        cursor = await db.execute(
            """DELETE FROM price_history
               WHERE recorded_at < datetime('now', ?)
                 AND EXISTS (
                   SELECT 1 FROM price_history_hourly h
                   WHERE h.exchange=price_history.exchange
                     AND h.market_id=price_history.market_id
                     AND h.hour=strftime('%Y-%m-%d %H:00:00',price_history.recorded_at)
                 )""",
            (f"-{raw_retention_days} days",),
        )
        await db.execute("RELEASE SAVEPOINT price_retention")
        await db.commit()
    except Exception:
        try:
            await db.execute("ROLLBACK TO SAVEPOINT price_retention")
            await db.execute("RELEASE SAVEPOINT price_retention")
        finally:
            await db.db.rollback()
        raise
    return rolled, cursor.rowcount
