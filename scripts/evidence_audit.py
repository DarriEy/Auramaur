"""Read-only evidence-throughput and lineage audit for an Auramaur database."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


QUERIES = {
    "ingestion": """SELECT COUNT(*) runs, MIN(started_at) oldest,
        MAX(started_at) newest, SUM(status='ok') ok, SUM(status!='ok') bad
        FROM ingestion_runs""",
    "evidence": """SELECT COUNT(*) items, COUNT(DISTINCT run_id) runs,
        COUNT(DISTINCT market_id) markets, COUNT(DISTINCT source) sources,
        MIN(observed_at) oldest, MAX(observed_at) newest
        FROM evidence_observations""",
    "lineage_integrity": """SELECT
        (SELECT COUNT(*) FROM ingestion_runs
          WHERE completed_at IS NULL OR status='running') unfinished_runs,
        (SELECT COUNT(*) FROM evidence_observations e
          LEFT JOIN ingestion_runs r ON r.id=e.run_id WHERE r.id IS NULL)
          orphan_evidence,
        (SELECT COUNT(*) FROM evidence_observations
          WHERE datetime(published_at)>datetime(observed_at,'+1 hour'))
          future_dated""",
    "forecast_yield": """SELECT COUNT(*) forecasts,
        COUNT(DISTINCT market_id) markets,
        SUM(actual_outcome IS NOT NULL) resolved,
        MIN(observed_at) oldest, MAX(observed_at) newest
        FROM forecast_snapshots""",
    "evidence_daily": """SELECT substr(observed_at,1,10) day, COUNT(*) items,
        COUNT(DISTINCT run_id) runs, COUNT(DISTINCT market_id) markets
        FROM evidence_observations GROUP BY 1 ORDER BY 1 DESC LIMIT 14""",
    "source_health_24h": """SELECT source,status,COUNT(*) runs,
        SUM(item_count) items,ROUND(AVG(latency_ms),1) avg_ms,
        MAX(latency_ms) max_ms,MAX(observed_at) latest
        FROM source_fetches WHERE observed_at>=datetime('now','-24 hours')
        GROUP BY source,status ORDER BY source,status""",
    "strategy_evidence": """SELECT strategy_source,COUNT(*) events,
        COUNT(DISTINCT market_id) markets,ROUND(SUM(pnl),4) pnl
        FROM pnl_ledger GROUP BY strategy_source ORDER BY events DESC""",
}


def audit(path: Path) -> dict[str, list[dict]]:
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return {name: [dict(row) for row in conn.execute(sql)]
                for name, sql in QUERIES.items()}
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=Path, help="Path to auramaur.db")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit(args.db)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return
    for section, rows in report.items():
        print(f"\n## {section}")
        for row in rows:
            print(json.dumps(row, default=str))


if __name__ == "__main__":
    main()
