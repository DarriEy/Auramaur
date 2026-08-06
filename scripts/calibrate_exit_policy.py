"""Chronological exit-policy audit; never mutates config or the database.

Usage: python scripts/calibrate_exit_policy.py auramaur.db
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path


MIN_TRAIN_EXITS = 30
MIN_TEST_EXITS = 15


def load(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("""
        SELECT d.observed_at, d.exchange, d.policy_action, d.net_pnl_pct,
               d.peak_pnl_pct, d.target_pct, d.market_id, d.token, d.is_paper,
               COALESCE((SELECT SUM(l.pnl) FROM pnl_ledger l
                         WHERE l.market_id=d.market_id AND l.token=d.token
                           AND l.is_paper=d.is_paper
                           AND l.realized_at >= d.observed_at), 0) AS realized_pnl
          FROM exit_decisions d
         WHERE d.policy_action <> 'HOLD'
         ORDER BY d.observed_at, d.id
    """).fetchall()


def summarize(label: str, rows: list[sqlite3.Row]) -> None:
    groups: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        groups[(row["exchange"], row["policy_action"])].append(row)
    print(f"\n{label}: {len(rows)} exits")
    for (venue, reason), values in sorted(groups.items()):
        pnl = sum(float(v["realized_pnl"]) for v in values)
        net = sum(float(v["net_pnl_pct"]) for v in values) / len(values)
        print(f"  {venue or 'unknown':14} {reason:16} n={len(values):4} "
              f"mean_net_at_decision={net:+7.2f}% realized=${pnl:+9.2f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    uri = f"file:{args.database.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = load(conn)
        except sqlite3.OperationalError as exc:
            print(f"No exit telemetry yet: {exc}")
            return 2
    cut = int(len(rows) * 0.7)
    train, test = rows[:cut], rows[cut:]
    summarize("TRAIN (oldest 70%)", train)
    summarize("HOLDOUT (newest 30%)", test)
    if len(train) < MIN_TRAIN_EXITS or len(test) < MIN_TEST_EXITS:
        print("\nNO RECOMMENDATION: insufficient chronological evidence "
              f"(need {MIN_TRAIN_EXITS} train and {MIN_TEST_EXITS} holdout exits).")
        return 2
    print("\nEvidence threshold met. Compare policy candidates in TRAIN, then accept "
          "only candidates whose direction and net result persist in HOLDOUT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
