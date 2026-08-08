"""Paired chronological exit-policy calibration; never mutates anything.

Usage: python scripts/calibrate_exit_policy.py auramaur.db
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from auramaur.risk.exit_calibration import (
    ExitCandidate, calibrate, completed_episodes, observation,
)
from config.settings import Settings


MIN_TRAIN_EXITS = 30
MIN_TEST_EXITS = 15


def load(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("""
        SELECT observed_at, exchange, policy_action, gross_pnl_pct, net_pnl_pct,
               peak_pnl_pct, target_pct, market_id, token, is_paper,
               entry_price, size
          FROM exit_decisions d
         ORDER BY d.observed_at, d.id
    """).fetchall()


def candidates(settings: Settings) -> tuple[ExitCandidate, list[ExitCandidate]]:
    cfg = settings.execution
    baseline = ExitCandidate("deployed", 1.0, cfg.trailing_stop_activation_pct,
                             cfg.trailing_stop_giveback_fraction)
    return baseline, [
        baseline,
        ExitCandidate("bank_earlier", 0.75, cfg.trailing_stop_activation_pct,
                      min(cfg.trailing_stop_giveback_fraction, 0.35)),
        ExitCandidate("bank_fast", 0.50, cfg.trailing_stop_activation_pct,
                      min(cfg.trailing_stop_giveback_fraction, 0.25)),
    ]


def print_scores(label, scores) -> None:
    print(f"\n{label}")
    for score in scores:
        lcb = ("-inf" if score.delta_lcb95_usd == float("-inf")
               else f"${score.delta_lcb95_usd:+.3f}")
        print(f"  {score.candidate.name:14} n={score.n:4} "
              f"mean=${score.mean_usd:+8.3f} total=${score.total_usd:+9.2f} "
              f"paired_delta=${score.mean_delta_usd:+7.3f} LCB95={lcb}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    uri = f"file:{args.database.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            rows = load(conn)
        except sqlite3.OperationalError as exc:
            print(f"No exit telemetry yet: {exc}")
            return 2
    episodes = completed_episodes(observation(row) for row in rows)
    baseline, options = candidates(Settings())
    report = calibrate(episodes, options, baseline,
                       min_train=MIN_TRAIN_EXITS, min_holdout=MIN_TEST_EXITS)
    print(f"Completed target/trailing episodes: {report.episodes} "
          f"(train={report.train_n}, holdout={report.holdout_n}); "
          "open/right-censored episodes are excluded.")
    print_scores("TRAIN (oldest 70%; candidate selected here)",
                 report.train_scores)
    if report.holdout_score is not None:
        print_scores("HOLDOUT (newest 30%; winner scored once)",
                     (report.holdout_score,))
    if report.recommendation is None:
        print(f"\nNO RECOMMENDATION: {report.reason}.")
        return 2
    print(f"\nREVIEW CANDIDATE: {report.recommendation}")
    print("Evidence only: configuration was NOT changed. Convert target_scale "
          "into the lifecycle target settings and review manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
