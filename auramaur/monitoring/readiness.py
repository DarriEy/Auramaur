"""Readiness checks — gate live trading until all criteria pass.

Eight criteria produce a CriterionResult with one of three statuses:
  PASS              — measurable and within threshold
  FAIL              — measurable and outside threshold
  INSUFFICIENT_DATA — not enough samples to evaluate honestly

Overall readiness passes only if every criterion is PASS.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from auramaur.db.database import Database

Status = Literal["PASS", "FAIL", "INSUFFICIENT_DATA"]


@dataclass
class CriterionResult:
    name: str
    status: Status
    value: str
    threshold: str
    detail: str = ""
    n_samples: int | None = None


@dataclass
class ReadinessReport:
    timestamp: datetime
    exchange: str | None
    window_days: int
    criteria: list[CriterionResult] = field(default_factory=list)

    @property
    def overall_pass(self) -> bool:
        return bool(self.criteria) and all(c.status == "PASS" for c in self.criteria)


# ---------------------------------------------------------------------------
# Criterion 1 — cycle health (log parsing with format-drift canary)
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = ("level", "timestamp", "event")
_ERROR_LEVELS = {"error", "critical"}


def _parse_log_for_errors(
    log_file: Path,
    since: datetime,
    sample_events_to_keep: int,
) -> tuple[int, int, int, int, list[str]]:
    total = 0
    well_formed = 0
    in_window = 0
    errors = 0
    error_events: list[str] = []

    with log_file.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            total += 1
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            if not all(k in entry for k in _REQUIRED_KEYS):
                continue
            try:
                ts = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            well_formed += 1
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < since:
                continue
            in_window += 1
            level = str(entry.get("level", "")).lower()
            if level in _ERROR_LEVELS or "exception" in entry:
                errors += 1
                if len(error_events) < sample_events_to_keep:
                    event = entry.get("event", "")
                    error_events.append(f"{level}:{event}")
    return total, well_formed, in_window, errors, error_events


async def check_cycle_health(
    log_file: Path,
    since: datetime,
    *,
    drift_threshold_pct: float = 5.0,
    sample_events_to_keep: int = 5,
) -> CriterionResult:
    if not log_file.exists():
        return CriterionResult(
            name="cycle_health",
            status="INSUFFICIENT_DATA",
            value="—",
            threshold="0 errors",
            detail=f"log file not found at {log_file.resolve()}",
        )

    import asyncio

    total, well_formed, in_window, errors, error_events = await asyncio.to_thread(
        _parse_log_for_errors, log_file, since, sample_events_to_keep
    )

    if total == 0:
        return CriterionResult(
            name="cycle_health",
            status="INSUFFICIENT_DATA",
            value="—",
            threshold="0 errors",
            detail="log file is empty",
        )

    drift_pct = ((total - well_formed) / total) * 100.0
    if drift_pct > drift_threshold_pct:
        return CriterionResult(
            name="cycle_health",
            status="FAIL",
            value=f"{drift_pct:.1f}% unparseable",
            threshold=f"≤{drift_threshold_pct:.1f}% unparseable",
            detail="log format has drifted; readiness parser may be unreliable",
        )

    if errors == 0:
        return CriterionResult(
            name="cycle_health",
            status="PASS",
            value="0 errors",
            threshold="0 errors",
            detail=f"{in_window} entries in window",
        )

    return CriterionResult(
        name="cycle_health",
        status="FAIL",
        value=f"{errors} error/critical events",
        threshold="0 errors",
        detail="; ".join(error_events),
    )


# ---------------------------------------------------------------------------
# Criterion 2 — data sources (news_items proxy for source health)
# ---------------------------------------------------------------------------


async def check_data_sources(
    db: Database,
    *,
    since_24h: datetime,
    since_window: datetime,
) -> CriterionResult:
    # v21 records every attempted fetch, including zero-result successes and
    # errors. This is a real health signal; evidence counts alone cannot tell
    # an irrelevant query from a dead provider.
    fetch_rows = await db.fetchall(
        "SELECT source,status,observed_at FROM source_fetches "
        "WHERE information_mode='production' AND observed_at>=?",
        (since_window.astimezone(timezone.utc).isoformat(),),
    )
    if fetch_rows:
        def parsed(value: str) -> datetime:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)

        window_start = since_window.astimezone(timezone.utc)
        day_start = since_24h.astimezone(timezone.utc)
        by_source: dict[str, list[tuple[datetime, str]]] = {}
        for row in fetch_rows:
            try:
                observed = parsed(row["observed_at"])
            except (TypeError, ValueError):
                continue
            if observed >= window_start:
                by_source.setdefault(row["source"], []).append((observed, row["status"]))
        if not by_source:
            return CriterionResult(
                name="data_sources", status="INSUFFICIENT_DATA",
                value="0 production sources active",
                threshold="recent successful production attempts",
                detail="no parseable production fetch attempts in the readiness window",
            )
        stale = []
        failing = []
        for source, attempts in by_source.items():
            attempts.sort(key=lambda item: item[0], reverse=True)
            if attempts[0][0] < day_start:
                stale.append(source)
                continue
            recent = [item for item in attempts if item[0] >= day_start]
            # Detect a fresh outage even when older successes exist: no success
            # in the SLA window, or the latest three calls all failed.
            if (not any(status == "ok" for _, status in recent)
                    or (len(recent) >= 3
                        and all(status == "error" for _, status in recent[:3]))):
                failing.append(source)
        if stale or failing:
            detail = []
            if stale:
                detail.append("silent: " + ", ".join(sorted(stale)))
            if failing:
                detail.append("all attempts failed: " + ", ".join(sorted(failing)))
            return CriterionResult(
                name="data_sources", status="FAIL",
                value=f"{len(stale) + len(failing)} unhealthy",
                threshold="recent successful attempts", detail="; ".join(detail),
            )
        return CriterionResult(
            name="data_sources", status="PASS", value=f"{len(by_source)} active",
            threshold="recent successful attempts",
        )

    # Compatibility fallback for databases predating durable fetch telemetry.
    rows_window = await db.fetchall(
        "SELECT source, COUNT(*) AS n FROM news_items "
        "WHERE created_at >= ? GROUP BY source",
        (since_window.isoformat(),),
    )
    rows_24h = await db.fetchall(
        "SELECT source, COUNT(*) AS n FROM news_items "
        "WHERE created_at >= ? GROUP BY source",
        (since_24h.isoformat(),),
    )
    counts_window = {r["source"]: r["n"] for r in rows_window}
    counts_24h = {r["source"]: r["n"] for r in rows_24h}

    if not counts_window:
        return CriterionResult(
            name="data_sources",
            status="INSUFFICIENT_DATA",
            value="0 sources active",
            threshold="all enabled sources active in last 24h",
            detail="no news_items in window — bot may not have run yet",
        )

    silent = sorted(s for s, n in counts_window.items() if counts_24h.get(s, 0) == 0)
    if silent:
        return CriterionResult(
            name="data_sources",
            status="FAIL",
            value=f"{len(silent)} silent in 24h",
            threshold="0 silent",
            detail=f"silent: {', '.join(silent)}",
        )

    return CriterionResult(
        name="data_sources",
        status="PASS",
        value=f"{len(counts_window)} active",
        threshold="all active",
    )


# ---------------------------------------------------------------------------
# Criterion 3 — risk gate pass rate
# ---------------------------------------------------------------------------


async def check_pass_rate(
    db: Database,
    *,
    since: datetime,
    exchange: str | None,
    min_pct: float = 0.5,
    max_pct: float = 10.0,
    min_samples: int = 30,
) -> CriterionResult:
    sig_clause = ""
    sig_params: list = [since.isoformat()]
    trade_clause = ""
    trade_params: list = [since.isoformat()]
    if exchange:
        sig_clause = " AND exchange = ?"
        sig_params.append(exchange)
        trade_clause = " AND exchange = ?"
        trade_params.append(exchange)

    sig_row = await db.fetchone(
        f"SELECT COUNT(*) AS n FROM signals WHERE timestamp >= ?{sig_clause}",
        tuple(sig_params),
    )
    trade_row = await db.fetchone(
        f"SELECT COUNT(*) AS n FROM trades "
        f"WHERE timestamp >= ? AND is_paper = 1{trade_clause}",
        tuple(trade_params),
    )
    n_signals = sig_row["n"] if sig_row else 0
    n_trades = trade_row["n"] if trade_row else 0

    if n_signals < min_samples:
        return CriterionResult(
            name="pass_rate",
            status="INSUFFICIENT_DATA",
            value=f"{n_signals} signals",
            threshold=f"≥{min_samples} signals; {min_pct}%–{max_pct}% pass",
            n_samples=n_signals,
        )

    pct = (n_trades / n_signals) * 100.0 if n_signals else 0.0
    status: Status = "PASS" if min_pct <= pct <= max_pct else "FAIL"
    return CriterionResult(
        name="pass_rate",
        status=status,
        value=f"{pct:.1f}% ({n_trades}/{n_signals})",
        threshold=f"{min_pct}%–{max_pct}%",
        n_samples=n_signals,
    )


# ---------------------------------------------------------------------------
# Criteria 4 & 5 — Brier scores (absolute + relative-to-market)
# ---------------------------------------------------------------------------


async def _resolved_predictions(db: Database, since: datetime) -> list[dict]:
    return await db.fetchall(
        """
        SELECT
            c.market_id      AS market_id,
            c.predicted_prob AS predicted_prob,
            c.actual_outcome AS actual_outcome,
            (
                SELECT s.market_prob
                FROM signals s
                WHERE s.market_id = c.market_id
                ORDER BY s.timestamp ASC
                LIMIT 1
            ) AS market_prob
        FROM calibration c
        WHERE c.actual_outcome IS NOT NULL
          AND c.predicted_prob IS NOT NULL
          AND c.resolved_at >= ?
        """,
        (since.isoformat(),),
    )


async def check_brier_absolute(
    db: Database,
    *,
    since: datetime,
    threshold: float = 0.24,
    min_samples: int = 30,
) -> CriterionResult:
    rows = await _resolved_predictions(db, since)
    if len(rows) < min_samples:
        return CriterionResult(
            name="brier_absolute",
            status="INSUFFICIENT_DATA",
            value=f"{len(rows)} resolved",
            threshold=f"≥{min_samples} resolved; Brier ≤ {threshold}",
            n_samples=len(rows),
        )
    brier = sum((r["predicted_prob"] - r["actual_outcome"]) ** 2 for r in rows) / len(rows)
    status: Status = "PASS" if brier <= threshold else "FAIL"
    return CriterionResult(
        name="brier_absolute",
        status=status,
        value=f"{brier:.3f}",
        threshold=f"≤{threshold}",
        n_samples=len(rows),
    )


async def check_brier_vs_market(
    db: Database,
    *,
    since: datetime,
    threshold: float = 0.02,
    min_samples: int = 30,
) -> CriterionResult:
    rows = await _resolved_predictions(db, since)
    paired = [r for r in rows if r["market_prob"] is not None]
    if len(paired) < min_samples:
        return CriterionResult(
            name="brier_vs_market",
            status="INSUFFICIENT_DATA",
            value=f"{len(paired)} paired",
            threshold=f"≥{min_samples} paired; bot ≥{threshold} lower than market",
            n_samples=len(paired),
        )
    bot_brier = sum(
        (r["predicted_prob"] - r["actual_outcome"]) ** 2 for r in paired
    ) / len(paired)
    market_brier = sum(
        (r["market_prob"] - r["actual_outcome"]) ** 2 for r in paired
    ) / len(paired)
    edge = market_brier - bot_brier
    status: Status = "PASS" if edge >= threshold else "FAIL"
    return CriterionResult(
        name="brier_vs_market",
        status=status,
        value=f"bot {bot_brier:.3f} vs market {market_brier:.3f} (edge {edge:+.3f})",
        threshold=f"bot ≥{threshold} lower than market",
        n_samples=len(paired),
    )


# ---------------------------------------------------------------------------
# Criterion 6 — win rate on resolved trades
# ---------------------------------------------------------------------------


async def check_win_rate(
    db: Database,
    *,
    since: datetime,
    exchange: str | None,
    threshold_pct: float = 52.0,
    min_samples: int = 30,
) -> CriterionResult:
    # Source: pnl_ledger, the authoritative realized-P&L store (one row per
    # sell/settlement). The legacy `trades.pnl` column is never populated in the
    # current path — the gateway co-writes a trades-mirror but realized P&L lives
    # only in the ledger — so reading `trades.pnl` always returned 0 rows and
    # this criterion was permanently INSUFFICIENT_DATA despite hundreds of
    # realization events. `venue` is the ledger's exchange column; `realized_at`
    # is the realization time.
    clause = ""
    params: list = [since.isoformat()]
    if exchange:
        clause = " AND venue = ?"
        params.append(exchange)
    rows = await db.fetchall(
        f"SELECT pnl FROM pnl_ledger "
        f"WHERE realized_at >= ? AND is_paper = 1{clause}",
        tuple(params),
    )
    if len(rows) < min_samples:
        return CriterionResult(
            name="win_rate",
            status="INSUFFICIENT_DATA",
            value=f"{len(rows)} resolved trades",
            threshold=f"≥{min_samples} resolved; ≥{threshold_pct:.1f}% wins",
            n_samples=len(rows),
        )
    wins = sum(1 for r in rows if (r["pnl"] or 0) > 0)
    pct = wins / len(rows) * 100.0
    status: Status = "PASS" if pct >= threshold_pct else "FAIL"
    return CriterionResult(
        name="win_rate",
        status=status,
        value=f"{pct:.1f}% ({wins}/{len(rows)})",
        threshold=f"≥{threshold_pct:.1f}%",
        n_samples=len(rows),
    )


# ---------------------------------------------------------------------------
# Criterion 7 — net PnL after fees
# ---------------------------------------------------------------------------


async def check_pnl_after_fees(
    db: Database,
    *,
    since: datetime,
    exchange: str | None,
    fee_rate: float,
    min_samples: int = 30,
) -> CriterionResult:
    # Source: pnl_ledger (see check_win_rate). IMPORTANT — the ledger's `pnl` is
    # ALREADY NET OF FEES (record_fill books `(price-avg_cost)*size - fee`), so
    # the net-after-fees figure is simply SUM(pnl); re-applying the fee_rate
    # estimate the legacy path used would DOUBLE-COUNT fees. The actual fees are
    # summed from the ledger's `fees` column for display. `fee_rate` is retained
    # in the signature for caller compatibility but no longer estimates drag.
    clause = ""
    params: list = [since.isoformat()]
    if exchange:
        clause = " AND venue = ?"
        params.append(exchange)
    rows = await db.fetchall(
        f"SELECT pnl, fees FROM pnl_ledger "
        f"WHERE realized_at >= ? AND is_paper = 1{clause}",
        tuple(params),
    )
    if len(rows) < min_samples:
        return CriterionResult(
            name="pnl_after_fees",
            status="INSUFFICIENT_DATA",
            value=f"{len(rows)} resolved trades",
            threshold=f"≥{min_samples} resolved; net PnL ≥ $0",
            n_samples=len(rows),
        )
    net = sum(r["pnl"] or 0 for r in rows)          # already net of fees
    fees = sum(r["fees"] or 0 for r in rows)
    gross = net + fees
    status: Status = "PASS" if net >= 0 else "FAIL"
    return CriterionResult(
        name="pnl_after_fees",
        status=status,
        value=f"${net:+.2f} (gross ${gross:+.2f}, fees ${fees:.2f})",
        threshold="≥ $0",
        n_samples=len(rows),
    )


# ---------------------------------------------------------------------------
# Criterion 8 — second-opinion divergence
# ---------------------------------------------------------------------------


async def check_divergence(
    db: Database,
    *,
    since: datetime,
    exchange: str | None,
    median_threshold: float = 0.15,
    p95_threshold: float = 0.30,
    min_samples: int = 30,
) -> CriterionResult:
    clause = ""
    params: list = [since.isoformat()]
    if exchange:
        clause = " AND exchange = ?"
        params.append(exchange)
    rows = await db.fetchall(
        f"SELECT divergence FROM signals "
        f"WHERE timestamp >= ? AND divergence IS NOT NULL{clause}",
        tuple(params),
    )
    values = [r["divergence"] for r in rows if r["divergence"] is not None]
    if len(values) < min_samples:
        return CriterionResult(
            name="divergence",
            status="INSUFFICIENT_DATA",
            value=f"{len(values)} signals with second opinion",
            threshold=(
                f"≥{min_samples} signals; "
                f"median ≤{median_threshold}, p95 ≤{p95_threshold}"
            ),
            n_samples=len(values),
        )
    median = statistics.median(values)
    p95 = statistics.quantiles(values, n=100, method="inclusive")[94]
    median_ok = median <= median_threshold
    p95_ok = p95 <= p95_threshold
    status: Status = "PASS" if median_ok and p95_ok else "FAIL"
    return CriterionResult(
        name="divergence",
        status=status,
        value=f"median {median:.3f}, p95 {p95:.3f}",
        threshold=f"median ≤{median_threshold}, p95 ≤{p95_threshold}",
        n_samples=len(values),
    )


# ---------------------------------------------------------------------------
# Top-level evaluator
# ---------------------------------------------------------------------------


async def evaluate_readiness(
    db: Database,
    *,
    log_file: Path | None = None,
    exchange: str | None = None,
    days: int = 7,
    fee_rate: float | None = None,
) -> ReadinessReport:
    now = datetime.now(timezone.utc)
    since_window = now - timedelta(days=days)
    since_24h = now - timedelta(hours=24)
    if log_file is None:
        # Follow the configured logging path (LOGGING__FILE / logging.file) so
        # cycle_health reads the file the bot actually writes — in a container
        # that is /app/logs/auramaur.log, not CWD/auramaur.log. Falls back to
        # the setting's default ("auramaur.log") for native runs.
        from auramaur.runtime import log_file_path

        log_file = log_file_path()
    fee_rate = 0.07 if fee_rate is None else fee_rate

    criteria = [
        await check_cycle_health(log_file, since_window),
        await check_data_sources(db, since_24h=since_24h, since_window=since_window),
        await check_pass_rate(db, since=since_window, exchange=exchange),
        await check_brier_absolute(db, since=since_window),
        await check_brier_vs_market(db, since=since_window),
        await check_win_rate(db, since=since_window, exchange=exchange),
        await check_pnl_after_fees(
            db, since=since_window, exchange=exchange, fee_rate=fee_rate
        ),
        await check_divergence(db, since=since_window, exchange=exchange),
    ]
    return ReadinessReport(
        timestamp=now,
        exchange=exchange,
        window_days=days,
        criteria=criteria,
    )
