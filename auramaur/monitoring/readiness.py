"""Readiness checks — gate live trading until all criteria pass.

Each criterion produces a CriterionResult with one of three statuses:
  PASS              — measurable and within threshold
  FAIL              — measurable and outside threshold
  INSUFFICIENT_DATA — not enough samples to evaluate honestly

Overall readiness passes only if every criterion is PASS.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from auramaur.broker.ledger import PHANTOM_STRATEGY, VENUE_STRATEGY
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
_ROTATION_SUFFIX = re.compile(r"\.(\d+)$")


def log_files_for(log_file: Path) -> list[Path]:
    """The active log plus its rotated backups, oldest first.

    ``_RotatingWriter`` renames the active file to ``<path>.1`` and shifts older
    backups up to ``<path>.<backups>`` (monitoring/logger.py), so a parser that
    opens exactly one path sees only what currently fits in ``rotate_max_mb``.
    Measured against the live deployment on 2026-08-06 that was 0.7 of the 7
    days the criterion advertises — 10% coverage, rotating every ~21-30 hours —
    and 1,487 error events sat in ``.1``/``.2``/``.3`` where nothing looked.
    Reading the backups is what makes a 7-day criterion actually score 7 days.

    A higher suffix is an OLDER file, so backups are walked in descending suffix
    order with the active file last. That keeps the fold chronological, which is
    what makes both the sampled error events (the first N in time, as before)
    and the earliest-timestamp coverage figure mean what they say.
    """
    backups: list[tuple[int, Path]] = []
    try:
        candidates = list(log_file.parent.glob(log_file.name + ".*"))
    except OSError:
        candidates = []
    for candidate in candidates:
        match = _ROTATION_SUFFIX.search(candidate.name)
        if match and candidate.is_file():
            backups.append((int(match.group(1)), candidate))
    ordered = [path for _suffix, path in sorted(backups, key=lambda item: -item[0])]
    if log_file.exists():
        ordered.append(log_file)
    return ordered


def _parse_log_for_errors(
    log_files: list[Path],
    since: datetime,
    sample_events_to_keep: int,
) -> tuple[int, int, int, int, list[str], datetime | None]:
    """Fold the error/entry counts across every log file, oldest first.

    Semantics per line are unchanged: unparseable and schema-drifted lines are
    tolerated (counted in ``total`` but not ``well_formed``, which is what feeds
    the drift canary), and only well-formed entries at or after ``since`` are
    scored. ``earliest`` is the earliest well-formed timestamp seen anywhere in
    the corpus — the parser already computed it implicitly and threw it away, and
    it is the only honest measure of how much of the window was actually read.
    """
    total = 0
    well_formed = 0
    in_window = 0
    errors = 0
    error_events: list[str] = []
    earliest: datetime | None = None

    for log_file in log_files:
        try:
            handle = log_file.open()
        except OSError:
            # Rotation is a sequence of os.replace calls, so a backup can be
            # renamed out from under the walk. Skipping it keeps the criterion
            # honest — the coverage figure below reports on what was read.
            continue
        with handle as f:
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
                    ts = datetime.fromisoformat(
                        entry["timestamp"].replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    continue
                well_formed += 1
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if earliest is None or ts < earliest:
                    earliest = ts
                if ts < since:
                    continue
                in_window += 1
                level = str(entry.get("level", "")).lower()
                if level in _ERROR_LEVELS or "exception" in entry:
                    errors += 1
                    if len(error_events) < sample_events_to_keep:
                        event = entry.get("event", "")
                        error_events.append(f"{level}:{event}")
    return total, well_formed, in_window, errors, error_events, earliest


def _coverage_note(earliest: datetime | None, since: datetime) -> str:
    """Say so when the logs do not reach back as far as the window claims.

    Empty when the corpus predates ``since`` — the criterion really did score
    the window it advertises. Otherwise it scored a shorter one, and reporting
    "0 errors" without that caveat is the same class of laundering as reporting
    an unreadable database as zero.
    """
    if earliest is None or earliest <= since:
        return ""
    now = datetime.now(timezone.utc)
    requested_days = max((now - since).total_seconds(), 0.0) / 86400.0
    covered_days = max((now - earliest).total_seconds(), 0.0) / 86400.0
    return f"; log covers {covered_days:.1f}d of {requested_days:.1f}d"


async def check_cycle_health(
    log_file: Path,
    since: datetime,
    *,
    drift_threshold_pct: float = 5.0,
    sample_events_to_keep: int = 5,
) -> CriterionResult:
    log_files = log_files_for(log_file)
    if not log_files:
        return CriterionResult(
            name="cycle_health",
            status="INSUFFICIENT_DATA",
            value="—",
            threshold="0 errors",
            detail=f"log file not found at {log_file.resolve()}",
        )

    import asyncio

    (total, well_formed, in_window, errors, error_events,
     earliest) = await asyncio.to_thread(
        _parse_log_for_errors, log_files, since, sample_events_to_keep
    )
    coverage = _coverage_note(earliest, since)

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

    if in_window == 0:
        # Zero entries is not zero errors. Nothing in the window means the bot
        # is stopped (or was, for the whole window) — every log file we could
        # find predates it. This is the residual case now that the rotated
        # backups are read; before that it also stood in for rotation
        # blindness, which log_files_for() addresses at the source.
        return CriterionResult(
            name="cycle_health",
            status="INSUFFICIENT_DATA",
            value="no log entries in window",
            threshold="0 errors",
            detail=(f"window is empty across {len(log_files)} log file(s) — "
                    "the bot may be stopped"),
        )

    if errors == 0:
        return CriterionResult(
            name="cycle_health",
            status="PASS",
            value=f"0 errors{coverage}",
            threshold="0 errors",
            detail=f"{in_window} entries in window across {len(log_files)} log file(s)",
        )

    return CriterionResult(
        name="cycle_health",
        status="FAIL",
        value=f"{errors} error/critical events{coverage}",
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


async def check_strategy_data_delivery(
    db: Database, *, since: datetime,
) -> CriterionResult:
    """Verify the datasets consumed by recently-running strategies.

    Active strategies fail closed on unknown contracts, missing required
    telemetry, empty/partial required datasets, missing required fields, and
    absent or stale source timestamps. This criterion is an execution-safety
    assertion, not a provider-uptime dashboard.
    """
    from auramaur.data_edge import canonical_strategy, requirements_for

    heartbeats = await db.fetchall(
        "SELECT strategy,interval_seconds FROM strategy_heartbeats "
        "WHERE datetime(last_beat_at) >= datetime(?)", (since.isoformat(),))
    if not heartbeats:
        return CriterionResult(
            name="strategy_data_delivery", status="INSUFFICIENT_DATA",
            value="0 active strategies", threshold="fresh required datasets",
            detail="no recent strategy heartbeats")

    now = datetime.now(timezone.utc)
    failures: list[str] = []
    unobserved: list[str] = []
    checked = 0
    for heartbeat in heartbeats:
        strategy = heartbeat["strategy"]
        canonical = canonical_strategy(strategy)
        requirements = requirements_for(strategy)
        if not requirements:
            failures.append(f"{strategy}=unknown data contract")
            continue
        for requirement in requirements:
            row = await db.fetchone(
                """SELECT status,observed_at,age_seconds,item_count,missing_fields
                   FROM strategy_data_deliveries
                   WHERE strategy=? AND component=?
                   ORDER BY observed_at DESC LIMIT 1""",
                (canonical, requirement.component))
            label = f"{strategy}:{requirement.component}"
            if row is None:
                (failures if requirement.fail_closed else unobserved).append(
                    f"{label}=unobserved")
                continue
            checked += 1
            try:
                observed = datetime.fromisoformat(
                    row["observed_at"].replace("Z", "+00:00"))
                if observed.tzinfo is None:
                    observed = observed.replace(tzinfo=timezone.utc)
                delivery_age = max(0.0, (now - observed).total_seconds())
            except (TypeError, ValueError):
                failures.append(f"{label}=bad timestamp")
                continue
            allowed = max(requirement.max_age_seconds,
                          float(heartbeat["interval_seconds"] or 0) * 2)
            if delivery_age > allowed:
                if row["status"] in ("stale", "timeout", "error", "unavailable"):
                    failures.append(f"{label}={row['status']} age={delivery_age:.0f}s")
                else:
                    target = failures if requirement.fail_closed else unobserved
                    target.append(f"{label}=expired age={delivery_age:.0f}s")
                continue
            source_age = row["age_seconds"]
            try:
                missing_fields = json.loads(row["missing_fields"] or "[]")
            except (TypeError, ValueError, json.JSONDecodeError):
                missing_fields = ["invalid missing_fields metadata"]
            reasons = []
            if row["status"] != "ok":
                reasons.append(str(row["status"]))
            if requirement.source_time_required and source_age is None:
                reasons.append("missing source time")
            elif source_age is not None and source_age > requirement.max_age_seconds:
                reasons.append(f"source age={source_age:.0f}s")
            if int(row["item_count"] or 0) < requirement.min_items:
                reasons.append(f"items={int(row['item_count'] or 0)}")
            if missing_fields:
                reasons.append("missing=" + ",".join(map(str, missing_fields)))
            if reasons:
                target = failures if requirement.fail_closed else unobserved
                target.append(f"{label}=" + " ".join(reasons))

    if failures:
        return CriterionResult(
            name="strategy_data_delivery", status="FAIL",
            value=f"{len(failures)} unhealthy datasets",
            threshold="all observed requirements fresh and complete",
            detail="; ".join(failures[:12]), n_samples=checked)
    if unobserved:
        return CriterionResult(
            name="strategy_data_delivery", status="INSUFFICIENT_DATA",
            value=f"{len(unobserved)} datasets not yet observed",
            threshold="telemetry for every required dataset",
            detail=", ".join(unobserved[:12]), n_samples=checked)
    return CriterionResult(
        name="strategy_data_delivery", status="PASS",
        value=f"{checked} datasets fresh", threshold="all requirements fresh",
        n_samples=checked)


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
# Criterion 9 — exit liveness (entries continue while exits are absent)
# ---------------------------------------------------------------------------

# Calibrated by replaying the criterion over the full trade history; see
# docs/exit-liveness-criterion.md for the measurement and the evidence.
_EXIT_LIVENESS_WINDOW_DAYS = 7
_EXIT_LIVENESS_MIN_ENTRIES = 3

# A realization event. `commission` rows are cash adjustments booked against a
# still-open position — money moving is not a position leaving.
_EXIT_KINDS = ("sell", "settlement")

# Attribution buckets, not strategy books. The ledger's entry-strategy
# resolution refuses to credit any of these as the entrant
# (broker/ledger.py::_market_context), so a realization can never be booked
# back to them and their exit count is zero *by construction*. Judging them
# would be a permanent structural false alarm rather than a signal. `exit` is
# the exit path's own attribution on the trades mirror, which is exactly why
# exits are counted from the ledger (entry-attributed) and not from trades.
_UNJUDGEABLE_BOOKS = frozenset({
    "", "exit", "order_monitor", "legacy_unattributed", "adopted_unknown",
    PHANTOM_STRATEGY, VENUE_STRATEGY,
})


def _exit_cell_label(venue: str, book: str, is_paper: int) -> str:
    return f"{venue or '?'}/{book} [{'paper' if is_paper else 'live'}]"


async def _exit_events_by_cell(
    db: Database, *, since: datetime, exchange: str | None, before: bool,
) -> dict[tuple[str, str, int], int]:
    """Realization counts per (venue, book, mode), either side of ``since``.

    ``datetime(col)`` rather than a raw string compare: realized_at is written
    both as ``YYYY-MM-DD HH:MM:SS`` (settlements) and as an offset-aware ISO
    string with a ``T`` (sell fills), and lexical ordering disagrees with
    chronological ordering across those two forms inside a single day.
    """
    comparison = "<" if before else ">="
    kinds = ", ".join("?" for _ in _EXIT_KINDS)
    params: list = [*_EXIT_KINDS, since.astimezone(timezone.utc).isoformat()]
    clause = ""
    if exchange:
        clause = " AND venue = ?"
        params.append(exchange)
    rows = await db.fetchall(
        f"""SELECT venue, strategy_source AS book, is_paper, COUNT(*) AS n
              FROM pnl_ledger
             WHERE kind IN ({kinds})
               AND datetime(realized_at) {comparison} datetime(?){clause}
             GROUP BY 1, 2, 3""",
        tuple(params),
    )
    return {
        ((r["venue"] or ""), (r["book"] or "").strip(), int(r["is_paper"] or 0)):
            int(r["n"] or 0)
        for r in rows
    }


async def check_exit_liveness(
    db: Database,
    *,
    since: datetime,
    exchange: str | None,
    min_entries: int = _EXIT_LIVENESS_MIN_ENTRIES,
) -> CriterionResult:
    """FAIL when a book keeps taking entries while its exits have stopped.

    Watches OUTCOMES, not mechanisms. The live prediction-market exit loop
    once raised on every tick, and the raise landed in a ``debug``-level
    handler — so cycle_health, which scores cycles by log level, reported
    clean while entries continued and exits were absent for an extended
    period. Any cause with that shape (an exception, a config mistake, a stuck
    lock, an adapter silently refusing sells) produces the same observable,
    and this criterion reads only the observable.

    Cells are ``(venue, book, mode)``. Paper and live are judged separately on
    purpose: paper exits kept working throughout that incident, which is
    precisely why nothing looked wrong in aggregate.

    Entries are BUY rows on ``trades`` (the gateway's mirror carries the
    exchange and the deciding strategy). Exits are ``pnl_ledger`` rows of kind
    ``sell`` or ``settlement``, because the ledger attributes a realization to
    the strategy that OPENED the position while the trades mirror attributes
    the SELL to the exit path itself. Settlements count: a long-dated book can
    legitimately hold for weeks with no sell while positions resolve, and
    excluding them fires constantly on ``long_horizon``.

    Three ways out of a FAIL, each deliberate:
      * fewer than ``min_entries`` entries — one entry proves nothing;
      * no exit event ever recorded for the cell — "stopped" presupposes it
        was running, and a brand-new book's first realization is genuinely
        weeks away. Reported as not-yet-judgeable rather than silently
        dropped;
      * a dormant book (no entries at all) is never a FAIL.
    """
    now = datetime.now(timezone.utc)
    window_days = max((now - since).total_seconds(), 0.0) / 86400.0
    threshold = (
        f"every book with ≥{min_entries} entries in {window_days:.0f}d "
        f"shows ≥1 exit"
    )

    params: list = [since.astimezone(timezone.utc).isoformat()]
    clause = ""
    if exchange:
        clause = " AND exchange = ?"
        params.append(exchange)
    entry_rows = await db.fetchall(
        f"""SELECT exchange AS venue, strategy_source AS book, is_paper,
                   COUNT(*) AS n
              FROM trades
             WHERE side = 'BUY'
               AND COALESCE(status, '') NOT IN ('cancelled', 'rejected')
               AND datetime(timestamp) >= datetime(?){clause}
             GROUP BY 1, 2, 3""",
        tuple(params),
    )

    in_window = await _exit_events_by_cell(
        db, since=since, exchange=exchange, before=False)
    earlier = await _exit_events_by_cell(
        db, since=since, exchange=exchange, before=True)

    stalled: list[str] = []
    unproven: list[str] = []
    active: list[str] = []
    for row in entry_rows:
        book = (row["book"] or "").strip()
        if book in _UNJUDGEABLE_BOOKS:
            continue
        entries = int(row["n"] or 0)
        if entries < min_entries:
            continue
        venue = row["venue"] or ""
        is_paper = int(row["is_paper"] or 0)
        cell = (venue, book, is_paper)
        label = _exit_cell_label(venue, book, is_paper)
        exits = in_window.get(cell, 0)
        if exits > 0:
            active.append(f"{label}: {entries} entries, {exits} exits")
        elif earlier.get(cell, 0) == 0:
            unproven.append(f"{label}: {entries} entries, no exit ever recorded")
        else:
            stalled.append(f"{label}: {entries} entries, 0 exits")

    judged = len(active) + len(stalled)
    if stalled:
        return CriterionResult(
            name="exit_liveness",
            status="FAIL",
            value=f"{len(stalled)} book(s) entering with no exits",
            threshold=threshold,
            detail="; ".join(sorted(stalled)),
            n_samples=judged,
        )
    if judged == 0:
        detail = (
            "; ".join(sorted(unproven)) if unproven
            else f"no book took ≥{min_entries} entries in the window"
        )
        return CriterionResult(
            name="exit_liveness",
            status="INSUFFICIENT_DATA",
            value="0 books judgeable",
            threshold=threshold,
            detail=detail,
            n_samples=0,
        )
    detail = "; ".join(sorted(active))
    if unproven:
        detail += " | not yet judgeable: " + "; ".join(sorted(unproven))
    return CriterionResult(
        name="exit_liveness",
        status="PASS",
        value=f"{judged} book(s) exiting",
        threshold=threshold,
        detail=detail,
        n_samples=judged,
    )


# ---------------------------------------------------------------------------
# Top-level evaluator
# ---------------------------------------------------------------------------


def _exit_liveness_settings() -> tuple[int, int]:
    """Operator overrides for the exit-liveness window and entry floor.

    They live under `monitoring:` — the operator-declared health-check
    contract — and NOT in any strategy's section. strategy_version hashes
    exactly {strategy_source, that strategy's own config section, and
    risk.min_edge_pct / max_spread_pct / confidence_floor}
    (broker/execution_gateway.py::_capture_decision), and `monitoring` is
    never resolved as a strategy section, so retuning these cannot reset a
    14-day graduation clock.

    Never raises: a config fault must degrade to the calibrated defaults
    rather than take down the whole readiness report.
    """
    try:
        from config.settings import Settings

        monitoring = Settings().monitoring
        return (
            max(1, int(monitoring.exit_liveness_window_days)),
            max(1, int(monitoring.exit_liveness_min_entries)),
        )
    except Exception:  # noqa: BLE001 - config must never break the report
        return _EXIT_LIVENESS_WINDOW_DAYS, _EXIT_LIVENESS_MIN_ENTRIES


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
    # exit_liveness keeps its own calibrated window rather than following
    # `days`: the entry floor is only meaningful against the window it was
    # measured for, and widening the report must not quietly change what the
    # criterion means. Its threshold string states the window it used.
    exit_window_days, exit_min_entries = _exit_liveness_settings()
    since_exits = now - timedelta(days=exit_window_days)
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
        await check_strategy_data_delivery(db, since=since_24h),
        await check_pass_rate(db, since=since_window, exchange=exchange),
        await check_brier_absolute(db, since=since_window),
        await check_brier_vs_market(db, since=since_window),
        await check_win_rate(db, since=since_window, exchange=exchange),
        await check_pnl_after_fees(
            db, since=since_window, exchange=exchange, fee_rate=fee_rate
        ),
        await check_divergence(db, since=since_window, exchange=exchange),
        await check_exit_liveness(
            db, since=since_exits, exchange=exchange,
            min_entries=exit_min_entries,
        ),
    ]
    return ReadinessReport(
        timestamp=now,
        exchange=exchange,
        window_days=days,
        criteria=criteria,
    )
