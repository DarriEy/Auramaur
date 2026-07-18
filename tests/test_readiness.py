"""Tests for the readiness check module."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from auramaur.db.database import Database
from auramaur.monitoring.readiness import (
    check_brier_absolute,
    check_brier_vs_market,
    check_cycle_health,
    check_data_sources,
    check_divergence,
    check_pass_rate,
    check_pnl_after_fees,
    check_win_rate,
    evaluate_readiness,
)


# ---------------------------------------------------------------------------
# DB fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path):
    db_path = tmp_path / "test.db"
    instance = Database(str(db_path))
    await instance.connect()
    yield instance
    await instance.close()


async def _seed_signals(
    db: Database,
    *,
    n: int,
    exchange: str = "kalshi",
    market_prefix: str = "mkt-",
    divergence: float | None = None,
    timestamp_offset_days: float = 0.0,
) -> None:
    ts = (
        datetime.now(timezone.utc) - timedelta(days=timestamp_offset_days)
    ).isoformat()
    for i in range(n):
        await db.execute(
            "INSERT INTO signals (market_id, exchange, timestamp, claude_prob, "
            "claude_confidence, market_prob, edge, divergence, action) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"{market_prefix}{i}",
                exchange,
                ts,
                0.55,
                "MEDIUM",
                0.50,
                5.0,
                divergence,
                "BUY",
            ),
        )
    await db.commit()


async def _seed_trades(
    db: Database,
    *,
    pnls: list[float],
    exchange: str = "kalshi",
    is_paper: int = 1,
    timestamp_offset_days: float = 0.0,
) -> None:
    ts = (
        datetime.now(timezone.utc) - timedelta(days=timestamp_offset_days)
    ).isoformat()
    for i, pnl in enumerate(pnls):
        await db.execute(
            "INSERT INTO trades (market_id, exchange, timestamp, side, size, "
            "price, is_paper, status, pnl) "
            "VALUES (?, ?, ?, 'BUY', 10.0, 0.50, ?, 'filled', ?)",
            (f"trade-mkt-{i}", exchange, ts, is_paper, pnl),
        )
    await db.commit()


async def _seed_ledger(
    db: Database,
    *,
    pnls: list[float],
    fees: float = 0.0,
    venue: str = "kalshi",
    is_paper: int = 1,
    timestamp_offset_days: float = 0.0,
) -> None:
    """Seed realization events into pnl_ledger — the authoritative source the
    win-rate / pnl-after-fees readiness criteria now read (the legacy
    trades.pnl column is never populated)."""
    ts = (
        datetime.now(timezone.utc) - timedelta(days=timestamp_offset_days)
    ).isoformat()
    for i, pnl in enumerate(pnls):
        await db.execute(
            "INSERT INTO pnl_ledger (market_id, venue, kind, token, qty, pnl, "
            "fees, is_paper, source_ref, realized_at) "
            "VALUES (?, ?, 'sell', 'YES', 10.0, ?, ?, ?, ?, ?)",
            (f"led-mkt-{i}", venue, pnl, fees, is_paper,
             f"ledtest:{venue}:{i}:{ts}", ts),
        )
    await db.commit()


async def _seed_calibration(
    db: Database,
    *,
    pairs: list[tuple[float, int]],
    market_probs: list[float] | None = None,
    timestamp_offset_days: float = 0.0,
) -> None:
    ts = (
        datetime.now(timezone.utc) - timedelta(days=timestamp_offset_days)
    ).isoformat()
    for i, (predicted, outcome) in enumerate(pairs):
        market_id = f"calib-mkt-{i}"
        await db.execute(
            "INSERT INTO calibration (market_id, predicted_prob, actual_outcome, "
            "resolved_at) VALUES (?, ?, ?, ?)",
            (market_id, predicted, outcome, ts),
        )
        if market_probs is not None:
            mp = market_probs[i]
            await db.execute(
                "INSERT INTO signals (market_id, exchange, timestamp, claude_prob, "
                "claude_confidence, market_prob, edge, action) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (market_id, "kalshi", ts, predicted, "MEDIUM", mp, 5.0, "BUY"),
            )
    await db.commit()


# ---------------------------------------------------------------------------
# Cycle health
# ---------------------------------------------------------------------------


def _structlog_line(level: str, event: str, ts: datetime, **extra) -> str:
    payload = {
        "level": level,
        "timestamp": ts.isoformat().replace("+00:00", "Z"),
        "event": event,
    }
    payload.update(extra)
    return json.dumps(payload) + "\n"


@pytest.mark.asyncio
async def test_cycle_health_pass_when_no_errors(tmp_path):
    log_file = tmp_path / "auramaur.log"
    now = datetime.now(timezone.utc)
    log_file.write_text(
        _structlog_line("info", "engine.cycle_complete", now)
        + _structlog_line("warning", "engine.skipped_junk", now)
    )
    result = await check_cycle_health(log_file, now - timedelta(days=7))
    assert result.status == "PASS"
    assert "0 errors" in result.value


@pytest.mark.asyncio
async def test_cycle_health_fail_on_error_level(tmp_path):
    log_file = tmp_path / "auramaur.log"
    now = datetime.now(timezone.utc)
    log_file.write_text(
        _structlog_line("info", "engine.cycle_complete", now)
        + _structlog_line("error", "exchange.order_failed", now)
    )
    result = await check_cycle_health(log_file, now - timedelta(days=7))
    assert result.status == "FAIL"
    assert "1 error" in result.value


@pytest.mark.asyncio
async def test_cycle_health_drift_canary_fails(tmp_path):
    log_file = tmp_path / "auramaur.log"
    now = datetime.now(timezone.utc)
    body = "this is not json\n" * 9
    body += _structlog_line("info", "ok", now)
    log_file.write_text(body)
    result = await check_cycle_health(log_file, now - timedelta(days=7))
    assert result.status == "FAIL"
    assert "format has drifted" in result.detail


@pytest.mark.asyncio
async def test_cycle_health_missing_log_is_insufficient_data(tmp_path):
    result = await check_cycle_health(
        tmp_path / "nonexistent.log", datetime.now(timezone.utc) - timedelta(days=7)
    )
    assert result.status == "INSUFFICIENT_DATA"


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_data_sources_pass_when_all_active(db: Database):
    now = datetime.now(timezone.utc)
    recent = now - timedelta(hours=1)
    for source in ("NewsAPI", "Reddit", "RSS"):
        for i in range(3):
            await db.execute(
                "INSERT INTO news_items (id, source, title, content, created_at) "
                "VALUES (?, ?, 'title', 'body', ?)",
                (f"{source}-{i}", source, recent.isoformat()),
            )
    await db.commit()
    result = await check_data_sources(
        db,
        since_24h=now - timedelta(hours=24),
        since_window=now - timedelta(days=7),
    )
    assert result.status == "PASS"


@pytest.mark.asyncio
async def test_data_sources_fail_when_one_silent(db: Database):
    now = datetime.now(timezone.utc)
    recent = now - timedelta(hours=1)
    old = now - timedelta(days=3)
    await db.execute(
        "INSERT INTO news_items (id, source, title, content, created_at) "
        "VALUES ('NewsAPI-1', 'NewsAPI', 't', 'b', ?)",
        (recent.isoformat(),),
    )
    await db.execute(
        "INSERT INTO news_items (id, source, title, content, created_at) "
        "VALUES ('Reddit-1', 'Reddit', 't', 'b', ?)",
        (old.isoformat(),),
    )
    await db.commit()
    result = await check_data_sources(
        db,
        since_24h=now - timedelta(hours=24),
        since_window=now - timedelta(days=7),
    )
    assert result.status == "FAIL"
    assert "Reddit" in result.detail


@pytest.mark.asyncio
async def test_data_sources_fail_on_three_fresh_errors_after_older_success(db: Database):
    now = datetime.now(timezone.utc)
    await db.execute(
        "INSERT INTO ingestion_runs (id,query,started_at) VALUES ('run','q',?)",
        ((now - timedelta(days=2)).isoformat(),),
    )
    rows = [
        ("old-ok", "source", "ok", (now - timedelta(days=2)).isoformat()),
        ("e1", "source", "error", (now - timedelta(minutes=30)).isoformat()),
        ("e2", "source", "error", (now - timedelta(minutes=20)).isoformat()),
        ("e3", "source", "error", (now - timedelta(minutes=10)).isoformat()),
    ]
    for run_id, source, status, observed in rows:
        if run_id != "old-ok":
            await db.execute(
                "INSERT INTO ingestion_runs (id,query,started_at) VALUES (?, 'q', ?)",
                (run_id, observed),
            )
        await db.execute(
            "INSERT INTO source_fetches (run_id,source,status,observed_at) VALUES (?,?,?,?)",
            (run_id if run_id != "old-ok" else "run", source, status, observed),
        )
    await db.commit()
    result = await check_data_sources(
        db, since_24h=now - timedelta(hours=24), since_window=now - timedelta(days=7),
    )
    assert result.status == "FAIL"
    assert "all attempts failed" in result.detail


@pytest.mark.asyncio
async def test_data_sources_compare_timestamp_instants_not_lexical_strings(db: Database):
    """A negative offset must be normalized before applying the SLA window."""
    now = datetime.now(timezone.utc)
    observed = (now - timedelta(minutes=30)).astimezone(
        timezone(timedelta(hours=-6))
    ).isoformat()
    await db.execute(
        "INSERT INTO ingestion_runs (id,query,started_at) VALUES ('offset-run','q',?)",
        (observed,),
    )
    await db.execute(
        "INSERT INTO source_fetches (run_id,source,status,observed_at) "
        "VALUES ('offset-run','offset-source','ok',?)",
        (observed,),
    )
    await db.commit()

    result = await check_data_sources(
        db,
        since_24h=now - timedelta(hours=24),
        since_window=now - timedelta(days=7),
    )
    assert result.status == "PASS"


# ---------------------------------------------------------------------------
# Pass rate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pass_rate_pass_in_band(db: Database):
    await _seed_signals(db, n=30)
    await _seed_trades(db, pnls=[1.0])
    now = datetime.now(timezone.utc)
    result = await check_pass_rate(db, since=now - timedelta(days=7), exchange="kalshi")
    assert result.status == "PASS"


@pytest.mark.asyncio
async def test_pass_rate_fail_too_high(db: Database):
    await _seed_signals(db, n=30)
    await _seed_trades(db, pnls=[1.0] * 15)
    now = datetime.now(timezone.utc)
    result = await check_pass_rate(db, since=now - timedelta(days=7), exchange="kalshi")
    assert result.status == "FAIL"


# ---------------------------------------------------------------------------
# Brier (absolute)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_brier_absolute_pass_well_calibrated(db: Database):
    pairs = [(0.7, 1)] * 21 + [(0.7, 0)] * 9
    await _seed_calibration(db, pairs=pairs)
    now = datetime.now(timezone.utc)
    result = await check_brier_absolute(db, since=now - timedelta(days=7))
    assert result.status == "PASS"
    assert "0.210" in result.value


@pytest.mark.asyncio
async def test_brier_absolute_fail_overconfident(db: Database):
    pairs = [(0.95, 1)] * 15 + [(0.95, 0)] * 15
    await _seed_calibration(db, pairs=pairs)
    now = datetime.now(timezone.utc)
    result = await check_brier_absolute(db, since=now - timedelta(days=7))
    assert result.status == "FAIL"


# ---------------------------------------------------------------------------
# Brier vs market
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_brier_vs_market_pass_when_bot_better(db: Database):
    pairs = [(0.7, 1)] * 21 + [(0.7, 0)] * 9
    market_probs = [0.5] * 30
    await _seed_calibration(db, pairs=pairs, market_probs=market_probs)
    now = datetime.now(timezone.utc)
    result = await check_brier_vs_market(db, since=now - timedelta(days=7))
    assert result.status == "PASS"


@pytest.mark.asyncio
async def test_brier_vs_market_fail_when_market_better(db: Database):
    pairs = [(0.6, 1)] * 21 + [(0.6, 0)] * 9
    market_probs = [0.7] * 30
    await _seed_calibration(db, pairs=pairs, market_probs=market_probs)
    now = datetime.now(timezone.utc)
    result = await check_brier_vs_market(db, since=now - timedelta(days=7))
    assert result.status == "FAIL"


# ---------------------------------------------------------------------------
# Win rate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_win_rate_pass_above_52(db: Database):
    pnls = [1.0] * 16 + [-1.0] * 14
    await _seed_ledger(db, pnls=pnls)
    now = datetime.now(timezone.utc)
    result = await check_win_rate(db, since=now - timedelta(days=7), exchange="kalshi")
    assert result.status == "PASS"


@pytest.mark.asyncio
async def test_win_rate_fail_at_50(db: Database):
    pnls = [1.0] * 15 + [-1.0] * 15
    await _seed_ledger(db, pnls=pnls)
    now = datetime.now(timezone.utc)
    result = await check_win_rate(db, since=now - timedelta(days=7), exchange="kalshi")
    assert result.status == "FAIL"


# ---------------------------------------------------------------------------
# PnL after fees
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pnl_after_fees_pass_when_net_positive(db: Database):
    pnls = [5.0] * 20 + [-1.0] * 10
    await _seed_ledger(db, pnls=pnls)
    now = datetime.now(timezone.utc)
    result = await check_pnl_after_fees(
        db, since=now - timedelta(days=7), exchange="kalshi", fee_rate=0.07
    )
    assert result.status == "PASS"


@pytest.mark.asyncio
async def test_pnl_after_fees_uses_ledger_net_not_double_counted(db: Database):
    """pnl_ledger.pnl is ALREADY net of fees — the criterion must SUM it, not
    re-apply a fee estimate. Seed fees and assert net == sum(pnl) (gross adds
    the fees back for display)."""
    await _seed_ledger(db, pnls=[2.0] * 30, fees=0.10)
    now = datetime.now(timezone.utc)
    result = await check_pnl_after_fees(
        db, since=now - timedelta(days=7), exchange="kalshi", fee_rate=0.07
    )
    assert result.status == "PASS"
    assert "$+60.00" in result.value          # net = 30 * 2.0, not re-discounted
    assert "fees $3.00" in result.value       # 30 * 0.10, reported from ledger


@pytest.mark.asyncio
async def test_readiness_ignores_legacy_trades_pnl(db: Database):
    """Regression: the win-rate/pnl criteria must read pnl_ledger, NOT the legacy
    trades.pnl column (never populated in the current path). Seeding only the
    trades table must leave both criteria INSUFFICIENT_DATA."""
    await _seed_trades(db, pnls=[1.0] * 40)   # legacy table only, no ledger rows
    now = datetime.now(timezone.utc)
    wr = await check_win_rate(db, since=now - timedelta(days=7), exchange="kalshi")
    pf = await check_pnl_after_fees(
        db, since=now - timedelta(days=7), exchange="kalshi", fee_rate=0.07)
    assert wr.status == "INSUFFICIENT_DATA"
    assert pf.status == "INSUFFICIENT_DATA"


# ---------------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_divergence_pass_when_low(db: Database):
    await _seed_signals(db, n=30, divergence=0.10)
    now = datetime.now(timezone.utc)
    result = await check_divergence(
        db, since=now - timedelta(days=7), exchange="kalshi"
    )
    assert result.status == "PASS"


@pytest.mark.asyncio
async def test_divergence_fail_when_median_high(db: Database):
    await _seed_signals(db, n=30, divergence=0.20)
    now = datetime.now(timezone.utc)
    result = await check_divergence(
        db, since=now - timedelta(days=7), exchange="kalshi"
    )
    assert result.status == "FAIL"


# ---------------------------------------------------------------------------
# Top-level evaluator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_readiness_aggregates_all_eight_criteria(db: Database, tmp_path):
    log_file = tmp_path / "auramaur.log"
    log_file.write_text("")
    report = await evaluate_readiness(db, log_file=log_file, exchange="kalshi", days=7)
    assert len(report.criteria) == 8
    assert not report.overall_pass
    names = [c.name for c in report.criteria]
    assert names == [
        "cycle_health",
        "data_sources",
        "pass_rate",
        "brier_absolute",
        "brier_vs_market",
        "win_rate",
        "pnl_after_fees",
        "divergence",
    ]
