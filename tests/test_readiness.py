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
    check_exit_liveness,
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


@pytest.mark.asyncio
async def test_data_sources_ignore_zero_influence_shadow_health(db: Database):
    now = datetime.now(timezone.utc)
    for values in (
        ("shadow", "nws", "error", now.isoformat(), "shadow"),
        ("production", "rss", "ok", now.isoformat(), "production"),
    ):
        await db.execute(
            "INSERT INTO source_fetches "
            "(run_id,source,status,observed_at,information_mode) VALUES (?,?,?,?,?)",
            values,
        )
    await db.commit()
    result = await check_data_sources(
        db, since_24h=now - timedelta(days=1), since_window=now - timedelta(days=7),
    )
    assert result.status == "PASS"
    assert result.value == "1 active"


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
# Exit liveness — "entries continue but exits stopped"
# ---------------------------------------------------------------------------


async def _seed_entries(
    db: Database,
    *,
    n: int,
    venue: str = "polymarket",
    book: str = "llm",
    is_paper: int = 0,
    days_ago: float = 1.0,
    status: str = "filled",
) -> None:
    """BUY rows on the gateway's trades mirror — the entry side of a cell."""
    for i in range(n):
        ts = (
            datetime.now(timezone.utc) - timedelta(days=days_ago, minutes=i)
        ).isoformat()
        await db.execute(
            "INSERT INTO trades (market_id, exchange, timestamp, side, size, "
            "price, is_paper, status, strategy_source) "
            "VALUES (?, ?, ?, 'BUY', 10.0, 0.50, ?, ?, ?)",
            (f"{book}-{is_paper}-{days_ago}-{i}", venue, ts, is_paper, status, book),
        )
    await db.commit()


async def _seed_exits(
    db: Database,
    *,
    n: int,
    venue: str = "polymarket",
    book: str = "llm",
    is_paper: int = 0,
    days_ago: float = 1.0,
    kind: str = "sell",
) -> None:
    """Realization rows — the exit side. The ledger attributes a realization
    to the strategy that OPENED the position, which is why exits are counted
    from here and not from the trades mirror (whose SELL rows are attributed
    to the exit path itself)."""
    for i in range(n):
        ts = (
            datetime.now(timezone.utc) - timedelta(days=days_ago, minutes=i)
        ).isoformat()
        await db.execute(
            "INSERT INTO pnl_ledger (market_id, venue, strategy_source, kind, "
            "token, qty, pnl, fees, is_paper, source_ref, realized_at) "
            "VALUES (?, ?, ?, ?, 'YES', 10.0, 1.0, 0.0, ?, ?, ?)",
            (f"{book}-{is_paper}-{i}", venue, book, kind, is_paper,
             f"exit:{venue}:{book}:{is_paper}:{kind}:{days_ago}:{i}", ts),
        )
    await db.commit()


@pytest.mark.asyncio
async def test_exit_liveness_fail_when_entries_continue_and_exits_are_absent(
    db: Database,
):
    """The core contract: a book with entries above the floor and ZERO exit
    events fails, and the failure names the cell and both counts."""
    await _seed_exits(db, n=4, days_ago=20.0)          # established cadence
    await _seed_entries(db, n=6, days_ago=2.0)         # entries kept coming
    now = datetime.now(timezone.utc)
    result = await check_exit_liveness(
        db, since=now - timedelta(days=7), exchange=None)
    assert result.status == "FAIL"
    assert "polymarket/llm [live]" in result.detail
    assert "6 entries" in result.detail
    assert "0 exits" in result.detail
    assert result.n_samples == 1


@pytest.mark.asyncio
async def test_exit_liveness_pass_when_a_book_enters_and_exits(db: Database):
    await _seed_exits(db, n=4, days_ago=20.0)
    await _seed_entries(db, n=6, days_ago=2.0)
    await _seed_exits(db, n=2, days_ago=1.0)
    now = datetime.now(timezone.utc)
    result = await check_exit_liveness(
        db, since=now - timedelta(days=7), exchange=None)
    assert result.status == "PASS"
    assert "polymarket/llm [live]" in result.detail
    assert result.n_samples == 1


@pytest.mark.asyncio
async def test_exit_liveness_dormant_book_with_no_entries_never_fails(db: Database):
    """A strategy that stopped trading is not a broken one. Exits present in
    history, none in the window, and no entries in the window either."""
    await _seed_exits(db, n=10, days_ago=20.0)
    now = datetime.now(timezone.utc)
    result = await check_exit_liveness(
        db, since=now - timedelta(days=7), exchange=None)
    assert result.status != "FAIL"
    assert result.status == "INSUFFICIENT_DATA"
    assert result.n_samples == 0


@pytest.mark.asyncio
async def test_exit_liveness_settlements_alone_count_as_exits(db: Database):
    """A long-dated book can go a long time with no SELL while positions
    resolve. Settlements are exits; without this the criterion fires forever
    on long_horizon."""
    await _seed_exits(db, n=2, book="long_horizon", days_ago=20.0,
                      kind="settlement")
    await _seed_entries(db, n=8, book="long_horizon", days_ago=3.0)
    await _seed_exits(db, n=3, book="long_horizon", days_ago=1.0,
                      kind="settlement")
    now = datetime.now(timezone.utc)
    result = await check_exit_liveness(
        db, since=now - timedelta(days=7), exchange=None)
    assert result.status == "PASS"
    assert "3 exits" in result.detail


@pytest.mark.asyncio
async def test_exit_liveness_commissions_are_not_exits(db: Database):
    """A commission is booked against a still-open position. Money moving is
    not a position leaving, so it must not mask a stalled exit path."""
    await _seed_exits(db, n=3, days_ago=20.0)
    await _seed_entries(db, n=5, days_ago=2.0)
    await _seed_exits(db, n=9, days_ago=1.0, kind="commission")
    now = datetime.now(timezone.utc)
    result = await check_exit_liveness(
        db, since=now - timedelta(days=7), exchange=None)
    assert result.status == "FAIL"
    assert "0 exits" in result.detail


@pytest.mark.asyncio
async def test_exit_liveness_judges_paper_and_live_independently(db: Database):
    """Paper exits kept working throughout the incident, which is exactly why
    nothing looked wrong in aggregate. The same book, same venue, healthy in
    paper and stalled in live: the paper half must not vouch for the live
    half. Collapse the two modes into one cell and this FAIL becomes a PASS.
    """
    await _seed_exits(db, n=3, is_paper=1, days_ago=20.0)
    await _seed_exits(db, n=3, is_paper=0, days_ago=20.0)
    await _seed_entries(db, n=6, is_paper=1, days_ago=2.0)
    await _seed_entries(db, n=6, is_paper=0, days_ago=2.0)
    await _seed_exits(db, n=4, is_paper=1, days_ago=1.0)   # paper still exiting
    now = datetime.now(timezone.utc)
    result = await check_exit_liveness(
        db, since=now - timedelta(days=7), exchange=None)
    assert result.status == "FAIL"
    assert result.detail == "polymarket/llm [live]: 6 entries, 0 exits"
    assert result.n_samples == 2                           # both cells judged


@pytest.mark.asyncio
async def test_exit_liveness_entry_floor_keeps_a_single_entry_from_failing(
    db: Database,
):
    """One entry proves nothing — a single position can legitimately be held."""
    await _seed_exits(db, n=3, days_ago=20.0)
    await _seed_entries(db, n=2, days_ago=2.0)
    now = datetime.now(timezone.utc)
    below = await check_exit_liveness(
        db, since=now - timedelta(days=7), exchange=None, min_entries=3)
    assert below.status == "INSUFFICIENT_DATA"
    at_floor = await check_exit_liveness(
        db, since=now - timedelta(days=7), exchange=None, min_entries=2)
    assert at_floor.status == "FAIL"


@pytest.mark.asyncio
async def test_exit_liveness_book_that_never_exited_is_not_judged(db: Database):
    """"Stopped" presupposes it was running. A brand-new book's first
    realization is genuinely weeks away, so it is reported as not-yet-
    judgeable instead of failed."""
    await _seed_entries(db, n=9, book="term_structure", days_ago=2.0)
    now = datetime.now(timezone.utc)
    result = await check_exit_liveness(
        db, since=now - timedelta(days=7), exchange=None)
    assert result.status == "INSUFFICIENT_DATA"
    assert "no exit ever recorded" in result.detail
    assert "polymarket/term_structure [live]" in result.detail


@pytest.mark.asyncio
async def test_exit_liveness_ignores_unattributed_entry_buckets(db: Database):
    """order_monitor / exit / legacy_unattributed are attribution buckets, not
    books. The ledger refuses to credit them as the entrant, so a realization
    can never be booked back to them — judging them is a permanent false
    alarm. Each bucket is given a realistic exit history and then entries with
    no exits in the window, which is exactly the shape that would fail a real
    book."""
    for bucket in ("order_monitor", "exit", "legacy_unattributed"):
        await _seed_exits(db, n=3, book=bucket, days_ago=20.0)
        await _seed_entries(db, n=8, book=bucket, days_ago=2.0)
    await _seed_exits(db, n=2, days_ago=20.0)              # one genuine book
    await _seed_entries(db, n=5, days_ago=2.0)
    await _seed_exits(db, n=2, days_ago=1.0)
    now = datetime.now(timezone.utc)
    result = await check_exit_liveness(
        db, since=now - timedelta(days=7), exchange=None)
    assert result.status == "PASS"
    assert result.detail == "polymarket/llm [live]: 5 entries, 2 exits"
    assert result.n_samples == 1


@pytest.mark.asyncio
async def test_exit_liveness_respects_the_exchange_filter(db: Database):
    """Scoping to one venue must not let another venue's cells leak into the
    verdict — in either direction."""
    await _seed_exits(db, n=3, venue="kalshi", days_ago=20.0)
    await _seed_entries(db, n=6, venue="kalshi", days_ago=2.0)   # kalshi stalled
    await _seed_exits(db, n=3, venue="polymarket", days_ago=20.0)
    await _seed_entries(db, n=6, venue="polymarket", days_ago=2.0)
    await _seed_exits(db, n=2, venue="polymarket", days_ago=1.0)  # poly healthy
    now = datetime.now(timezone.utc)

    poly = await check_exit_liveness(
        db, since=now - timedelta(days=7), exchange="polymarket")
    assert poly.status == "PASS"
    assert "kalshi" not in poly.detail

    kalshi = await check_exit_liveness(
        db, since=now - timedelta(days=7), exchange="kalshi")
    assert kalshi.status == "FAIL"
    assert kalshi.detail == "kalshi/llm [live]: 6 entries, 0 exits"

    both = await check_exit_liveness(
        db, since=now - timedelta(days=7), exchange=None)
    assert both.status == "FAIL"
    assert both.n_samples == 2


@pytest.mark.asyncio
async def test_exit_liveness_detects_the_prediction_market_exit_outage(db: Database):
    """Reconstruction of the incident this criterion exists to catch.

    The live prediction-market exit loop raised on every tick for an extended
    period. Entries kept being taken, live SELL fills went to zero, and the
    only live realizations left were settlements on OTHER books' positions.
    Paper exits were unaffected. Readiness reported clean the whole time
    because cycle_health scores log level and the raise was logged at debug.
    """
    now = datetime.now(timezone.utc)
    # Before the outage: the live book was entering AND exiting normally.
    for day in (14.0, 12.0, 10.0):
        await _seed_entries(db, n=2, days_ago=day)
        await _seed_exits(db, n=2, days_ago=day)
    # The outage: entries accrue every day, not one live realization lands.
    for day in (6.0, 5.0, 4.0, 3.0, 2.0, 1.0):
        await _seed_entries(db, n=2, days_ago=day)
    # Paper kept entering and exiting throughout — the reason it looked fine.
    await _seed_exits(db, n=2, is_paper=1, days_ago=10.0)
    for day in (6.0, 4.0, 2.0):
        await _seed_entries(db, n=2, is_paper=1, days_ago=day)
        await _seed_exits(db, n=2, is_paper=1, days_ago=day)
    # A different live book still settled, so venue-level activity was not zero.
    await _seed_exits(db, n=3, book="market_maker", days_ago=3.0,
                      kind="settlement")

    result = await check_exit_liveness(
        db, since=now - timedelta(days=7), exchange=None)
    assert result.status == "FAIL"
    assert "polymarket/llm [live]: 12 entries, 0 exits" in result.detail
    assert "[paper]" not in result.detail

    # And the same window BEFORE the outage began must be clean, so the
    # failure is attributable to the outage and not to the fixture shape.
    healthy = await check_exit_liveness(
        db, since=now - timedelta(days=15), exchange=None,
        min_entries=100)
    assert healthy.status == "INSUFFICIENT_DATA"


# ---------------------------------------------------------------------------
# Top-level evaluator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_readiness_aggregates_all_ten_criteria(db: Database, tmp_path):
    log_file = tmp_path / "auramaur.log"
    log_file.write_text("")
    report = await evaluate_readiness(db, log_file=log_file, exchange="kalshi", days=7)
    assert len(report.criteria) == 10
    assert not report.overall_pass
    names = [c.name for c in report.criteria]
    assert names == [
        "cycle_health",
        "data_sources",
        "strategy_data_delivery",
        "pass_rate",
        "brier_absolute",
        "brier_vs_market",
        "win_rate",
        "pnl_after_fees",
        "divergence",
        "exit_liveness",
    ]


@pytest.mark.asyncio
async def test_evaluate_readiness_surfaces_a_stalled_exit_path(
    db: Database, tmp_path,
):
    """The criterion must reach the report, not just exist. A live book that
    keeps entering while its exits are absent has to show up as a FAIL in the
    aggregate the operator actually reads."""
    log_file = tmp_path / "auramaur.log"
    log_file.write_text("")
    await _seed_exits(db, n=4, venue="kalshi", days_ago=20.0)
    await _seed_entries(db, n=6, venue="kalshi", days_ago=2.0)
    report = await evaluate_readiness(db, log_file=log_file, exchange="kalshi", days=7)
    criterion = next(c for c in report.criteria if c.name == "exit_liveness")
    assert criterion.status == "FAIL"
    assert "kalshi/llm [live]" in criterion.detail
    assert not report.overall_pass


@pytest.mark.asyncio
async def test_evaluate_readiness_defaults_to_configured_log_file(
    db: Database, tmp_path, monkeypatch
):
    """With no explicit log_file, cycle_health must read the configured
    logging path (LOGGING__FILE / logging.file) — the container sets it to
    /app/logs/auramaur.log, not CWD/auramaur.log."""
    log_file = tmp_path / "logs" / "auramaur.log"
    log_file.parent.mkdir()
    now = datetime.now(timezone.utc)
    log_file.write_text(
        json.dumps(
            {"level": "info", "timestamp": now.isoformat(), "event": "cycle.ok"}
        )
        + "\n"
    )
    monkeypatch.setenv("LOGGING__FILE", str(log_file))
    report = await evaluate_readiness(db, exchange="kalshi", days=7)
    cycle = report.criteria[0]
    assert cycle.name == "cycle_health"
    assert cycle.status == "PASS"
    # The count of files folded in is part of the detail now that cycle_health
    # also reads the rotated backups (log_files_for) — here there are none.
    assert cycle.detail == "1 entries in window across 1 log file(s)"


@pytest.mark.asyncio
async def test_evaluate_readiness_explicit_log_file_wins(
    db: Database, tmp_path, monkeypatch
):
    monkeypatch.setenv("LOGGING__FILE", str(tmp_path / "ignored.log"))
    explicit = tmp_path / "explicit.log"
    explicit.write_text("")
    report = await evaluate_readiness(db, log_file=explicit, exchange="kalshi", days=7)
    cycle = report.criteria[0]
    assert cycle.status == "INSUFFICIENT_DATA"
    assert cycle.detail == "log file is empty"
