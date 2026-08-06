"""Degraded input must not read as a benign value.

Five sinks turned "I could not measure this" into a plausible number: an empty
Kraken balance reading as a flat book, an unreadable database reading as zero
resolved markets, a log window nobody could see reading as zero errors, and two
CWD-relative state paths reading as "not present" under any deployment that
relocates state.

Every test here executes the production path. An earlier revision of this file
asserted on ``inspect.getsource`` strings, which would have passed against
``if not bal and False: return`` — a test that cannot fail measures nothing.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auramaur.db.database import Database
from auramaur.exchange.models import OrderSide
from auramaur.treasury import kraken_pillar as kraken_pillar_module
from auramaur.treasury.kraken_pillar import KrakenPillar


# ---------------------------------------------------------------------------
# Kraken directional harness (mirrors tests/test_kraken_directional_exit.py)
# ---------------------------------------------------------------------------


def _kcfg(**over):
    base = dict(
        directional_enabled=True,
        directional_pairs=["APEUSDC"],
        directional_momentum_pct=3.0,
        directional_entry_momentum_pct=2.0,
        directional_exit_momentum_pct=4.0,
        directional_stop_loss_pct=12.0,
        directional_lookback=12,
        directional_budget_usd=50.0,
        max_order_usd=25.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _client(price, *, balance):
    """Fake KrakenSpotClient. Base asset APE; ``balance`` is what the wallet read
    returns — ``{}`` is the API-failure shape get_free_balance/get_balance
    produce on any error (kraken.py:121-123 logs and returns the empty result
    rather than raising)."""
    k = SimpleNamespace()
    k.get_balance = AsyncMock(return_value=dict(balance))
    k.get_free_balance = AsyncMock(return_value=dict(balance))
    k.get_pair_quote = AsyncMock(return_value="USDC")
    k.get_price = AsyncMock(return_value=price)
    k._public = AsyncMock(return_value={
        "APEUSDC": {"altname": "APEUSDC", "base": "APE", "ordermin": "1"},
    })
    k.usd_notional = AsyncMock(side_effect=lambda pair, vol, px=None: vol * (px or price))
    k.place_spot_order = AsyncMock(return_value=SimpleNamespace(
        order_id="OK", status="filled", error_message=""))
    k.size_for_usd = AsyncMock(return_value=4.0)
    return k


def _pillar(price, kcfg, *, balance, live=False, db=None):
    settings = SimpleNamespace(kraken=kcfg, is_live=live, risk_tolerance=50.0)
    bot = SimpleNamespace(_components={"db": db}) if db is not None else None
    return KrakenPillar(settings, _client(price, balance=balance), bot=bot)


@pytest.fixture
async def db(tmp_path):
    instance = Database(str(tmp_path / "degraded.db"))
    await instance.connect()
    yield instance
    await instance.close()


@pytest.fixture(autouse=True)
def _isolate_state_dir(tmp_path, monkeypatch):
    """Keep runtime paths (risk-tolerance override, db_path) off the real repo."""
    monkeypatch.setenv("AURAMAUR_STATE_DIR", str(tmp_path / "state"))


def _events(mock_method) -> list[str]:
    return [call.args[0] for call in mock_method.call_args_list if call.args]


# ---------------------------------------------------------------------------
# 1. Kraken balance outage — the SAFETY half: live state must survive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_balance_outage_preserves_position_state_and_places_nothing(db):
    """One 5xx / rate-limit / invalid-nonce made every tracked position look
    closed: _dir_long cleared, _clear_peak() DELETED the position_peaks row —
    the trailing-stop high-water mark, which is NOT recoverable from cost_basis.
    A position that peaked +40% and sat at +30% re-anchored at +30%."""
    pillar = _pillar(0.80, _kcfg(), balance={}, live=True, db=db)
    pillar._dir_long = {"APEUSDC": 1.00}
    pillar._momentum = AsyncMock(return_value=-1.0)
    await db.execute(
        "INSERT INTO position_peaks (market_id, peak_pnl_pct, updated_at) "
        "VALUES (?, ?, datetime('now'))", ("kraken-live:APEUSDC", 40.0))
    await db.commit()

    await pillar._directional()

    assert pillar._dir_long == {"APEUSDC": 1.00}, "the book must not be forgotten"
    row = await db.fetchone(
        "SELECT peak_pnl_pct FROM position_peaks WHERE market_id = ?",
        ("kraken-live:APEUSDC",))
    assert row is not None, "the trailing-stop high-water mark must survive"
    assert row["peak_pnl_pct"] == 40.0
    pillar._k.place_spot_order.assert_not_called()


@pytest.mark.asyncio
async def test_live_genuinely_flat_book_still_clears_the_position(db):
    """The discriminator for the test above: a SUCCESSFUL read that happens to
    hold no APE is a real close, and must still drop the position and its stale
    peak. A guard that returned unconditionally would break this."""
    pillar = _pillar(0.80, _kcfg(), balance={"USDC": 1000.0}, live=True, db=db)
    pillar._dir_long = {"APEUSDC": 1.00}
    pillar._momentum = AsyncMock(return_value=-1.0)
    await db.execute(
        "INSERT INTO position_peaks (market_id, peak_pnl_pct, updated_at) "
        "VALUES (?, ?, datetime('now'))", ("kraken-live:APEUSDC", 40.0))
    await db.commit()

    await pillar._directional()

    assert "APEUSDC" not in pillar._dir_long
    row = await db.fetchone(
        "SELECT peak_pnl_pct FROM position_peaks WHERE market_id = ?",
        ("kraken-live:APEUSDC",))
    assert row is None, "a real external close must still drop the stale peak"


# ---------------------------------------------------------------------------
# 2. Kraken balance outage — the AVAILABILITY half: paper must keep trading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_paper_book_still_exits_when_the_balance_is_unavailable(db):
    """Paper position state lives in kraken_paper_positions, not the wallet, so
    the guard must NOT stop the paper book from running its exit ladder. This is
    the availability half of the carve-out and the #407 regression shape."""
    pillar = _pillar(0.80, _kcfg(), balance={}, live=False, db=db)
    pillar._momentum = AsyncMock(return_value=-1.0)
    await db.execute(
        "INSERT INTO kraken_paper_positions "
        "(strategy, pair, quantity, entry_price, peak_gain_pct, opened_at, updated_at) "
        "VALUES ('momentum', 'APEUSDC', 10.0, 1.00, 0, datetime('now'), datetime('now'))")
    await db.commit()

    await pillar._directional()

    # Down 20% from entry 1.00 -> the 12% stop-loss fires even with no wallet.
    pillar._k.place_spot_order.assert_called_once()
    args = pillar._k.place_spot_order.call_args.args
    kwargs = pillar._k.place_spot_order.call_args.kwargs
    assert OrderSide.SELL in args
    assert kwargs["volume"] == 10.0          # the actual shadow-book quantity
    assert kwargs["dry_run"] is True         # paper stays validate-only
    assert "APEUSDC" not in pillar._dir_long
    row = await db.fetchone(
        "SELECT pair FROM kraken_paper_positions "
        "WHERE strategy='momentum' AND pair='APEUSDC'")
    assert row is None, "the paper position must be closed in the shadow book"


@pytest.mark.asyncio
async def test_paper_book_still_enters_when_the_balance_is_unavailable(db):
    """Paper entries deliberately skip the funding gate (no real quote balance
    is needed), so an empty wallet read must not silence the entry side either."""
    pillar = _pillar(1.00, _kcfg(), balance={}, live=False, db=db)
    pillar._momentum = AsyncMock(return_value=5.0)   # clears the 2% entry bar

    await pillar._directional()

    pillar._k.place_spot_order.assert_called_once()
    args = pillar._k.place_spot_order.call_args.args
    assert OrderSide.BUY in args
    row = await db.fetchone(
        "SELECT quantity FROM kraken_paper_positions "
        "WHERE strategy='momentum' AND pair='APEUSDC'")
    assert row is not None and row["quantity"] > 0


# ---------------------------------------------------------------------------
# 3. Kraken balance outage — escalation, so a revoked key cannot stay silent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_balance_outage_stays_a_warning(monkeypatch):
    """Measured transient rate on the live deployment: 5 empties / 865 reads
    (0.58%, ~once daily). A single skip is normal and must not alarm."""
    fake_log = MagicMock()
    monkeypatch.setattr(kraken_pillar_module, "log", fake_log)
    pillar = _pillar(1.00, _kcfg(), balance={}, live=True)

    await pillar._directional()

    assert "kraken.directional.balance_unavailable" in _events(fake_log.warning)
    assert "kraken.directional.balance_unavailable_sustained" not in _events(fake_log.error)


@pytest.mark.asyncio
async def test_sustained_balance_outage_escalates_to_error(monkeypatch):
    """A revoked/expired API key returns {} forever, and the live guard then
    skips the ENTIRE exit ladder behind a log.warning — a level readiness's
    _ERROR_LEVELS does not include, so nothing ever escalates."""
    fake_log = MagicMock()
    monkeypatch.setattr(kraken_pillar_module, "log", fake_log)
    pillar = _pillar(1.00, _kcfg(), balance={}, live=True)

    for _ in range(kraken_pillar_module._BALANCE_OUTAGE_ESCALATE_CYCLES):
        await pillar._directional()

    errors = _events(fake_log.error)
    assert errors.count("kraken.directional.balance_unavailable_sustained") == 1
    assert fake_log.error.call_args.kwargs["cycles"] == 3
    assert _events(fake_log.warning).count("kraken.directional.balance_unavailable") == 2
    # readiness reads structlog levels: only "error"/"critical" fail the check.
    from auramaur.monitoring.readiness import _ERROR_LEVELS
    assert "error" in _ERROR_LEVELS and "warning" not in _ERROR_LEVELS


@pytest.mark.asyncio
async def test_healthy_balance_read_resets_the_outage_counter(monkeypatch):
    fake_log = MagicMock()
    monkeypatch.setattr(kraken_pillar_module, "log", fake_log)
    pillar = _pillar(1.00, _kcfg(), balance={}, live=True)
    pillar._momentum = AsyncMock(return_value=0.0)

    await pillar._directional()
    await pillar._directional()
    assert pillar._empty_balance_cycles == 2

    pillar._k.get_free_balance = AsyncMock(return_value={"USDC": 1000.0})
    await pillar._directional()
    assert pillar._empty_balance_cycles == 0

    pillar._k.get_free_balance = AsyncMock(return_value={})
    await pillar._directional()
    await pillar._directional()
    assert "kraken.directional.balance_unavailable_sustained" not in _events(fake_log.error)


@pytest.mark.asyncio
async def test_outage_counter_is_not_reset_by_a_paper_live_flip(monkeypatch):
    """The counter must not leak across mode changes in a way that suppresses a
    real alert: two empty paper cycles plus one empty live cycle is still a
    15-minute blackout, and must escalate."""
    fake_log = MagicMock()
    monkeypatch.setattr(kraken_pillar_module, "log", fake_log)
    pillar = _pillar(1.00, _kcfg(), balance={}, live=False)
    pillar._momentum = AsyncMock(return_value=0.0)

    await pillar._directional()
    await pillar._directional()
    assert pillar._empty_balance_cycles == 2
    assert "kraken.directional.balance_unavailable_sustained" not in _events(fake_log.error)

    pillar._s.is_live = True
    await pillar._directional()

    assert "kraken.directional.balance_unavailable_sustained" in _events(fake_log.error)
    assert fake_log.error.call_args.kwargs["cycles"] == 3


# ---------------------------------------------------------------------------
# 4. Cycle health — a window nobody could see is not a clean one
# ---------------------------------------------------------------------------


def _line(level: str, event: str, ts: datetime) -> str:
    return json.dumps({
        "level": level,
        "timestamp": ts.isoformat().replace("+00:00", "Z"),
        "event": event,
    }) + "\n"


@pytest.mark.asyncio
async def test_empty_window_is_insufficient_data_not_pass(tmp_path):
    from auramaur.monitoring.readiness import check_cycle_health

    now = datetime.now(timezone.utc)
    log_file = tmp_path / "auramaur.log"
    log_file.write_text(_line("info", "engine.cycle_complete", now - timedelta(days=30)))

    result = await check_cycle_health(log_file, now - timedelta(days=7))

    assert result.status == "INSUFFICIENT_DATA"
    assert "no log entries in window" in result.value


@pytest.mark.asyncio
async def test_non_empty_window_without_errors_still_passes(tmp_path):
    from auramaur.monitoring.readiness import check_cycle_health

    now = datetime.now(timezone.utc)
    log_file = tmp_path / "auramaur.log"
    log_file.write_text(
        _line("info", "engine.cycle_complete", now - timedelta(hours=1))
        + _line("warning", "engine.skipped_junk", now - timedelta(hours=1))
    )

    result = await check_cycle_health(log_file, now - timedelta(days=7))

    assert result.status == "PASS"
    assert "0 errors" in result.value


@pytest.mark.asyncio
async def test_errors_in_window_still_fail(tmp_path):
    from auramaur.monitoring.readiness import check_cycle_health

    now = datetime.now(timezone.utc)
    log_file = tmp_path / "auramaur.log"
    log_file.write_text(
        _line("info", "engine.cycle_complete", now - timedelta(hours=1))
        + _line("error", "exchange.order_failed", now - timedelta(hours=1))
    )

    result = await check_cycle_health(log_file, now - timedelta(days=7))

    assert result.status == "FAIL"
    assert "1 error" in result.value
    assert "exchange.order_failed" in result.detail


@pytest.mark.asyncio
async def test_errors_carried_away_by_rotation_are_still_seen(tmp_path):
    """The rotation case the criterion always claimed to cover and never did.
    Against the live deployment the active file held 0.7 of the 7 days it
    advertises (10% coverage, rotating every ~21-30h) while 1,487 error events
    sat in .1/.2/.3. Reading only the active file scores this window PASS."""
    from auramaur.monitoring.readiness import check_cycle_health, log_files_for

    now = datetime.now(timezone.utc)
    log_file = tmp_path / "auramaur.log"
    log_file.write_text(_line("info", "engine.cycle_complete", now - timedelta(hours=1)))
    (tmp_path / "auramaur.log.1").write_text(
        _line("error", "burst.one_rotation_ago", now - timedelta(days=2)))
    (tmp_path / "auramaur.log.2").write_text(
        _line("critical", "burst.two_rotations_ago", now - timedelta(days=4)))

    assert [p.name for p in log_files_for(log_file)] == [
        "auramaur.log.2", "auramaur.log.1", "auramaur.log",
    ], "oldest first, so sampled events and the earliest timestamp stay chronological"

    result = await check_cycle_health(log_file, now - timedelta(days=7))

    assert result.status == "FAIL"
    assert "2 error/critical events" in result.value
    assert "burst.two_rotations_ago" in result.detail
    assert "burst.one_rotation_ago" in result.detail


@pytest.mark.asyncio
async def test_partial_coverage_is_reported_not_silently_scored(tmp_path):
    """A 7-day criterion scoring 2 days of log must say so."""
    from auramaur.monitoring.readiness import check_cycle_health

    now = datetime.now(timezone.utc)
    log_file = tmp_path / "auramaur.log"
    log_file.write_text(_line("info", "engine.cycle_complete", now - timedelta(days=2)))

    result = await check_cycle_health(log_file, now - timedelta(days=7))

    assert result.status == "PASS"
    assert "log covers 2.0d of 7.0d" in result.value


@pytest.mark.asyncio
async def test_full_coverage_adds_no_caveat(tmp_path):
    from auramaur.monitoring.readiness import check_cycle_health

    now = datetime.now(timezone.utc)
    log_file = tmp_path / "auramaur.log"
    log_file.write_text(
        _line("info", "engine.started", now - timedelta(days=9))
        + _line("info", "engine.cycle_complete", now - timedelta(hours=1))
    )

    result = await check_cycle_health(log_file, now - timedelta(days=7))

    assert result.status == "PASS"
    assert "log covers" not in result.value


@pytest.mark.asyncio
async def test_drift_canary_survives_the_multi_file_walk(tmp_path):
    """Unparseable-line tolerance and drift accounting are measured across the
    whole corpus, not reset per file."""
    from auramaur.monitoring.readiness import check_cycle_health

    now = datetime.now(timezone.utc)
    log_file = tmp_path / "auramaur.log"
    log_file.write_text(_line("info", "ok", now - timedelta(hours=1)))
    (tmp_path / "auramaur.log.1").write_text("this is not json\n" * 9)

    result = await check_cycle_health(log_file, now - timedelta(days=7))

    assert result.status == "FAIL"
    assert "format has drifted" in result.detail


# ---------------------------------------------------------------------------
# 5. Gate dashboard — an unreadable database is not an empty one
# ---------------------------------------------------------------------------


def _gate_settings():
    settings = MagicMock()
    settings.risk_tolerance = 50.0
    return settings


def test_unreadable_database_is_not_reported_as_no_data():
    """0 told the operator to keep waiting for a sample that already exists."""
    from auramaur.monitoring import gates

    assert gates._resolved_dollar_markets("/nonexistent/definitely/not/here.db") == -1


def test_gather_renders_unknown_rather_than_a_plausible_zero(tmp_path):
    from auramaur.monitoring import gates

    rows = gates.gather(_gate_settings(), db_path=str(tmp_path / "missing.db"))
    row = next(r for r in rows if r["feature"].startswith("Divergence filter"))

    assert row["verdict"] == "UNKNOWN (cannot read db)"
    assert row["status"] == "database unreadable"
    assert gates.render(rows) is not None      # the verdict string reaches the table


@pytest.mark.asyncio
async def test_gather_still_says_wait_for_a_readable_but_empty_database(db, tmp_path):
    """The discriminator: "no data yet" and "cannot read" must stay distinct."""
    from auramaur.monitoring import gates

    rows = gates.gather(_gate_settings(), db_path=str(tmp_path / "degraded.db"))
    row = next(r for r in rows if r["feature"].startswith("Divergence filter"))

    assert row["verdict"] == "WAIT (need data)"
    assert row["status"] == "0/100 resolved-$ markets"


def test_gather_defaults_to_the_configured_database(tmp_path, monkeypatch):
    """The old default was the bare literal "auramaur.db", which resolves only
    under a CWD that happens to be the repo root."""
    from auramaur.monitoring import gates

    target = tmp_path / "elsewhere" / "auramaur.db"
    monkeypatch.setenv("AURAMAUR_DB_PATH", str(target))
    seen: list[str] = []

    def _spy(path: str) -> int:
        seen.append(path)
        return 0

    monkeypatch.setattr(gates, "_resolved_dollar_markets", _spy)

    gates.gather(_gate_settings())

    assert seen == [str(target)]


# ---------------------------------------------------------------------------
# 6. Risk-tolerance override — anchored to the state dir, resolved at call time
# ---------------------------------------------------------------------------


def test_risk_tolerance_override_is_state_dir_anchored_and_call_time(tmp_path, monkeypatch):
    """A stale data/risk_tolerance in a launch directory silently put the book
    at YOLO; `auramaur risk 0` from elsewhere had no effect while saying it did.
    Resolving at call time is what makes the anchor followable at runtime."""
    from auramaur.risk import tolerance

    settings = SimpleNamespace(risk_tolerance=50.0)
    first = tmp_path / "first"
    monkeypatch.setenv("AURAMAUR_STATE_DIR", str(first))

    assert tolerance._override_path() == first / "data" / "risk_tolerance"
    assert tolerance._override_path().is_absolute()
    assert tolerance.current_tolerance(settings) == 50.0   # no file -> config

    tolerance.set_tolerance(0.0)
    assert (first / "data" / "risk_tolerance").read_text().strip() == "0.0"
    assert tolerance.current_tolerance(settings) == 0.0

    # Call-time resolution: an import-time constant would still be reading the
    # first path here, which is precisely why it was unmonkeypatchable.
    second = tmp_path / "second"
    monkeypatch.setenv("AURAMAUR_STATE_DIR", str(second))
    assert tolerance.current_tolerance(settings) == 50.0
    tolerance.set_tolerance(100.0)
    assert tolerance.current_tolerance(settings) == 100.0
    assert (first / "data" / "risk_tolerance").read_text().strip() == "0.0"


def test_risk_tolerance_override_is_clamped_and_ignores_junk(tmp_path, monkeypatch):
    from auramaur.risk import tolerance

    settings = SimpleNamespace(risk_tolerance=50.0)
    monkeypatch.setenv("AURAMAUR_STATE_DIR", str(tmp_path))

    tolerance.set_tolerance(500.0)
    assert tolerance.current_tolerance(settings) == 100.0

    tolerance._override_path().write_text("not a number")
    assert tolerance.current_tolerance(settings) == 50.0


# ---------------------------------------------------------------------------
# 7. dust-exit — attach to the database the deployment actually uses
# ---------------------------------------------------------------------------


def test_dust_exit_attaches_to_the_configured_database(tmp_path, monkeypatch):
    """Its ONLY concurrency guard is a flock on f'{db_path}.lock'. With a
    CWD-relative path that lock is uncontended, so the 'bot is running' refusal
    never engages and it can double-sell live positions."""
    from click.testing import CliRunner

    from auramaur.cli._base import main

    target = tmp_path / "state" / "auramaur.db"
    monkeypatch.setenv("AURAMAUR_DB_PATH", str(target))

    with patch("auramaur.cli.AuramaurBot") as MockBot:
        instance = MockBot.return_value
        instance._init_components = AsyncMock(
            side_effect=RuntimeError("Database is already locked by another instance"))
        result = CliRunner().invoke(main, ["dust-exit"])

    assert result.exit_code == 0, result.output
    assert MockBot.call_args.kwargs["db_path"] == str(Path(target))
    assert MockBot.call_args.kwargs["db_path"] != "auramaur.db"
