"""Degraded input must not read as a benign value.

Four sinks turned "I could not measure this" into a plausible number: an empty
Kraken balance reading as a flat book, an unreadable database reading as zero
resolved markets, an empty log window reading as zero errors, and two
CWD-relative state paths reading as "not present" under any deployment that
relocates state.
"""

import inspect


def test_directional_skips_the_cycle_when_the_balance_is_unavailable():
    """get_free_balance falls back to get_balance, which returns {} on ANY API
    error rather than raising — so one 5xx made every position look closed,
    cleared _dir_long, and DELETED the position_peaks trailing-stop marks,
    which are not recoverable from cost_basis. _treasury:419 already guards."""
    from auramaur.treasury import kraken_pillar
    src = inspect.getsource(kraken_pillar.KrakenPillar._directional)
    guard = src.split("get_free_balance()", 1)[1].split("_resolve_pairs", 1)[0]
    assert "if not bal" in guard and "return" in guard


def test_unreadable_database_is_not_reported_as_no_data():
    """0 told the operator to keep waiting for a sample that already exists."""
    from auramaur.monitoring import gates
    n = gates._resolved_dollar_markets("/nonexistent/definitely/not/here.db")
    assert n == -1, "an unreadable db must be distinguishable from an empty one"

    src = inspect.getsource(gates.gather)
    assert "runtime_db_path()" in src, "must follow AURAMAUR_DB_PATH"
    rendered = inspect.getsource(gates)
    assert "UNKNOWN (cannot read db)" in rendered


def test_empty_log_window_is_insufficient_data_not_pass():
    """The window can be empty because the bot is dead, or because rotation
    carried the errors away — _rotate() renames at rotate_max_mb and the
    parser opens exactly one path."""
    from auramaur.monitoring import readiness
    src = inspect.getsource(readiness)
    assert 'if in_window == 0:' in src
    empty_arm = src.split("if in_window == 0:", 1)[1].split("if errors == 0:", 1)[0]
    assert 'status="INSUFFICIENT_DATA"' in empty_arm
    # overall_pass requires every criterion to be PASS, so this blocks it.
    assert 'all(c.status == "PASS" for c in self.criteria)' in src


def test_risk_tolerance_override_is_anchored_to_the_state_dir():
    """A stale data/risk_tolerance in a launch directory silently put the book
    at YOLO; `auramaur risk 0` from elsewhere had no effect while saying it did."""
    from auramaur.risk import tolerance
    assert tolerance._OVERRIDE.is_absolute()
    assert "state_dir()" in inspect.getsource(tolerance).split("_OVERRIDE =", 1)[1][:80]


def test_dust_exit_attaches_to_the_real_database():
    """Its ONLY concurrency guard is a flock on f'{db_path}.lock'. With a
    CWD-relative path that lock is uncontended, so the 'bot is running' refusal
    never engages and it can double-sell live positions."""
    from auramaur.cli import redeem
    src = inspect.getsource(redeem)
    assert 'db_path=str(runtime_db_path())' in src
    assert 'db_path="auramaur.db"' not in src
