import ast
import inspect
import re
import textwrap
from datetime import timezone
from types import SimpleNamespace

import pytest

from auramaur.db.database import Database
from auramaur.db.models import SCHEMA_VERSION
from auramaur.exchange.models import TokenType
from auramaur.risk.exit_policy import (
    binary_exit_economics,
    lifecycle_profit_target,
    trailing_stop_triggered,
)
from auramaur.risk.portfolio import PortfolioTracker


def test_binary_profit_is_net_of_the_executable_exit_fee_only():
    """Only the exit fee is charged, because only the exit fee is realized.

    ``broker/pnl.py`` realizes ``(price - avg_cost) * size - fee`` on the SELL
    branch and builds the ledger event only there, so ``avg_cost`` is
    fee-exclusive and the entry fee never reaches ``pnl_ledger``. Charging it
    in the gate demanded a cost the accounting never records.
    """
    result = binary_exit_economics(
        entry_price=0.40, exit_price=0.60, size=10, fee_coefficient=0.25)
    assert result.gross_pnl_pct == pytest.approx(50.0)
    # 0.25 * 0.60 * 0.40 * 10 — the exit leg alone.
    assert result.estimated_fees == pytest.approx(0.60)
    assert result.net_pnl_pct == pytest.approx(35.0)
    # The entry leg survives as a diagnostic, and is NOT netted off.
    assert result.entry_fee == pytest.approx(0.60)
    assert result.net_pnl_pct == pytest.approx(
        result.gross_pnl_pct - result.estimated_fees / (0.40 * 10) * 100)

    short = binary_exit_economics(
        entry_price=0.60, exit_price=0.40, size=10,
        fee_coefficient=0.0, is_long=False)
    assert short.net_pnl_pct == pytest.approx(100 / 3)


def test_lifecycle_bands_and_trailing_policy_are_pure_and_configurable():
    args = dict(base_pct=50, early_pct=75, late_pct=25,
                early_fraction=0.5, late_fraction=0.1)
    assert lifecycle_profit_target(fraction_remaining=0.8, **args) == 75
    assert lifecycle_profit_target(fraction_remaining=0.3, **args) == 50
    assert lifecycle_profit_target(fraction_remaining=0.05, **args) == 25
    assert trailing_stop_triggered(peak_pct=20, current_pct=10,
                                   activation_pct=12, giveback_fraction=0.45)


def test_zero_disables_the_trailing_stop_instead_of_arming_it():
    """Zero in either knob must mean OFF, not 'fire on the first downtick'.

    Unguarded, ``activation_pct = 0`` arms the tier against every non-negative
    peak and ``giveback_fraction = 0`` reduces the giveback test to
    ``peak > current`` — so an operator disabling the tier would have got the
    most aggressive possible trailing stop instead.
    """
    armed = dict(peak_pct=50.0, current_pct=49.0)
    assert not trailing_stop_triggered(
        activation_pct=0.0, giveback_fraction=0.45, **armed)
    assert not trailing_stop_triggered(
        activation_pct=12.0, giveback_fraction=0.0, **armed)
    assert not trailing_stop_triggered(
        activation_pct=0.0, giveback_fraction=0.0, **armed)
    # And the configured tier is unaffected by the guard.
    assert trailing_stop_triggered(peak_pct=50.0, current_pct=10.0,
                                   activation_pct=12.0, giveback_fraction=0.45)


def test_config_accepts_zero_for_both_trailing_knobs():
    """``Field(gt=0)`` turned "disable the trailing stop" into a startup crash,
    stricter than the neighbouring stop_loss_pct / profit_target_pct."""
    from config.settings import ExecutionConfig

    cfg = ExecutionConfig(trailing_stop_activation_pct=0.0,
                          trailing_stop_giveback_fraction=0.0)
    assert cfg.trailing_stop_activation_pct == 0.0
    assert cfg.trailing_stop_giveback_fraction == 0.0


async def _seed_fills(db, rows):
    for order_id, token, side, size, paper, timestamp in rows:
        await db.execute(
            """INSERT INTO fills
               (order_id, market_id, token, side, size, price, is_paper, timestamp)
               VALUES (?, 'M1', ?, ?, ?, 0.5, ?, ?)""",
            (order_id, token, side, size, paper, timestamp))
    await db.commit()


@pytest.mark.asyncio
async def test_current_entry_uses_only_inventory_surviving_reentry(tmp_path):
    """Old round trips, the opposite token and the other book must not age the
    current inventory — even when they are NEWER than the entry we want.

    The ordering matters: the reverse-inventory walk starts at the newest fill,
    so a fixture whose newest fill is already the answer cannot detect a lost
    token or mode scope. Here both contaminants sit after the real entry, so
    dropping either scope returns a visibly different timestamp.
    """
    db = Database(str(tmp_path / "entry-age.db"))
    await db.connect()
    try:
        await _seed_fills(db, [
            ("old-buy", "YES", "BUY", 10, 1, "2026-01-01T00:00:00+00:00"),
            ("old-sell", "YES", "SELL", 10, 1, "2026-01-02T00:00:00+00:00"),
            # The entry that actually opened the held paper YES inventory.
            ("new-buy", "YES", "BUY", 5, 1, "2026-04-01T12:00:00+02:00"),
            # Newer, and each large enough to satisfy qty >= size on its own.
            ("no-buy", "NO", "BUY", 20, 1, "2026-05-01T00:00:00+00:00"),
            ("live-buy", "YES", "BUY", 20, 0, "2026-06-01T00:00:00+00:00"),
        ])
        tracker = PortfolioTracker(db)
        pos = SimpleNamespace(market_id="M1", token=TokenType.YES, size=5,
                              is_paper=True)
        entered = await tracker._current_position_entry_time(pos, 1)
        assert entered.isoformat() == "2026-04-01T10:00:00+00:00"
        assert entered.tzinfo == timezone.utc
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_entry_time_falls_back_to_trades_when_no_fill_ancestry(tmp_path):
    """No fill ancestry must not read as "entered just now".

    Positions predating the fills ledger — and live rows a venue mirror
    created without ever writing a fill — have no reverse-inventory answer.
    Returning None there pins ``fraction_remaining`` at exactly 1.0 on every
    tick, so the profit target freezes in its widest band and the near-expiry
    band becomes permanently unreachable.
    """
    db = Database(str(tmp_path / "trades-fallback.db"))
    await db.connect()
    try:
        for stamp in ("2026-02-01T00:00:00+00:00", "2026-03-01T00:00:00+00:00"):
            await db.execute(
                """INSERT INTO trades
                   (market_id, timestamp, side, size, price, is_paper)
                   VALUES ('M1', ?, 'BUY', 10, 0.5, 1)""", (stamp,))
        await db.commit()
        tracker = PortfolioTracker(db)
        pos = SimpleNamespace(market_id="M1", token=TokenType.YES, size=10,
                              is_paper=True)
        entered = await tracker._current_position_entry_time(pos, 1)
        assert entered is not None, "an old position must not read as brand new"
        assert entered.isoformat() == "2026-02-01T00:00:00+00:00"

        # The fallback is still book-scoped: the live book has no trades here.
        live = SimpleNamespace(market_id="M1", token=TokenType.YES, size=10,
                               is_paper=False)
        assert await tracker._current_position_entry_time(live, 0) is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_fills_a_hair_short_of_the_stored_size_still_count(tmp_path):
    """Float accumulation must not erase a position's whole fill ancestry.

    0.7 + 0.1 is 0.7999999999999999 in IEEE-754, so an exact ``qty >= 0.8``
    reports NO ancestry for a position whose fills plainly cover it.
    """
    db = Database(str(tmp_path / "epsilon.db"))
    await db.connect()
    try:
        await _seed_fills(db, [
            ("a", "YES", "BUY", 0.7, 1, "2026-01-01T00:00:00+00:00"),
            ("b", "YES", "BUY", 0.1, 1, "2026-01-02T00:00:00+00:00"),
        ])
        tracker = PortfolioTracker(db)
        pos = SimpleNamespace(market_id="M1", token=TokenType.YES, size=0.8,
                              is_paper=True)
        entered = await tracker._current_position_entry_time(pos, 1)
        assert entered is not None, "an exact >= comparison lost this position"
        assert entered.isoformat() == "2026-01-01T00:00:00+00:00"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_unscoped_read_uses_the_positions_own_book_not_paper(tmp_path):
    """An unscoped read must ask the position which book it belongs to.

    ``check_exits`` passes ``is_paper=None`` when ``settings.is_live`` is not a
    bool, leaving ``get_positions`` unscoped over BOTH books. Defaulting that
    to paper answered a LIVE position's lifecycle question off the paper
    ledger — silently, and with no way for the caller to notice. Since #420 the
    Position carries ``is_paper``, so it can answer for itself.
    """
    db = Database(str(tmp_path / "mode.db"))
    await db.connect()
    try:
        await _seed_fills(db, [
            ("live", "YES", "BUY", 5, 0, "2026-01-01T00:00:00+00:00"),
            ("paper", "YES", "BUY", 5, 1, "2026-02-01T00:00:00+00:00"),
        ])
        tracker = PortfolioTracker(db)
        live = SimpleNamespace(market_id="M1", token=TokenType.YES, size=5,
                               is_paper=False)
        paper = SimpleNamespace(market_id="M1", token=TokenType.YES, size=5,
                                is_paper=True)
        assert (await tracker._current_position_entry_time(live, None)).isoformat() \
            == "2026-01-01T00:00:00+00:00"
        assert (await tracker._current_position_entry_time(paper, None)).isoformat() \
            == "2026-02-01T00:00:00+00:00"
        # An explicit mode flag still wins over the position's own field.
        assert (await tracker._current_position_entry_time(live, 1)).isoformat() \
            == "2026-02-01T00:00:00+00:00"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_v50_migration_adds_exit_decision_telemetry(tmp_path):
    """Call the migration DIRECTLY, never through connect().

    ``_init_schema`` runs ``executescript(TABLES)`` before dispatching
    migrations, so a reconnect-and-migrate test rebuilds the table from
    TABLES and then runs the migration against a schema that already
    satisfies it — the migration's own DDL is never observed, and gutting
    it leaves the test green (verified 2026-08-06).
    """
    path = tmp_path / "migration.db"
    db = Database(str(path))
    await db.connect()
    try:
        await db.execute("DROP TABLE exit_decisions")
        await db.execute("UPDATE schema_version SET version = 49")

        await db._migrate_v49_to_v50()

        version = await db.fetchone("SELECT version FROM schema_version")
        columns = await db.fetchall("PRAGMA table_info(exit_decisions)")
        assert version["version"] == 50
        assert {row["name"] for row in columns} >= {
            "policy_action", "net_pnl_pct", "estimated_fees", "peak_pnl_pct"
        }
        # Idempotent: a second dispatch must not raise or lose rows.
        await db._migrate_v49_to_v50()
        assert (await db.fetchone(
            "SELECT version FROM schema_version"))["version"] == 50
    finally:
        await db.close()


async def _prune_plan(db) -> list[str]:
    """Query plan for the SHIPPED retention delete, not a copy of it."""
    rows = await db.fetchall(
        f"EXPLAIN QUERY PLAN {PortfolioTracker._EXIT_DECISION_PRUNE}",
        ("-14 days",))
    return [row["detail"] for row in rows]


@pytest.mark.asyncio
async def test_retention_prune_never_scans_the_exit_decision_table(tmp_path):
    """The per-cycle prune must SEEK, on a fresh schema and a migrated one.

    ``idx_exit_decisions_position_time`` looks like it covers this — it ends in
    ``observed_at`` — but that is its FOURTH column and the predicate
    constrains none of the three before it, so SQLite has no seekable prefix
    and reports ``SCAN exit_decisions``. That scan runs inside the
    ``portfolio.exit_decisions`` write span on every exit cycle, holding the
    process-wide write lock against a table that only grows: the one place
    where a full scan costs closing a position.

    Both schema paths are checked because they are written in different files
    and can drift: a fresh database gets ``db/models.py``'s TABLES script, an
    existing one gets the v49 -> v50 migration in ``db/database.py``.

    The migration half calls the step DIRECTLY rather than reconnecting.
    ``_init_schema`` executes TABLES *before* dispatching migrations, so a
    reconnect would create the table and its indexes from TABLES and then run
    the migration against a schema that already satisfies it — the migration's
    own DDL would never be observed, and an index missing there would pass.
    """
    fresh = Database(str(tmp_path / "fresh.db"))
    await fresh.connect()
    try:
        plan = await _prune_plan(fresh)
        assert not any("SCAN" in step for step in plan), (
            f"retention delete full-scans a growing table: {plan}")
        assert any("SEARCH" in step for step in plan), plan
    finally:
        await fresh.close()

    # And the same for a database that arrives at v50 by migration.
    migrated = Database(str(tmp_path / "migrated-plan.db"))
    await migrated.connect()
    try:
        # Dropping the table takes its indexes with it, so what follows is
        # built by the migration script alone.
        await migrated.execute("DROP TABLE exit_decisions")
        await migrated.execute("UPDATE schema_version SET version = 49")
        await migrated.commit()
        await migrated._migrate_v49_to_v50()

        plan = await _prune_plan(migrated)
        assert not any("SCAN" in step for step in plan), (
            f"the v49 -> v50 migration omitted the index: {plan}")
        assert any("SEARCH" in step for step in plan), plan
    finally:
        await migrated.close()


_MIGRATION_STEP = re.compile(r"^_migrate_v(\d+)_to_v(\d+)$")


def test_every_schema_step_dispatches_to_a_distinct_migration():
    """Two branches claiming the same version step must be impossible to merge.

    A duplicate ``async def _migrate_vN_to_vM`` in the class body is silently
    shadowed — Python keeps the last definition and ruff does not flag it — so
    one migration's DDL never runs against a live database while the version
    counter still advances. That collision has now happened twice, each time
    surviving review and CI.

    Three properties close it: every step is DECLARED once, every declared step
    is DISPATCHED exactly once, and the dispatch resolves to as many distinct
    function objects as there are version steps.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(Database))).body[0]
    definitions = [
        node.name for node in tree.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        and _MIGRATION_STEP.match(node.name)
    ]
    shadowed = sorted({n for n in definitions if definitions.count(n) > 1})
    assert not shadowed, f"silently shadowed migration definitions: {shadowed}"

    runner = next(node for node in tree.body
                  if isinstance(node, ast.AsyncFunctionDef)
                  and node.name == "_run_migrations")
    dispatched = [
        node.func.attr for node in ast.walk(runner)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and _MIGRATION_STEP.match(node.func.attr)
    ]
    repeated = sorted({n for n in dispatched if dispatched.count(n) > 1})
    assert not repeated, f"migration dispatched more than once: {repeated}"
    assert set(dispatched) == set(definitions), (
        "declared and dispatched migrations disagree: "
        f"{sorted(set(definitions) ^ set(dispatched))}")

    # The property stated directly: as many DISTINCT method objects as steps.
    assert len({getattr(Database, name) for name in dispatched}) == len(dispatched)

    # And the steps form one unbroken chain up to the declared schema version,
    # so a renumber cannot leave a hole or overshoot SCHEMA_VERSION.
    targets = sorted(int(_MIGRATION_STEP.match(n).group(2)) for n in dispatched)
    assert targets == list(range(targets[0], SCHEMA_VERSION + 1))
    for name in dispatched:
        start, end = (int(g) for g in _MIGRATION_STEP.match(name).groups())
        assert end == start + 1, f"{name} is not a single-version step"


def test_degraded_default_matches_the_tracked_setting():
    """The module fallback and the tracked default must not drift apart —
    a degraded config must never retain LONGER than a healthy one."""
    from config.settings import Settings

    from auramaur.risk import portfolio as portfolio_mod

    assert (portfolio_mod._DEFAULT_RETENTION_DAYS
            == Settings().execution.exit_decision_retention_days)


async def test_exit_decision_write_is_one_atomic_span(tmp_path):
    """Insert batch and retention prune land together or not at all.

    Nothing pinned the span before 2026-08-06: replacing the
    ``db.transaction`` context manager with ``if True:`` left the suite
    green, so a partial write (rows inserted, prune skipped) was
    undetectable.
    """
    db = Database(str(tmp_path / "span.db"))
    await db.connect()
    try:
        tracker = PortfolioTracker(db)
        owners: list[str] = []
        original = db.transaction

        def _spy(owner=None):
            owners.append(owner or "")
            return original(owner=owner)

        db.transaction = _spy
        rows = [(
            "2026-08-06 00:00:00", "m1", "polymarket", "YES", 1, "HOLD",
            10.0, 9.0, 12.0, 50.0, 0.1, 0.6, 0.5, 10.0,
        )]
        await tracker._record_exit_decisions(rows, 3)
        assert owners == ["portfolio.exit_decisions"], (
            "the batch must take exactly one named span")
    finally:
        db.transaction = original
        await db.close()
