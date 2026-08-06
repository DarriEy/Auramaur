from datetime import timezone
from types import SimpleNamespace

import pytest

from auramaur.db.database import Database
from auramaur.exchange.models import TokenType
from auramaur.risk.exit_policy import (
    binary_exit_economics,
    lifecycle_profit_target,
    trailing_stop_triggered,
)
from auramaur.risk.portfolio import PortfolioTracker


def test_binary_profit_is_net_of_executable_exit_fee():
    result = binary_exit_economics(
        entry_price=0.40, exit_price=0.60, size=10, fee_coefficient=0.25)
    assert result.gross_pnl_pct == pytest.approx(50.0)
    assert result.estimated_fees == pytest.approx(1.20)
    assert result.net_pnl_pct == pytest.approx(20.0)

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


@pytest.mark.asyncio
async def test_current_entry_uses_only_inventory_surviving_reentry(tmp_path):
    db = Database(str(tmp_path / "entry-age.db"))
    await db.connect()
    try:
        # Old round trip, then the current position. NO and live fills must not
        # contaminate the paper YES inventory clock.
        rows = [
            ("old-buy", "YES", "BUY", 10, 1, "2026-01-01T00:00:00+00:00"),
            ("old-sell", "YES", "SELL", 10, 1, "2026-01-02T00:00:00+00:00"),
            ("no-buy", "NO", "BUY", 20, 1, "2026-02-01T00:00:00+00:00"),
            ("live-buy", "YES", "BUY", 20, 0, "2026-03-01T00:00:00+00:00"),
            ("new-buy", "YES", "BUY", 5, 1, "2026-04-01T12:00:00+02:00"),
        ]
        for order_id, token, side, size, paper, timestamp in rows:
            await db.execute(
                """INSERT INTO fills
                   (order_id, market_id, token, side, size, price, is_paper, timestamp)
                   VALUES (?, 'M1', ?, ?, ?, 0.5, ?, ?)""",
                (order_id, token, side, size, paper, timestamp))
        await db.commit()
        tracker = PortfolioTracker(db)
        pos = SimpleNamespace(market_id="M1", token=TokenType.YES, size=5)
        entered = await tracker._current_position_entry_time(pos, 1)
        assert entered.isoformat() == "2026-04-01T10:00:00+00:00"
        assert entered.tzinfo == timezone.utc
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_v49_migration_adds_exit_decision_telemetry(tmp_path):
    path = tmp_path / "migration.db"
    db = Database(str(path))
    await db.connect()
    await db.execute("DROP TABLE exit_decisions")
    await db.execute("UPDATE schema_version SET version = 49")
    await db.commit()
    await db.close()

    upgraded = Database(str(path))
    await upgraded.connect()
    try:
        version = await upgraded.fetchone("SELECT version FROM schema_version")
        columns = await upgraded.fetchall("PRAGMA table_info(exit_decisions)")
        assert version["version"] == 50
        assert {row["name"] for row in columns} >= {
            "policy_action", "net_pnl_pct", "estimated_fees", "peak_pnl_pct"
        }
    finally:
        await upgraded.close()
