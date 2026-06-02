"""Unified cockpit view — regression guard.

`cockpit` and `status` now share a single state gatherer (`gather_state`) and a
single renderer (`make_layout`). Before unification the cockpit summed the
legacy `trades.pnl` mirror while the dashboard computed authoritative P&L, so
the two commands could disagree about the same account. These tests pin the
cockpit to the authoritative computation (cost-basis realized + unrealized
mark-to-market) and prove both render paths build.
"""

import asyncio
from types import SimpleNamespace

import pytest

from auramaur.db.database import Database
from auramaur.monitoring import cockpit as ck


def _fake_settings() -> SimpleNamespace:
    """Minimal stand-in covering every attribute gather_state touches.

    Venues disabled (no network), log file absent (empty activity feed),
    paper mode.
    """
    return SimpleNamespace(
        is_live=False,
        transfers_armed=False,
        kill_switch_active=False,
        execution=SimpleNamespace(paper_initial_balance=100.0),
        logging=SimpleNamespace(file="/nonexistent/auramaur.log"),
        kalshi=SimpleNamespace(enabled=False),
        kraken=SimpleNamespace(enabled=False),
    )


async def _seed(db: Database) -> None:
    await db.execute(
        "INSERT INTO markets (id, question, last_updated) VALUES (?,?,datetime('now'))",
        ("m1", "Will it rain tomorrow?"),
    )
    # Paper position: 10 units, bought at 0.30, now marked at 0.50.
    await db.execute(
        "INSERT INTO portfolio (market_id, exchange, side, size, avg_price, "
        "current_price, token, token_id, is_paper) VALUES (?,?,?,?,?,?,?,?,1)",
        ("m1", "polymarket", "BUY", 10.0, 0.30, 0.50, "YES", "tok-m1"),
    )
    # Authoritative basis: realized 2.00, avg_cost 0.30.
    await db.execute(
        "INSERT INTO cost_basis (market_id, token, size, avg_cost, total_cost, "
        "realized_pnl, is_paper) VALUES (?,?,?,?,?,?,1)",
        ("m1", "YES", 10.0, 0.30, 3.0, 2.0),
    )
    # A divergent trades.pnl mirror — the value the old cockpit would have summed.
    # If gather_state ever regresses to trusting it, total_pnl jumps to 999.
    await db.execute(
        "INSERT INTO trades (market_id, side, size, price, pnl, is_paper) "
        "VALUES (?,?,?,?,?,1)",
        ("m1", "BUY", 10.0, 0.30, 999.0),
    )
    await db.commit()


def test_cockpit_pnl_is_authoritative_not_trades_mirror():
    async def run():
        db = Database(":memory:")
        await db.connect()
        await _seed(db)
        state = await ck.gather_state(db, _fake_settings(), cache=None)

        # realized 2.00 + unrealized (0.50 - 0.30) * 10 = 4.00 — NOT 999.
        assert state["total_pnl"] == pytest.approx(4.0)
        assert state["position_count"] == 1
        # Balance = paper float + authoritative P&L.
        assert state["balance"] == pytest.approx(104.0)

    asyncio.run(run())


def test_both_render_paths_build():
    async def run():
        db = Database(":memory:")
        await db.connect()
        await _seed(db)
        state = await ck.gather_state(db, _fake_settings(), cache=None)

        # Live full-screen Layout and one-shot compact Group must both build.
        from rich.layout import Layout
        from rich.console import Group

        assert isinstance(ck.make_layout(state), Layout)
        assert isinstance(ck.make_layout(state, compact=True), Group)

    asyncio.run(run())
