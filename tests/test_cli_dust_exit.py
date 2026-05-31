"""Tests for the `dust-exit` CLI command."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from auramaur.cli import main
from auramaur.db.database import Database
from auramaur.exchange.models import ExitReason


class _FakeMarket:
    outcome_yes_price = 0.2
    outcome_no_price = 0.8


class _FakeDiscovery:
    async def get_market(self, market_id):
        return _FakeMarket()


# Two dust positions, BOTH stored with the stale label "sports". One is a real
# sports market; the other is a politics market that only *looks* sports in the
# stored column. --category sports must use the live classifier and keep only
# the real one.
_SEED = [
    ("spt1", "Arkansas Razorbacks vs. Arizona Wildcats", ""),
    ("pol1", "Will the Republicans win the Mississippi Senate race in 2026?",
     "This market will resolve according to the winner of the election."),
]


class _FakeBot:
    """Stands in for AuramaurBot: seeds an in-memory DB inside the command's
    own event loop (so the aiosqlite connection is bound to the right loop)."""

    def __init__(self, *args, **kwargs):
        self._components = {}

    async def _init_components(self):
        db = Database(":memory:")
        await db.connect()
        for mid, q, desc in _SEED:
            await db.execute(
                "INSERT INTO markets (id, question, description, category, last_updated, "
                "outcome_yes_price, outcome_no_price) VALUES (?,?,?,?,datetime('now'),?,?)",
                (mid, q, desc, "sports", 0.2, 0.8),
            )
            for is_paper in (0, 1):  # seed both modes so the test is mode-agnostic
                await db.execute(
                    "INSERT INTO portfolio (market_id, exchange, side, size, avg_price, "
                    "current_price, category, token, token_id, is_paper) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (mid, "polymarket", "BUY", 10.0, 0.3, 0.2, "sports", "YES", f"tok-{mid}", is_paper),
                )
        await db.commit()
        self._components = {
            "db": db,
            "discoveries": {"polymarket": _FakeDiscovery()},
            "exchanges": {"polymarket": object()},
            "alerts": object(),
        }

    async def shutdown(self):
        await self._components["db"].close()


@pytest.fixture(autouse=True)
def _restore_event_loop():
    """The command invokes ``asyncio.run()``, which closes the process-global
    event loop. Other test modules still use the legacy
    ``asyncio.get_event_loop()`` pattern, so hand them a fresh open loop after
    each test here to avoid cross-module contamination."""
    yield
    asyncio.set_event_loop(asyncio.new_event_loop())


def test_dust_cleanup_reason_exists():
    assert ExitReason.DUST_CLEANUP.value == "DUST_CLEANUP"


def test_dust_exit_registered_with_help():
    result = CliRunner().invoke(main, ["dust-exit", "--help"])
    assert result.exit_code == 0
    assert "dust" in result.output.lower()
    assert "--category" in result.output


def test_category_filter_uses_live_classification():
    """--category sports keeps the real sports market and drops the politics
    market that merely has a stale 'sports' label stored."""
    with patch("auramaur.cli.AuramaurBot", _FakeBot):
        result = CliRunner().invoke(main, ["dust-exit", "--category", "sports", "--max-notional", "100"])
    assert result.exit_code == 0, result.output
    assert "spt1" in result.output       # real sports market kept
    assert "pol1" not in result.output   # stale-labelled politics market excluded


def test_no_category_filter_includes_both():
    with patch("auramaur.cli.AuramaurBot", _FakeBot):
        result = CliRunner().invoke(main, ["dust-exit", "--max-notional", "100"])
    assert result.exit_code == 0, result.output
    assert "spt1" in result.output
    assert "pol1" in result.output


def test_dust_exit_refuses_when_bot_is_running():
    """If the primary DB is locked (bot running), the command must refuse
    gracefully rather than crash or attach to a stray DB slot — closing
    positions alongside the live bot risks double-selling."""
    with patch("auramaur.cli.AuramaurBot") as MockBot:
        inst = MockBot.return_value
        inst._init_components = AsyncMock(
            side_effect=RuntimeError(
                "Database auramaur.db is already locked by another instance"
            )
        )
        result = CliRunner().invoke(main, ["dust-exit"])

    assert result.exit_code == 0
    out = result.output.lower()
    assert "already running" in out or "locked" in out
    # Must never reach execution when it can't safely attach.
    inst._execute_poly_exit.assert_not_called()
