"""Tests for the `dust-exit` CLI command."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from auramaur.cli import main
from auramaur.exchange.models import ExitReason


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
    assert "--execute" in result.output
    assert "--max-notional" in result.output


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
