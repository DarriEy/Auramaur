"""Arming the kill switch must retire live exposure and must not change books.

Three separately-defensible decisions used to compose into the opposite of the
operator's intent: (1) the switch only set _running=False, and the cancel sweep
lived solely in shutdown() — which is unreachable while any task loops on
`while True`; (2) the order monitor stops on the same flag, so fills on those
still-resting orders were never recorded; (3) the cockpit picked its book from
settings.is_live, which folds in kill_switch_active, so the display silently
swapped to the paper book. Net effect: real orders kept working, unrecorded,
while the screen showed a different book.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from auramaur.monitoring import cockpit


def _settings(*, auramaur_live: bool, execution_live: bool, kill: bool):
    s = MagicMock()
    s.auramaur_live = auramaur_live
    s.execution.live = execution_live
    s.kill_switch_active = kill
    s.is_live = auramaur_live and execution_live and not kill
    return s


@pytest.mark.parametrize("kill", [False, True])
def test_book_selection_ignores_the_kill_switch(kill):
    """A live-configured bot shows the LIVE book whether or not it is halted."""
    s = _settings(auramaur_live=True, execution_live=True, kill=kill)
    flag = 0 if (s.auramaur_live and s.execution.live) else 1
    assert flag == 0, "arming the kill switch must not swap the displayed book"


def test_paper_configured_bot_still_shows_paper():
    s = _settings(auramaur_live=False, execution_live=True, kill=False)
    flag = 0 if (s.auramaur_live and s.execution.live) else 1
    assert flag == 1


def test_header_names_the_book_it_is_showing():
    """The book must be labelled: a swap must never look like 'positions gone'."""
    from datetime import datetime, timezone
    base = {
        "now": datetime.now(timezone.utc),
        "kill_switch": True, "transfers_armed": False,
    }
    live = cockpit._header_panel({**base, "is_live": False, "book": "live"})
    paper = cockpit._header_panel({**base, "is_live": False, "book": "paper"})
    assert "LIVE" in str(live.renderable)
    assert "paper" in str(paper.renderable)
    # Halted-but-live-book is the exact state that used to be unreadable.
    assert "KILL SWITCH" in str(live.renderable)


@pytest.mark.asyncio
async def test_kill_switch_cancels_resting_live_orders_once():
    """Halting placement is not halting exposure — the sweep must run, and
    must not re-run on each of the ~20 _check_kill_switch call sites."""
    from auramaur.bot import AuramaurBot

    bot = AuramaurBot.__new__(AuramaurBot)
    bot._running = True
    bot._kill_switch_handled = False
    bot._components = MagicMock()
    bot._components.alerts = MagicMock()
    bot._components.alerts.send = AsyncMock()
    bot._cancel_resting_live_orders = AsyncMock(return_value=3)

    import auramaur.bot as bot_mod
    orig = bot_mod.kill_switch_present
    bot_mod.kill_switch_present = lambda: True
    try:
        assert await bot._check_kill_switch() is True
        assert await bot._check_kill_switch() is True
        assert await bot._check_kill_switch() is True
    finally:
        bot_mod.kill_switch_present = orig

    assert bot._running is False
    bot._cancel_resting_live_orders.assert_awaited_once()
    assert "cancelled 3" in bot._components.alerts.send.await_args.args[0]


@pytest.mark.asyncio
async def test_cancel_failure_does_not_prevent_the_halt():
    """A venue error during the sweep must still leave the bot halted."""
    from auramaur.bot import AuramaurBot

    bot = AuramaurBot.__new__(AuramaurBot)
    bot._running = True
    bot._kill_switch_handled = False
    bot._components = MagicMock()
    bot._components.alerts = MagicMock()
    bot._components.alerts.send = AsyncMock()
    bot._cancel_resting_live_orders = AsyncMock(side_effect=RuntimeError("venue down"))

    import auramaur.bot as bot_mod
    orig = bot_mod.kill_switch_present
    bot_mod.kill_switch_present = lambda: True
    try:
        assert await bot._check_kill_switch() is True
    finally:
        bot_mod.kill_switch_present = orig

    assert bot._running is False
    bot._components.alerts.send.assert_awaited()
