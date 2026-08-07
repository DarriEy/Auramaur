"""The portfolio monitor's exit loop — the block that actually sells.

Nothing drove ``AuramaurBot._task_portfolio_monitor``'s ``for pos, reason in
exit_list`` block, so from 2026-07-25 (fdf79fc) to 2026-08-06 it raised
``AttributeError: 'Position' object has no attribute 'is_paper'`` on the FIRST
position of every tick. The raise escaped to the tick-level ``except
Exception: log.debug(...)``, which abandoned the rest of the tick and left no
operator-visible trace: live prediction-market stop-losses, profit targets and
trailing stops were structurally dead for 12 days (last live SELL fill
2026-07-25T13:04:06Z; 29 live BUY fills and zero live SELL fills after it).

The exit *policy* was well covered (tests/test_exits.py) and the exit *key* was
covered by rebuilding the f-string in the test (test_exits.py's
``test_exit_suppression_expires_and_is_keyed_per_position``) — a reimplementation
cannot fail when the real loop does. These tests run the real loop instead: one
tick of ``_task_portfolio_monitor`` with injected components, asserting that an
exit is ATTEMPTED for a position ``check_exits`` flagged.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from structlog.testing import capture_logs

from auramaur.bot import AuramaurBot
from auramaur.components import Components
from auramaur.exchange.models import ExitReason, OrderSide, Position, TokenType
from auramaur.risk.exit_lifecycle import ExitAttempt


def _position(market_id: str = "M1", *, is_paper: bool,
              token: TokenType = TokenType.YES) -> Position:
    return Position(
        market_id=market_id, exchange="polymarket", side=OrderSide.BUY,
        size=20.0, avg_price=0.50, current_price=0.30, category="politics",
        token=token, token_id="tok", is_paper=is_paper,
    )


def _bot(exit_list: list, *, is_live: bool = True):
    """A bare bot wired for exactly one portfolio-monitor tick.

    ``check_exits`` clears ``_running`` so the ``while`` loop runs once, and
    ``portfolio_check_seconds`` is 0 so the trailing sleep returns immediately.
    """
    bot = AuramaurBot.__new__(AuramaurBot)
    bot._running = True
    bot._exit_pending = set()
    bot._exit_failures = {}
    bot._last_known_cash = 0.0
    bot.settings = SimpleNamespace(
        is_live=is_live,
        intervals=SimpleNamespace(portfolio_check_seconds=0,
                                  adaptive_enabled=False),
    )

    async def _check_exits(_settings, _discovery, exchange=None):
        bot._running = False  # one tick
        return list(exit_list)

    tracker = SimpleNamespace(check_exits=_check_exits, note_equity=AsyncMock())
    attempted: list[tuple] = []

    async def _execute_poly_exit(pos, reason, _discovery, _exchange, _alerts):
        attempted.append((pos, reason))
        return ExitAttempt(True)

    bot._execute_poly_exit = _execute_poly_exit
    bot._components = Components({
        "db": AsyncMock(),
        "syncers": [],
        "pnl_tracker": SimpleNamespace(get_total_pnl=AsyncMock(return_value=0.0)),
        "discoveries": {"polymarket": SimpleNamespace()},
        # No _live_pending to walk; the reserved-collateral sum skips it.
        "exchanges": {"polymarket": SimpleNamespace(_live_pending={})},
        "alerts": AsyncMock(),
        "risk_manager": SimpleNamespace(portfolio=tracker),
        "attributor": None,
        "reconciler": None,
    })
    return bot, attempted


async def _one_tick(bot) -> list[dict]:
    """Run a single monitor tick, returning the structlog events it emitted."""
    with capture_logs() as events:
        # Bounded: if a defect aborts the tick before check_exits can clear
        # _running, this fails loudly instead of hanging CI.
        await asyncio.wait_for(bot._task_portfolio_monitor(), timeout=10)
    return events


@pytest.mark.asyncio
async def test_flagged_position_reaches_the_exit_executor():
    """The regression itself: a flagged live position must be sold.

    Under the bug the loop raised on ``pos.is_paper`` while building the exit
    key — before any executor was called — so this asserts the *execution path
    was reached*, not merely that nothing propagated.
    """
    pos = _position(is_paper=False)
    bot, attempted = _bot([(pos, ExitReason.STOP_LOSS)])

    events = await _one_tick(bot)

    assert attempted == [(pos, ExitReason.STOP_LOSS)], (
        "no exit was attempted for a position check_exits flagged")
    # And the accepted exit is suppressed under its own per-position key.
    assert bot._exit_pending == {"exit:polymarket:M1:YES:0"}
    assert bot._exit_failures == {}
    # The tick completed: nothing fell through to the tick-level handler.
    assert not [e for e in events if e["event"] == "portfolio_monitor_error"]


@pytest.mark.asyncio
async def test_paper_and_live_legs_of_one_market_get_distinct_keys():
    """Why the field exists at all.

    ``check_exits`` passes ``is_paper=None`` whenever ``settings.is_live`` is
    not a bool, which leaves ``get_positions`` UNSCOPED — the list can hold
    both books. A caller-side mode flag would collapse the two legs of one
    (market, token) onto a single key, so the second leg would read as already
    pending and never sell: the same dust-pinning failure the per-position key
    was introduced to fix on 2026-07-25.
    """
    live = _position(is_paper=False)
    paper = _position(is_paper=True)
    bot, attempted = _bot([(live, ExitReason.STOP_LOSS),
                           (paper, ExitReason.STOP_LOSS)])

    await _one_tick(bot)

    assert [p for p, _ in attempted] == [live, paper], (
        "both books' legs must be attempted, not just the first")
    assert bot._exit_pending == {"exit:polymarket:M1:YES:0",
                                 "exit:polymarket:M1:YES:1"}


@pytest.mark.asyncio
async def test_a_broken_position_surfaces_and_does_not_abort_the_rest():
    """The swallowing ``except`` must not hide this class of defect again.

    Two properties, both absent on 2026-07-25: an unexpected AttributeError in
    the loop body is logged at a level ``readiness._ERROR_LEVELS`` scores (so
    12 days of silence becomes a failed cycle-health criterion), and it costs
    only its own position instead of the whole tick.
    """
    from auramaur.monitoring.readiness import _ERROR_LEVELS

    # A position-shaped object missing is_paper: the exact 2026-07-25 shape.
    broken = SimpleNamespace(market_id="M0", token=TokenType.YES,
                             unrealized_pnl=-4.20)
    healthy = _position("M2", is_paper=False)
    bot, attempted = _bot([(broken, ExitReason.STOP_LOSS),
                           (healthy, ExitReason.STOP_LOSS)])

    events = await _one_tick(bot)

    assert [p for p, _ in attempted] == [healthy], (
        "one bad position must not abort the remaining exits")
    loop_errors = [e for e in events if e["event"] == "exit.loop_error"]
    assert len(loop_errors) == 1, events
    assert loop_errors[0]["log_level"] in _ERROR_LEVELS, (
        "a swallowed defect that readiness cannot see is a silent outage")
    assert loop_errors[0]["error_type"] == "AttributeError"
    assert loop_errors[0]["market_id"] == "M0"
    # The tick itself survived, so the tick-level handler never fired.
    assert not [e for e in events if e["event"] == "portfolio_monitor_error"]


@pytest.mark.asyncio
async def test_tick_level_failures_are_no_longer_debug_only():
    """A tick that dies takes every venue's exits with it — that is an error.

    ``log.debug`` is the reason the outage was invisible: readiness parses logs
    by level and never reads debug.
    """
    from auramaur.monitoring.readiness import _ERROR_LEVELS

    bot, attempted = _bot([(_position(is_paper=False), ExitReason.STOP_LOSS)])

    async def _boom(_positions):
        bot._running = False  # one tick
        raise TypeError("total_pnl exploded")

    bot._components["pnl_tracker"] = SimpleNamespace(get_total_pnl=_boom)

    events = await _one_tick(bot)

    aborted = [e for e in events if e["event"] == "portfolio_monitor_error"]
    assert aborted, "a dead tick must say so"
    assert aborted[0]["log_level"] in _ERROR_LEVELS
    assert aborted[0]["error_type"] == "TypeError"
    # And the cost of a dead tick is exactly what went unseen for 12 days.
    assert attempted == []


@pytest.mark.asyncio
async def test_get_positions_carries_the_book_it_read(tmp_path):
    """The model field is only useful if the DB read populates it.

    ``SELECT p.*`` already returned ``is_paper``; the model dropped it. With an
    unscoped read (settings.is_live not a bool) both books come back in one
    list, which is precisely when the flag has to be per-row.
    """
    from auramaur.db.database import Database
    from auramaur.risk.portfolio import PortfolioTracker

    db = Database(str(tmp_path / "book.db"))
    await db.connect()
    try:
        await db.execute(
            """INSERT INTO portfolio
               (market_id, exchange, side, size, avg_price, current_price,
                category, token, token_id, is_paper)
               VALUES
               ('M1','polymarket','BUY',10,0.50,0.30,'politics','YES','t',0),
               ('M1','polymarket','BUY',10,0.50,0.30,'politics','YES','t',1)""")
        await db.commit()

        tracker = PortfolioTracker(db, SimpleNamespace(is_live=None))
        positions = await tracker.get_positions(exchange="polymarket")
        assert sorted(p.is_paper for p in positions) == [False, True]

        scoped = await tracker.get_positions(exchange="polymarket", is_paper=False)
        assert [p.is_paper for p in scoped] == [False]
    finally:
        await db.close()
