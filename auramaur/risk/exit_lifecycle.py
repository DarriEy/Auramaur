"""Durable, position-scoped exit disposition state.

The bot's in-memory suppression sets (``_exit_pending`` / ``_exit_failures``)
remain the authoritative retry gate; this table is the durable, operator-facing
record of where each economic position's exit stands and why. Rows are
advisory telemetry: a write failure — or a slow write — must never block or
delay an exit, so every write here is bounded by ``_WRITE_TIMEOUT_SECONDS``
and exception-contained. Never call these helpers from inside a
``db.transaction()`` span: ``execute()`` would silently join the span and an
outer rollback would discard the row while the caller logged success.
"""

from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple

import structlog

if TYPE_CHECKING:
    from auramaur.db.database import Database
    from auramaur.exchange.models import Position

log = structlog.get_logger()

# Caps how long the exit loop can stall on the shared serializer for a
# telemetry write. The exception containment below only covers RAISED errors;
# a wedged transaction() holder (2026-07-25: 45 minutes) blocks the lock
# acquisition without raising, and an unbounded await here would freeze every
# remaining position's exit on every venue.
_WRITE_TIMEOUT_SECONDS = 5.0


class ExitState(StrEnum):
    # Every member has a writer: ORDER_WORKING / RETRYABLE / UNSALEABLE_DUST /
    # CLOSED from the portfolio monitor's post-attempt write (dispositions come
    # from the executors — bot_exits.py), UNMARKABLE from check_exits' frozen-
    # mark branch, CLOSED also from the order monitor's terminal-fill path.
    # Add a state only together with its writer: a member nothing writes is a
    # coverage claim the doctor cannot honor.
    ORDER_WORKING = "ORDER_WORKING"
    RETRYABLE = "RETRYABLE"
    UNSALEABLE_DUST = "UNSALEABLE_DUST"
    UNMARKABLE = "UNMARKABLE"
    CLOSED = "CLOSED"


class ExitAttempt(NamedTuple):
    """An executor's verdict: did the sell go in, and if not, what class of no.

    ``disposition`` is set by the code that actually knows why the attempt
    failed (venue minimums, on-chain balances, book state live in the
    executors); the portfolio monitor must not re-derive it from ``pos.size``.
    ``detail`` is the machine-readable cause ("too_small", "no_bid", ...).
    """

    ok: bool
    disposition: ExitState | None = None
    detail: str = ""


def position_key(pos: Position, exchange: str) -> tuple[str, str, str, int]:
    """Canonical (exchange, market_id, token, is_paper) identity of a position.

    Attribute access is deliberately raw: a position object missing ``token``
    or ``is_paper`` must raise here, not be silently misfiled — the 12-day
    exit outage (#420) was a position model missing ``is_paper``, and a
    getattr default would have filed live rows under the paper key, exactly
    where live-mode diagnostics never look.
    """
    token = pos.token
    return (
        exchange or pos.exchange,
        pos.market_id,
        getattr(token, "value", str(token)),  # TokenType member or DB string
        int(bool(pos.is_paper)),
    )


async def record_exit_state(
    db: Database,
    key: tuple[str, str, str, int],
    state: ExitState,
    *,
    reason: str = "",
    error: str | None = None,
    retry_after_seconds: float | None = None,
    increment_attempt: bool = False,
) -> None:
    """Upsert the latest lifecycle state for one economic position.

    ``next_retry_at`` is computed SQL-side via ``datetime('now', ?)`` so the
    column stays lexically comparable with its siblings and with consumer-side
    ``datetime('now')`` (readiness.py documents what a T-format/space-format
    mix inside one table costs); ``datetime('now', NULL)`` is NULL, so passing
    ``retry_after_seconds=None`` clears the wall.

    A row found in CLOSED is a dead episode: the upsert restarts
    ``attempt_count`` and ``requested_at`` so a re-entered position doesn't
    inherit a prior position's counters.

    The single INSERT is atomic on its own under the autocommit connection
    (txn-migration classification B) — no commit call belongs here.
    """
    venue, market_id, token, is_paper = key
    step = int(increment_attempt)
    retry_mod = (
        f"+{int(retry_after_seconds)} seconds"
        if retry_after_seconds is not None else None
    )
    try:
        await asyncio.wait_for(
            db.execute(
                """INSERT INTO exit_lifecycle
                       (exchange, market_id, token, is_paper, state, reason,
                        attempt_count, last_error, next_retry_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now', ?))
                     ON CONFLICT(exchange, market_id, token, is_paper) DO UPDATE SET
                       state=excluded.state, reason=excluded.reason,
                       attempt_count=CASE WHEN exit_lifecycle.state = 'CLOSED'
                                          THEN excluded.attempt_count
                                          ELSE exit_lifecycle.attempt_count + ?
                                     END,
                       requested_at=CASE WHEN exit_lifecycle.state = 'CLOSED'
                                         THEN datetime('now')
                                         ELSE exit_lifecycle.requested_at
                                    END,
                       last_error=excluded.last_error,
                       next_retry_at=excluded.next_retry_at,
                       updated_at=datetime('now')""",
                (venue, market_id, token, is_paper, state.value, reason,
                 step, error, retry_mod, step),
            ),
            timeout=_WRITE_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # telemetry must never block or fail an exit
        log.warning("exit.lifecycle_write_failed", exchange=venue,
                    market_id=market_id, state=state.value, error=str(exc))


async def advance_terminal(
    db: Database,
    exchange: str,
    market_id: str,
    is_paper: int,
    *,
    filled: bool,
    status: str,
) -> None:
    """Advance a market's lifecycle rows when a resting SELL turns terminal.

    The order monitor only knows the market (its exit orders don't carry the
    position's token), so this mirrors ``_clear_exit_suppression`` and touches
    every token's row under the (exchange, market, mode). A fill CLOSES the
    episode; a cancel/expiry re-opens it as RETRYABLE with no wall — the
    suppression keys were just cleared, so the portfolio monitor re-attempts
    on its next tick and overwrites this with the real outcome.
    """
    new_state = ExitState.CLOSED if filled else ExitState.RETRYABLE
    try:
        await asyncio.wait_for(
            db.execute(
                """UPDATE exit_lifecycle
                      SET state=?, reason=?, next_retry_at=NULL,
                          updated_at=datetime('now')
                    WHERE exchange=? AND market_id=? AND is_paper=?
                      AND state != 'CLOSED'""",
                (new_state.value, status, exchange, market_id, is_paper),
            ),
            timeout=_WRITE_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        log.warning("exit.lifecycle_terminal_failed", exchange=exchange,
                    market_id=market_id, status=status, error=str(exc))
