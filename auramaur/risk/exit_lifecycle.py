"""Durable, position-scoped exit disposition state."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

import structlog

log = structlog.get_logger()


class ExitState(StrEnum):
    REQUESTED = "REQUESTED"
    ORDER_WORKING = "ORDER_WORKING"
    RETRYABLE = "RETRYABLE"
    UNSALEABLE_DUST = "UNSALEABLE_DUST"
    UNMARKABLE = "UNMARKABLE"
    CLOSED = "CLOSED"


def position_key(pos, exchange: str | None = None) -> tuple[str, str, str, int]:
    token = getattr(pos, "token", "YES")
    return (
        exchange or getattr(pos, "exchange", "") or "",
        pos.market_id,
        getattr(token, "value", token),
        int(bool(getattr(pos, "is_paper", True))),
    )


async def record_exit_state(
    db, pos, state: ExitState, *, exchange: str | None = None,
    reason: str = "", error: str | None = None,
    next_retry_at: datetime | None = None, increment_attempt: bool = False,
) -> None:
    """Upsert the latest lifecycle state for one economic position."""
    venue, market_id, token, is_paper = position_key(pos, exchange)
    retry = next_retry_at.astimezone(timezone.utc).isoformat() if next_retry_at else None
    step = int(increment_attempt)
    try:
        await db.execute(
            """INSERT INTO exit_lifecycle
                   (exchange, market_id, token, is_paper, state, reason,
                    attempt_count, last_error, next_retry_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                 ON CONFLICT(exchange, market_id, token, is_paper) DO UPDATE SET
                   state=excluded.state, reason=excluded.reason,
                   attempt_count=exit_lifecycle.attempt_count + ?,
                   last_error=excluded.last_error, next_retry_at=excluded.next_retry_at,
                   updated_at=datetime('now')""",
            (venue, market_id, token, is_paper, state.value, reason,
             step, error, retry, step),
        )
        await db.commit()
    except Exception as exc:  # telemetry must never block an exit
        log.warning("exit.lifecycle_write_failed", exchange=venue,
                    market_id=market_id, state=state.value, error=str(exc))
