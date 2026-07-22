"""Uniform strategy heartbeats — every pillar cycle leaves a persistent trace.

2026-07-22: one sweep found FOUR strategies indistinguishable from dead —
two never logged their zero-entry cycles (platform_consensus,
cross_venue_arb), one ran 48 opaque cycles/day (settlement_arb), and a
stranded position sat in a book that never cycles at all. Per-pillar cycle
logs fix visibility one bug at a time; the heartbeat fixes the CLASS: the
task loops record every cycle (or its failure) here, so "no data" can only
ever mean "the task is not running" — and the dashboard can say which,
with the pillar's own cadence as the yardstick instead of a global
freshness threshold.

Read-only consumers (the web dashboard) read this table via their normal
``mode=ro`` connections; only the bot process writes it.
"""

from __future__ import annotations

import json

import structlog

log = structlog.get_logger()

_SCHEMA = """CREATE TABLE IF NOT EXISTS strategy_heartbeats (
    strategy TEXT PRIMARY KEY,
    last_beat_at TEXT NOT NULL DEFAULT (datetime('now')),
    status TEXT NOT NULL DEFAULT 'ok',
    entries INTEGER,
    cycles INTEGER NOT NULL DEFAULT 0,
    interval_seconds REAL,
    detail TEXT NOT NULL DEFAULT ''
)"""


async def beat(db, strategy: str, *, status: str = "ok",
               entries: int | None = None,
               interval_seconds: float | None = None,
               detail: dict | None = None) -> None:
    """Record one strategy cycle. NEVER raises — monitoring must not be able
    to kill the pillar it monitors."""
    try:
        if not getattr(db, "_heartbeat_schema_ready", False):
            async with db.transaction():
                await db.execute(_SCHEMA)
            db._heartbeat_schema_ready = True
        detail_json = ""
        if detail:
            try:
                detail_json = json.dumps(detail, default=str)[:2000]
            except Exception:
                detail_json = ""
        async with db.transaction():
            await db.execute(
                """INSERT INTO strategy_heartbeats
                       (strategy, last_beat_at, status, entries, cycles,
                        interval_seconds, detail)
                   VALUES (?, datetime('now'), ?, ?, 1, ?, ?)
                   ON CONFLICT(strategy) DO UPDATE SET
                       last_beat_at = datetime('now'),
                       status = excluded.status,
                       entries = excluded.entries,
                       cycles = strategy_heartbeats.cycles + 1,
                       interval_seconds = excluded.interval_seconds,
                       detail = excluded.detail""",
                (strategy, status, entries, interval_seconds, detail_json))
    except Exception as e:  # noqa: BLE001 — see docstring
        log.debug("heartbeat.write_error", strategy=strategy, error=str(e))


async def run_pillar_once(db, pillar, *,
                          interval_seconds: float | None = None) -> int:
    """Run one pillar cycle and heartbeat the outcome either way.

    On success records status=ok with the entry count and the pillar's
    optional ``last_cycle_detail`` dict (funnel counters). On exception
    records status=error with the message, then RE-RAISES so each task
    loop's own error handling still runs.
    """
    name = getattr(pillar, "name", type(pillar).__name__)
    try:
        entered = await pillar.run_once()
    except Exception as e:
        await beat(db, name, status="error",
                   interval_seconds=interval_seconds,
                   detail={"error": str(e)[:300]})
        raise
    await beat(db, name, status="ok",
               entries=int(entered) if isinstance(entered, (int, float)) else None,
               interval_seconds=interval_seconds,
               detail=getattr(pillar, "last_cycle_detail", None))
    return entered if isinstance(entered, int) else 0
