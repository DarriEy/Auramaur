"""Single state-gathering loop shared by every dashboard client.

One broker per process owns the refresh cadence, so N open browser tabs cost
the same DB/venue load as one. It also owns degraded-mode behavior: the
service starts — and stays up — with a missing or schemaless database,
reporting the reason in its envelope instead of 500ing, and recovers on its
own the moment the bot creates real state. The dashboard's job is to make the
system legible; that must extend to the dashboard's own failure modes.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from auramaur.monitoring import cockpit
from auramaur.web import queries
from auramaur.web.db import ReadOnlyDatabase
from auramaur.web.serialize import serialize_state
from config.settings import Settings


class StateBroker:
    def __init__(
        self,
        db: ReadOnlyDatabase,
        settings: Settings,
        refresh_seconds: float = 2.0,
    ):
        self.db = db
        self.settings = settings
        self.refresh_seconds = refresh_seconds
        # last GOOD {"paper": ..., "live": ...} — kept through errors
        self.latest: dict | None = None
        self.error: str | None = None
        self.updated_at: str | None = None
        # Primed bal_ts (re-stamped every cycle): gather_state must never see
        # a stale cache and fetch venue balances itself — this process holds
        # no venue credentials. Balances come from the venue_balances table
        # the bot's recorder maintains.
        self._cache: dict = {"bal_ts": time.time(), "venues": {}}
        self._first_tick = asyncio.Event()
        # Avoid publishing the schema-only instant while a newly appearing DB
        # is still receiving its first transaction.
        self._initial_empty_since: float | None = None
        self._tasks: list[asyncio.Task] = []

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._tasks = [asyncio.create_task(self._state_loop())]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.db.close()

    async def wait_first(self, timeout: float = 15.0) -> None:
        """Block until the first gather attempt has finished (ok or not)."""
        try:
            await asyncio.wait_for(self._first_tick.wait(), timeout)
        except TimeoutError:
            pass

    # -- the envelope every endpoint serves ---------------------------------

    def envelope(self) -> dict:
        # "Armed" mode (the two deliberate gates), independent of the kill
        # switch — a halt is its own banner, not a mode change.
        armed_live = self.settings.auramaur_live and self.settings.execution.live
        return {
            "ok": self.error is None and self.latest is not None,
            "error": self.error,
            "updated_at": self.updated_at,
            "bot_mode": "live" if armed_live else "paper",
            "books": self.latest,
        }

    # -- loops --------------------------------------------------------------

    async def _gather_book(self, book: str) -> dict:
        state = serialize_state(
            await cockpit.gather_state(self.db, self.settings, self._cache, book=book)
        )
        flag = 0 if book == "live" else 1
        state["strategies"] = await queries.strategy_breakdown(
            self.db, flag, self.settings)
        state["heartbeats"] = await queries.strategy_heartbeats(self.db)
        state["categories"] = await queries.category_exposure(self.db, flag)
        if book == "paper":
            # Kraken's directional paper book lives in its own table and is
            # invisible to the portfolio query — surface it explicitly.
            state["kraken_paper"] = await queries.kraken_paper_positions(self.db)
        return state

    async def _state_loop(self) -> None:
        # The connection is opened per CYCLE and closed after, never held.
        # A long-lived mode=ro connection to the WAL database held a
        # persistent read lock on the main file that STARVED the bot's
        # writers (observed 2026-07-20: /proc/locks showed this process
        # pinning SQLite's shared-lock byte range while every position-sync
        # cycle failed "database is locked"). Read-only must also mean
        # contention-free: transient connections keep every lock as brief
        # as the queries themselves.
        while True:
            try:
                # Re-stamp so gather_state's 20s staleness check can never
                # trigger an inline venue fetch from this credential-less
                # process.
                self._cache["bal_ts"] = time.time()
                await self.db.connect()
                books = {
                    "paper": await self._gather_book("paper"),
                    "live": await self._gather_book("live"),
                }
                # Venue cash is book-independent: recorded by the bot,
                # served on both views with its age.
                venues = await queries.venue_balances(self.db)
                reconciliation = await queries.venue_reconciliation(self.db)
                ibkr_books = await queries.ibkr_paper_books(self.db)
                local_llm = await queries.local_llm_stats(self.db)
                intel_eval = await queries.intelligence_eval_summary(self.db)
                performance = await queries.performance_history(self.db)
                for state in books.values():
                    state["venues"] = venues
                    state["reconciliation"] = reconciliation
                    state["ibkr_books"] = ibkr_books
                    state["local_llm"] = local_llm
                    state["intelligence_eval"] = intel_eval
                    state["performance_history"] = performance
                initial_empty = all(
                    state["position_count"] == 0 and state["trade_count"] == 0
                    for state in books.values())
                if self.latest is None and initial_empty:
                    now_mono = time.monotonic()
                    if self._initial_empty_since is None:
                        self._initial_empty_since = now_mono
                    initializing = now_mono - self._initial_empty_since < 1.0
                else:
                    initializing = False
                if initializing:
                    self.error = (
                        "database schema is ready; waiting for first committed state")
                else:
                    self.latest = books
                    self.error = None
                    self.updated_at = datetime.now(timezone.utc).isoformat()
            except Exception as exc:
                self.error = self._describe(exc)
            finally:
                # Always drop the handle — also covers a DB that appears,
                # is replaced, or was mid-creation on the first attempt.
                await self.db.close()
            self._first_tick.set()
            await asyncio.sleep(self.refresh_seconds)

    def _describe(self, exc: Exception) -> str:
        msg = str(exc)
        if "unable to open database" in msg:
            return (
                f"database not found or unreadable at {self.db.db_path} — "
                "start the bot once to create it, or set AURAMAUR_DB_PATH"
            )
        if "no such table" in msg:
            return (
                f"database at {self.db.db_path} has no bot schema ({msg}) — "
                "point AURAMAUR_DB_PATH at the bot's real auramaur.db"
            )
        return f"{type(exc).__name__}: {msg}"
