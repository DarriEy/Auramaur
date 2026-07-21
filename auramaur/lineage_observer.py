"""Best-effort lineage writer batching onto the shared database connection.

Phase 3 of docs/plans/db-contention-plan.md: the observer used to open a
SECOND full read-write ``Database`` on the same file — the last extra
in-process writer. It now runs on the already-connected shared ``Database``;
its queue still decouples producers (they never write or wait for lineage),
and every batch runs inside ``Database.transaction()`` so it is serialized
against the other pillar tasks and atomic (a mid-batch failure leaves zero
partial rows).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

from auramaur.db.database import Database
from auramaur.information_graduation import InformationGraduation

log = structlog.get_logger()


@dataclass(frozen=True)
class LineageEvent:
    kind: str
    payload: dict[str, Any]


class LineageObserver:
    """Owns a queue (not a connection); producers never write or wait for lineage."""

    def __init__(self, db: Database, ladder: InformationGraduation) -> None:
        self.db = db
        self.ladder = ladder
        self._queue: asyncio.Queue[LineageEvent | None] = asyncio.Queue()
        self._closed = False
        self._task = asyncio.create_task(self._run())

    @classmethod
    async def create(cls, db: Database, **graduation) -> LineageObserver:
        """Build an observer on the already-connected shared ``Database``.

        The observer does NOT own the connection: ``close()`` drains the queue
        and stops the worker but leaves the database open for its owner
        (the component registry's ``db`` entry) to close afterwards.
        """
        return cls(db, InformationGraduation(db, **graduation))

    def ingestion(self, **payload: Any) -> None:
        self._queue.put_nowait(LineageEvent("ingestion", payload))

    def forecast(self, **payload: Any) -> None:
        self._queue.put_nowait(LineageEvent("forecast", payload))

    def resolution(self, market_id: str, outcome: bool) -> None:
        self._queue.put_nowait(LineageEvent(
            "resolution", {"market_id": market_id, "outcome": outcome},
        ))

    async def flush(self) -> None:
        await self._queue.join()

    async def close(self) -> None:
        """Drain the queue and stop the worker.

        Does NOT close the shared ``Database`` (see :meth:`create`). Idempotent:
        bot shutdown reaches it twice — via the component registry (ordered
        BEFORE the database closes) and again via ``Aggregator.close()``.
        """
        if self._closed:
            return
        self._closed = True
        await self.flush()
        await self._queue.put(None)
        await self._task

    async def _run(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                if event is None:
                    return
                await getattr(self, f"_write_{event.kind}")(**event.payload)
            except Exception as exc:
                # transaction() already rolled back this batch; a bare
                # rollback() here would run on the SHARED connection and could
                # discard another task's uncommitted writes.
                log.warning("lineage_observer.write_failed", kind=event.kind if event else None,
                            error=str(exc)[:200])
            finally:
                self._queue.task_done()

    async def _write_ingestion(self, *, run_id: str, query: str, category: str,
                               market_id: str | None, started_at: str,
                               observed_at: str, fetch_rows: list[tuple],
                               raw_items: int, items: list[dict],
                               active_sources: list[dict]) -> None:
        async with self.db.transaction():
            await self.db.execute(
                "INSERT INTO ingestion_runs "
                "(id,query,category,market_id,started_at,completed_at,status,active_sources,raw_items,unique_items) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run_id, query, category, market_id, started_at, observed_at,
                 "partial" if any(row[2] == "error" for row in fetch_rows) else "ok",
                 len(active_sources), raw_items,
                 sum(i["rank_position"] is not None for i in items)),
            )
            if fetch_rows:
                await self.db.executemany(
                    "INSERT OR REPLACE INTO source_fetches "
                    "(run_id,source,status,item_count,latency_ms,error,observed_at,information_mode) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    fetch_rows,
                )
            if items:
                await self.db.executemany(
                    "INSERT OR REPLACE INTO evidence_observations "
                    "(run_id,item_id,source,title,url,content_hash,excerpt,published_at,observed_at,"
                    "timestamp_quality,relevance_score,rank_position,market_id,information_mode) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [tuple(item[k] for k in (
                        "run_id", "item_id", "source", "title", "url", "content_hash",
                        "excerpt", "published_at", "observed_at", "timestamp_quality",
                        "relevance_score", "rank_position", "market_id", "information_mode",
                    )) for item in items],
                )
            for source in active_sources:
                if (source["mode"] != "shadow" or not source["had_items"] or not market_id
                        or source["market_price"] is None):
                    continue
                sid = await self.ladder.register(
                    source["name"], category, source["horizon"], source["event_type"],
                    commit=False,
                )
                await self.ladder.assign(
                    sid, market_id, datetime.fromisoformat(observed_at),
                    source["market_price"], commit=False,
                )

    async def _write_forecast(self, **p: Any) -> None:
        fingerprint = hashlib.sha256(json.dumps(
            p.pop("config"), sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()[:16]
        async with self.db.transaction():
            await self.db.execute(
                """INSERT INTO forecast_snapshots
                   (market_id,exchange,category,forecast_purpose,raw_probability,
                    calibrated_probability,market_yes_price,market_no_price,observed_at,
                    evidence_run_ids,model,strategy_source,config_fingerprint)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (p["market_id"], p["exchange"], p["category"], "analysis", p["raw_probability"],
                 p["calibrated_probability"], p["market_yes_price"], p["market_no_price"],
                 p["observed_at"], json.dumps(p["evidence_run_ids"]), p["model"],
                 p["strategy_source"], fingerprint),
            )

    async def _write_resolution(self, market_id: str, outcome: bool) -> None:
        actual = int(outcome)
        async with self.db.transaction():
            await self.db.execute(
                "UPDATE forecast_snapshots SET actual_outcome=?,resolved_at=datetime('now') "
                "WHERE market_id=? AND actual_outcome IS NULL", (actual, market_id),
            )
            trials = await self.db.fetchall(
                "SELECT id FROM information_trials WHERE market_id=? AND resolved_outcome IS NULL",
                (market_id,),
            )
            for trial in trials:
                await self.ladder.resolve(trial["id"], outcome, commit=False)
