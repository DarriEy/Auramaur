"""Read-only SQLite access for the web dashboard.

Deliberately NOT ``auramaur.db.Database``: its ``connect()`` applies schema
DDL and write PRAGMAs, and the dashboard must be structurally unable to write
to the trading database. This wrapper opens the file with the ``mode=ro`` URI
(the same approach as ``auramaur.health`` and the agent MCP server) and
exposes just the ``fetchone``/``fetchall`` surface the cockpit query layer
needs, so ``monitoring.cockpit.gather_state`` runs unchanged on top of it.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite

from auramaur.runtime import db_path as runtime_db_path


class ReadOnlyDatabase:
    """Async SQLite handle that can only read."""

    def __init__(self, db_path: str | None = None):
        self.db_path = str(runtime_db_path()) if db_path is None else db_path
        self._db: aiosqlite.Connection | None = None

    @property
    def connected(self) -> bool:
        return self._db is not None

    async def connect(self) -> None:
        if self._db is not None:
            return
        # as_uri() yields file:///C:/... on Windows and file:///app/... on
        # Linux — both forms SQLite accepts; string-built URIs with
        # backslashes are not portable.
        uri = f"{Path(self.db_path).resolve().as_uri()}?mode=ro"
        self._db = await aiosqlite.connect(uri, uri=True)
        self._db.row_factory = aiosqlite.Row
        # WAL readers can hit a writer's checkpoint lock; wait briefly instead
        # of erroring (the bot holds busy_timeout=30000 on its side).
        await self._db.execute("PRAGMA busy_timeout=5000")
        # Belt on top of mode=ro: even a coding mistake in this process cannot
        # turn a query into a write.
        await self._db.execute("PRAGMA query_only=ON")

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._db

    async def fetchone(self, sql: str, params: tuple = ()) -> aiosqlite.Row | None:
        cursor = await self.db.execute(sql, params)
        return await cursor.fetchone()

    async def fetchall(self, sql: str, params: tuple = ()) -> list[aiosqlite.Row]:
        cursor = await self.db.execute(sql, params)
        return await cursor.fetchall()
