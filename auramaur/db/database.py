"""SQLite database manager with async support."""

from __future__ import annotations

import aiosqlite
import structlog

from auramaur.db.models import SCHEMA_VERSION, TABLES

log = structlog.get_logger()


class Database:
    def __init__(self, db_path: str = "auramaur.db"):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=OFF")
        await self._init_schema()
        log.info("database.connected", path=self.db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def _init_schema(self) -> None:
        await self._db.executescript(TABLES)
        # Check/set schema version
        cursor = await self._db.execute("SELECT version FROM schema_version LIMIT 1")
        row = await cursor.fetchone()
        if row is None:
            await self._db.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
        else:
            current_version = row[0] if isinstance(row[0], int) else row["version"]
            if current_version < SCHEMA_VERSION:
                await self._run_migrations(current_version)
        await self._db.commit()

    async def _run_migrations(self, from_version: int) -> None:
        """Run all pending migrations sequentially."""
        if from_version < 2:
            await self._migrate_v1_to_v2()
        if from_version < 3:
            await self._migrate_v2_to_v3()
        if from_version < 4:
            await self._migrate_v3_to_v4()
        if from_version < 5:
            await self._migrate_v4_to_v5()
        if from_version < 6:
            await self._migrate_v5_to_v6()
        if from_version < 7:
            await self._migrate_v6_to_v7()
        if from_version < 8:
            await self._migrate_v7_to_v8()
        if from_version < 9:
            await self._migrate_v8_to_v9()

    async def _migrate_v1_to_v2(self) -> None:
        """Add category to calibration, add new tables."""
        # Add category column to calibration (SQLite ALTER TABLE ADD COLUMN)
        try:
            await self._db.execute(
                "ALTER TABLE calibration ADD COLUMN category TEXT DEFAULT ''"
            )
        except Exception:
            # Column may already exist if tables were recreated
            pass
        await self._db.execute("UPDATE schema_version SET version = 2")
        await self._db.commit()
        log.info("database.migrated", from_version=1, to_version=2)

    async def _migrate_v2_to_v3(self) -> None:
        """Add exchange and ticker columns for multi-exchange support."""
        alterations = [
            ("markets", "exchange TEXT DEFAULT 'polymarket'"),
            ("markets", "ticker TEXT DEFAULT ''"),
            ("signals", "exchange TEXT DEFAULT 'polymarket'"),
            ("trades", "exchange TEXT DEFAULT 'polymarket'"),
            ("portfolio", "exchange TEXT DEFAULT 'polymarket'"),
            ("price_history", "exchange TEXT DEFAULT 'polymarket'"),
        ]
        for table, column_def in alterations:
            try:
                await self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")
            except Exception:
                pass  # Column may already exist
        # Relax condition_id NOT NULL → already handled by new CREATE TABLE
        await self._db.execute("UPDATE schema_version SET version = 3")
        await self._db.commit()
        log.info("database.migrated", from_version=2, to_version=3)

    async def _migrate_v3_to_v4(self) -> None:
        """Add fills and cost_basis tables for broker layer."""
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                token_id TEXT DEFAULT '',
                side TEXT NOT NULL,
                token TEXT NOT NULL DEFAULT 'YES',
                size REAL NOT NULL,
                price REAL NOT NULL,
                fee REAL DEFAULT 0,
                is_paper INTEGER NOT NULL DEFAULT 1,
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS cost_basis (
                market_id TEXT PRIMARY KEY,
                token TEXT NOT NULL DEFAULT 'YES',
                token_id TEXT DEFAULT '',
                size REAL NOT NULL,
                avg_cost REAL NOT NULL,
                total_cost REAL NOT NULL,
                realized_pnl REAL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_fills_market ON fills(market_id);
            CREATE INDEX IF NOT EXISTS idx_fills_order ON fills(order_id);
        """)
        await self._db.execute("UPDATE schema_version SET version = 4")
        await self._db.commit()
        log.info("database.migrated", from_version=3, to_version=4)

    async def _migrate_v4_to_v5(self) -> None:
        """Add ensemble_predictions table for multi-LLM ensemble tracking."""
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS ensemble_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                model TEXT NOT NULL,
                category TEXT DEFAULT '',
                probability REAL NOT NULL,
                actual_outcome INTEGER,
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_ensemble_model ON ensemble_predictions(model);
            CREATE INDEX IF NOT EXISTS idx_ensemble_market ON ensemble_predictions(market_id);
        """)
        await self._db.execute("UPDATE schema_version SET version = 5")
        await self._db.commit()
        log.info("database.migrated", from_version=4, to_version=5)

    async def _migrate_v5_to_v6(self) -> None:
        """Add token and token_id columns to portfolio for correct exit pricing."""
        alterations = [
            ("portfolio", "token TEXT NOT NULL DEFAULT 'YES'"),
            ("portfolio", "token_id TEXT DEFAULT ''"),
        ]
        for table, column_def in alterations:
            try:
                await self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")
            except Exception:
                pass  # Column may already exist

        # Backfill from cost_basis where available
        try:
            await self._db.execute("""
                UPDATE portfolio SET
                    token = COALESCE((SELECT token FROM cost_basis WHERE cost_basis.market_id = portfolio.market_id), 'YES'),
                    token_id = COALESCE((SELECT token_id FROM cost_basis WHERE cost_basis.market_id = portfolio.market_id), '')
            """)
        except Exception:
            pass

        await self._db.execute("UPDATE schema_version SET version = 6")
        await self._db.commit()
        log.info("database.migrated", from_version=5, to_version=6)

    async def _migrate_v6_to_v7(self) -> None:
        """Add market_price column to nlp_cache for price-move invalidation."""
        try:
            await self._db.execute(
                "ALTER TABLE nlp_cache ADD COLUMN market_price REAL DEFAULT 0"
            )
        except Exception:
            pass  # Column may already exist
        await self._db.execute("UPDATE schema_version SET version = 7")
        await self._db.commit()
        log.info("database.migrated", from_version=6, to_version=7)

    async def _migrate_v7_to_v8(self) -> None:
        """Add redemptions table for on-chain CTF redemption tracking."""
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS redemptions (
                condition_id TEXT PRIMARY KEY,
                asset_id TEXT DEFAULT '',
                title TEXT DEFAULT '',
                neg_risk INTEGER DEFAULT 0,
                size REAL NOT NULL,
                expected_payout REAL NOT NULL,
                safe_nonce INTEGER,
                tx_hash TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                submitted_at TEXT,
                confirmed_at TEXT,
                error TEXT DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_redemptions_status ON redemptions(status);
        """)
        await self._db.execute("UPDATE schema_version SET version = 8")
        await self._db.commit()
        log.info("database.migrated", from_version=7, to_version=8)

    async def _migrate_v8_to_v9(self) -> None:
        """Add is_paper column to cost_basis and portfolio.

        Existing rows are backfilled with is_paper=1 because all prior
        state in these tables was written before the paper/live split
        was enforced — it's unsafe to assume any of it is live.
        """
        alterations = [
            ("cost_basis", "is_paper INTEGER NOT NULL DEFAULT 1"),
            ("portfolio", "is_paper INTEGER NOT NULL DEFAULT 1"),
        ]
        for table, column_def in alterations:
            try:
                await self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")
            except Exception:
                pass  # Column may already exist

        await self._db.execute("UPDATE schema_version SET version = 9")
        await self._db.commit()
        log.info("database.migrated", from_version=8, to_version=9)

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._db

    async def execute(self, sql: str, params: tuple = ()) -> aiosqlite.Cursor:
        return await self.db.execute(sql, params)

    async def executemany(self, sql: str, params_seq: list[tuple]) -> None:
        await self.db.executemany(sql, params_seq)

    async def fetchone(self, sql: str, params: tuple = ()) -> aiosqlite.Row | None:
        cursor = await self.db.execute(sql, params)
        return await cursor.fetchone()

    async def fetchall(self, sql: str, params: tuple = ()) -> list[aiosqlite.Row]:
        cursor = await self.db.execute(sql, params)
        return await cursor.fetchall()

    async def commit(self) -> None:
        await self.db.commit()
