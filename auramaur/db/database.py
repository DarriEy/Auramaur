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
        # CLI commands share the file with the running bot; without a busy
        # timeout a writer collision fails instantly ("database is locked" —
        # bit the ledger backfill 2026-06-10). 5s covers any sane write txn.
        await self._db.execute("PRAGMA busy_timeout=5000")
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
        if from_version < 10:
            await self._migrate_v9_to_v10()
        if from_version < 11:
            await self._migrate_v10_to_v11()
        if from_version < 12:
            await self._migrate_v11_to_v12()
        if from_version < 13:
            await self._migrate_v12_to_v13()
        if from_version < 14:
            await self._migrate_v13_to_v14()
        if from_version < 15:
            await self._migrate_v14_to_v15()
        if from_version < 16:
            await self._migrate_v15_to_v16()
        if from_version < 17:
            await self._migrate_v16_to_v17()
        if from_version < 18:
            await self._migrate_v17_to_v18()

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

    async def _migrate_v9_to_v10(self) -> None:
        """Make cost_basis primary key composite (market_id, is_paper).

        Previously the PK was ``market_id`` alone, so paper and live fills
        for the same market collided: ``record_fill()`` upserts with
        ``ON CONFLICT(market_id)`` would overwrite each other's cost basis
        and realized PnL.  Recreate the table with the composite PK and
        copy existing rows across.
        """
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS cost_basis_new (
                market_id TEXT NOT NULL,
                token TEXT NOT NULL DEFAULT 'YES',
                token_id TEXT DEFAULT '',
                size REAL NOT NULL,
                avg_cost REAL NOT NULL,
                total_cost REAL NOT NULL,
                realized_pnl REAL DEFAULT 0,
                is_paper INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (market_id, is_paper)
            );
            INSERT OR IGNORE INTO cost_basis_new
                (market_id, token, token_id, size, avg_cost, total_cost,
                 realized_pnl, is_paper, updated_at)
            SELECT market_id, token, token_id, size, avg_cost, total_cost,
                   realized_pnl, is_paper, updated_at
            FROM cost_basis;
            DROP TABLE cost_basis;
            ALTER TABLE cost_basis_new RENAME TO cost_basis;
        """)
        await self._db.execute("UPDATE schema_version SET version = 10")
        await self._db.commit()
        log.info("database.migrated", from_version=9, to_version=10)

    async def _migrate_v10_to_v11(self) -> None:
        """Make portfolio primary key composite (market_id, is_paper).

        Same treatment as cost_basis got in v9→v10: paper and live rows
        for the same market can now coexist.
        """
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS portfolio_new (
                market_id TEXT NOT NULL,
                exchange TEXT DEFAULT 'polymarket',
                side TEXT NOT NULL,
                size REAL NOT NULL,
                avg_price REAL NOT NULL,
                current_price REAL,
                unrealized_pnl REAL DEFAULT 0,
                category TEXT,
                token TEXT NOT NULL DEFAULT 'YES',
                token_id TEXT DEFAULT '',
                is_paper INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (market_id, is_paper),
                FOREIGN KEY (market_id) REFERENCES markets(id)
            );
            INSERT OR IGNORE INTO portfolio_new
                (market_id, exchange, side, size, avg_price, current_price,
                 unrealized_pnl, category, token, token_id, is_paper, updated_at)
            SELECT market_id, exchange, side, size, avg_price, current_price,
                   unrealized_pnl, category, token, token_id, is_paper, updated_at
            FROM portfolio;
            DROP TABLE portfolio;
            ALTER TABLE portfolio_new RENAME TO portfolio;
        """)
        await self._db.execute("UPDATE schema_version SET version = 11")
        await self._db.commit()
        log.info("database.migrated", from_version=10, to_version=11)

    async def _migrate_v13_to_v14(self) -> None:
        """Add ``token`` to the cost_basis and portfolio primary keys.

        The PKs were ``(market_id, is_paper)``, allowing only one row per
        market per mode. But a strategy can hold BOTH the YES and NO outcome in
        the same market, so the second side overwrote the first — corrupting
        cost basis, PnL, exits, and exposure attribution. ``token`` ('YES'/'NO',
        NOT NULL) is the field that distinguishes the two sides, so it belongs
        in the key. (token_id is not used: it defaults to '' and can be empty on
        legacy/Kalshi rows, which would re-introduce collisions.)

        Existing rows have at most one row per (market_id, is_paper) and each
        carries a token value, so they map to unique (market_id, is_paper,
        token) keys — INSERT OR IGNORE preserves them with no loss.
        """
        cursor = await self._db.execute(
            """SELECT market_id, is_paper, COUNT(*) AS c FROM cost_basis
               GROUP BY market_id, is_paper HAVING c > 1"""
        )
        for row in await cursor.fetchall():
            log.warning(
                "migration.v13_to_v14.preexisting_dup", table="cost_basis",
                market_id=row[0], is_paper=row[1], rows=row[2],
            )

        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS cost_basis_new (
                market_id TEXT NOT NULL,
                token TEXT NOT NULL DEFAULT 'YES',
                token_id TEXT DEFAULT '',
                size REAL NOT NULL,
                avg_cost REAL NOT NULL,
                total_cost REAL NOT NULL,
                realized_pnl REAL DEFAULT 0,
                is_paper INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (market_id, is_paper, token)
            );
            INSERT OR IGNORE INTO cost_basis_new
                (market_id, token, token_id, size, avg_cost, total_cost,
                 realized_pnl, is_paper, updated_at)
            SELECT market_id, token, token_id, size, avg_cost, total_cost,
                   realized_pnl, is_paper, updated_at
            FROM cost_basis;
            DROP TABLE cost_basis;
            ALTER TABLE cost_basis_new RENAME TO cost_basis;

            CREATE TABLE IF NOT EXISTS portfolio_new (
                market_id TEXT NOT NULL,
                exchange TEXT DEFAULT 'polymarket',
                side TEXT NOT NULL,
                size REAL NOT NULL,
                avg_price REAL NOT NULL,
                current_price REAL,
                unrealized_pnl REAL DEFAULT 0,
                category TEXT,
                token TEXT NOT NULL DEFAULT 'YES',
                token_id TEXT DEFAULT '',
                is_paper INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (market_id, is_paper, token),
                FOREIGN KEY (market_id) REFERENCES markets(id)
            );
            INSERT OR IGNORE INTO portfolio_new
                (market_id, exchange, side, size, avg_price, current_price,
                 unrealized_pnl, category, token, token_id, is_paper, updated_at)
            SELECT market_id, exchange, side, size, avg_price, current_price,
                   unrealized_pnl, category, token, token_id, is_paper, updated_at
            FROM portfolio;
            DROP TABLE portfolio;
            ALTER TABLE portfolio_new RENAME TO portfolio;
        """)
        await self._db.execute("UPDATE schema_version SET version = 14")
        await self._db.commit()
        log.info("database.migrated", from_version=13, to_version=14)

    async def _migrate_v14_to_v15(self) -> None:
        """Add the pnl_ledger table (created by TABLES executescript; this just
        stamps the version so the backfill CLI can tell a fresh ledger from a
        pre-ledger database)."""
        await self._db.execute("UPDATE schema_version SET version = 15")
        await self._db.commit()
        log.info("database.migrated", from_version=14, to_version=15)

    async def _migrate_v15_to_v16(self) -> None:
        """Add the entailment_verdicts table (created by TABLES executescript)."""
        await self._db.execute("UPDATE schema_version SET version = 16")
        await self._db.commit()
        log.info("database.migrated", from_version=15, to_version=16)

    async def _migrate_v16_to_v17(self) -> None:
        """Add gap_audits + lens_verdicts tables (created by TABLES executescript)."""
        await self._db.execute("UPDATE schema_version SET version = 17")
        await self._db.commit()
        log.info("database.migrated", from_version=16, to_version=17)

    async def _migrate_v17_to_v18(self) -> None:
        """Add the oddlot_filings table (created by TABLES executescript)."""
        await self._db.execute("UPDATE schema_version SET version = 18")
        await self._db.commit()
        log.info("database.migrated", from_version=17, to_version=18)

    async def _migrate_v11_to_v12(self) -> None:
        """Add strategy_source column to signals and trades for hybrid mode attribution."""
        for table in ("signals", "trades"):
            try:
                await self._db.execute(
                    f"ALTER TABLE {table} ADD COLUMN strategy_source TEXT DEFAULT 'llm'"
                )
            except Exception:
                pass  # Column already exists
        await self._db.execute("UPDATE schema_version SET version = 12")
        await self._db.commit()
        log.info("database.migrated", from_version=11, to_version=12)

    async def _migrate_v12_to_v13(self) -> None:
        """Relax legacy ``markets.condition_id`` constraints.

        Older live DBs have ``condition_id TEXT NOT NULL`` with no default,
        even though non-CLOB venues such as Kalshi do not have a condition id.
        Recreate the table with the current nullable/defaulted schema so
        exchange syncers can upsert market metadata consistently.
        """
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS markets_new (
                id TEXT PRIMARY KEY,
                exchange TEXT DEFAULT 'polymarket',
                condition_id TEXT DEFAULT '',
                ticker TEXT DEFAULT '',
                question TEXT NOT NULL,
                description TEXT,
                category TEXT,
                end_date TEXT,
                active INTEGER DEFAULT 1,
                outcome_yes_price REAL,
                outcome_no_price REAL,
                volume REAL DEFAULT 0,
                liquidity REAL DEFAULT 0,
                last_updated TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            INSERT OR IGNORE INTO markets_new
                (id, exchange, condition_id, ticker, question, description,
                 category, end_date, active, outcome_yes_price, outcome_no_price,
                 volume, liquidity, last_updated, created_at)
            SELECT id,
                   COALESCE(exchange, 'polymarket'),
                   COALESCE(condition_id, ''),
                   COALESCE(ticker, ''),
                   COALESCE(question, id),
                   description,
                   category,
                   end_date,
                   COALESCE(active, 1),
                   outcome_yes_price,
                   outcome_no_price,
                   COALESCE(volume, 0),
                   COALESCE(liquidity, 0),
                   COALESCE(last_updated, datetime('now')),
                   COALESCE(created_at, datetime('now'))
            FROM markets;
            DROP TABLE markets;
            ALTER TABLE markets_new RENAME TO markets;
        """)
        await self._db.execute("UPDATE schema_version SET version = 13")
        await self._db.commit()
        log.info("database.migrated", from_version=12, to_version=13)

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
