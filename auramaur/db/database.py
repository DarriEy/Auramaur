"""SQLite database manager with async support."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import aiosqlite
import structlog

from auramaur.db.models import SCHEMA_VERSION, TABLES
from auramaur.runtime import db_path as runtime_db_path

log = structlog.get_logger()


class Database:
    def __init__(self, db_path: str | None = None):
        self.db_path = str(runtime_db_path()) if db_path is None else db_path
        self._db: aiosqlite.Connection | None = None
        self._txn_lock = asyncio.Lock()
        self._txn_task: asyncio.Task | None = None

    async def connect(self, ensure_schema: bool = True) -> None:
        """Open the connection.

        ``ensure_schema=False`` is the fast path for CLI/tooling callers: when
        the stored schema_version already matches SCHEMA_VERSION, the DDL
        executescript is skipped entirely, so a routine CLI invocation takes
        NO write locks against the live bot's database. A behind/fresh
        database still gets the full init — the flag can never leave a caller
        on a stale schema.
        """
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        # foreign_keys stays OFF by design, NOT as repair debt. The markets
        # table is curated/ephemeral — resolved markets age out of it — so a
        # trade/signal/portfolio row legitimately outlives its markets row.
        # The schema's trades->markets FK is therefore too strict: an audit
        # (PRAGMA foreign_key_check, 2026-06-23) found ~1843 such legitimate
        # orphans. Re-enabling FKs would reject valid inserts; the fix, if ever
        # wanted, is to RELAX the FK declarations, not flip this PRAGMA.
        await self._db.execute("PRAGMA foreign_keys=OFF")
        # CLI commands share the file with the running bot; without a busy
        # timeout a writer collision fails instantly ("database is locked" —
        # bit the ledger backfill 2026-06-10). The writer WAITS up to this long
        # for the lock instead of erroring. Raised 5s -> 30s on 2026-06-25 after
        # a restart write-burst exceeded 5s and DROPPED 3 LIVE polymarket fills
        # in order_monitor.record_fill (logged, then skipped — record_fill is
        # not idempotent mid-transaction, so a retry could double-book; making
        # the lock WAIT is the safe fix). 30s covers transient bursts; a lock
        # beyond it would signal sustained write saturation, a capacity problem.
        await self._db.execute("PRAGMA busy_timeout=30000")
        # WAL-safe fsync reduction: NORMAL syncs the WAL at checkpoint, not on
        # every commit. Durability loss is bounded to the last commit(s) on
        # POWER FAILURE only (app crashes lose nothing) — an accepted trade
        # for shorter write-lock hold times in a contended single-writer file.
        await self._db.execute("PRAGMA synchronous=NORMAL")
        if ensure_schema or not await self._schema_is_current():
            await self._init_schema()
        log.info("database.connected", path=self.db_path)

    async def _schema_is_current(self) -> bool:
        try:
            cursor = await self._db.execute(
                "SELECT version FROM schema_version LIMIT 1")
            row = await cursor.fetchone()
        except aiosqlite.OperationalError:
            return False  # fresh file — no schema_version table yet
        if row is None:
            return False
        current = row[0] if isinstance(row[0], int) else row["version"]
        return current >= SCHEMA_VERSION

    @asynccontextmanager
    async def transaction(self):
        """Serialized, atomic write transaction on the shared connection.

        ~30 pillar tasks share ONE aiosqlite connection with implicit
        deferred transactions. Without serialization, task B's ``commit()``
        lands task A's half-written rows, and an error-path ``rollback()``
        can discard ANOTHER task's uncommitted writes — the reason
        record_fill has never been retry-safe. ``BEGIN IMMEDIATE`` under an
        asyncio.Lock gives each adopter a private, atomic write that claims
        the file's write lock up front.

        NEVER await network I/O while holding this — the >250ms warning
        exists to catch exactly that regression.
        """
        # Same-task re-entrancy JOINS the outer transaction instead of issuing
        # a nested BEGIN (sqlite: "cannot start a transaction within a
        # transaction" — the 2026-07-20 position_sync errors). The outer
        # holder's commit/rollback governs the joined work.
        if self._txn_task is not None and self._txn_task is asyncio.current_task():
            yield self
            return
        async with self._txn_lock:
            # A legacy autocommit-style writer may be mid-flight (implicit
            # txn open, its commit() imminent). Wait it out rather than
            # committing on its behalf — that would be the exact bleed this
            # context manager exists to remove. Bounded so a genuinely wedged
            # writer still fails loudly in BEGIN — but the bound must OUTLAST
            # real write bursts: measured engine bursts run 1-3.4s, and a 2s
            # bound turned every unlucky overlap into a position_sync error
            # (2026-07-20). 15s clears every burst ever observed while still
            # bounding a true wedge.
            deadline = time.monotonic() + 15.0
            while self._db._conn.in_transaction and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
            started = time.monotonic()
            await self.db.execute("BEGIN IMMEDIATE")
            self._txn_task = asyncio.current_task()
            try:
                yield self
            except BaseException:
                await self.db.execute("ROLLBACK")
                raise
            else:
                try:
                    await self.db.execute("COMMIT")
                except Exception as exc:  # noqa: BLE001 — narrow handling
                    if "no transaction is active" not in str(exc).lower():
                        raise
                    # A legacy writer's commit() landed mid-transaction and
                    # ended it early: the batch's rows ARE durable, but its
                    # atomicity was violated. Log loudly (this is the bleed
                    # phase 5 exists to retire) without failing the writer —
                    # failing here reports durable writes as failures.
                    log.warning("database.transaction_commit_bled")
            finally:
                self._txn_task = None
                held = time.monotonic() - started
                if held > 0.25:
                    log.warning("database.transaction_held_long",
                                seconds=round(held, 3))

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
        # Indexes on migration-added columns must be created HERE, after both
        # paths guarantee the column exists (fresh DDL or just-run migration) —
        # never in TABLES, which executes before migrations.
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_manager_proposals_class "
            "ON manager_proposals(thesis_class, status)")
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
        if from_version < 19:
            await self._migrate_v18_to_v19()
        if from_version < 20:
            await self._migrate_v19_to_v20()
        if from_version < 21:
            await self._migrate_v20_to_v21()
        if from_version < 22:
            await self._migrate_v21_to_v22()
        if from_version < 23:
            await self._migrate_v22_to_v23()
        if from_version < 24:
            await self._migrate_v23_to_v24()
        if from_version < 25:
            await self._migrate_v24_to_v25()
        if from_version < 26:
            await self._migrate_v25_to_v26()
        if from_version < 27:
            await self._migrate_v26_to_v27()
        if from_version < 28:
            await self._migrate_v27_to_v28()
        if from_version < 29:
            await self._migrate_v28_to_v29()
        if from_version < 30:
            await self._migrate_v29_to_v30()
        if from_version < 31:
            await self._migrate_v30_to_v31()
        if from_version < 32:
            await self._migrate_v31_to_v32()
        if from_version < 33:
            await self._migrate_v32_to_v33()
        if from_version < 34:
            await self._migrate_v33_to_v34()
        if from_version < 35:
            await self._migrate_v34_to_v35()

    async def _migrate_v29_to_v30(self) -> None:
        """Add cost-adjusted IBKR round-trip observations."""
        for column in (
            "entry_commission_usd REAL NOT NULL DEFAULT 0",
            "entry_fill_ref TEXT NOT NULL DEFAULT ''",
        ):
            try:
                await self._db.execute(
                    f"ALTER TABLE ibkr_paper_positions ADD COLUMN {column}")
            except aiosqlite.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS ibkr_paper_round_trips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book TEXT NOT NULL, instrument_key TEXT NOT NULL,
                entry_fill_ref TEXT NOT NULL DEFAULT '',
                exit_fill_ref TEXT NOT NULL UNIQUE,
                gross_pnl_usd REAL NOT NULL,
                entry_commission_usd REAL NOT NULL DEFAULT 0,
                exit_commission_usd REAL NOT NULL DEFAULT 0,
                financing_usd REAL NOT NULL DEFAULT 0,
                borrow_usd REAL NOT NULL DEFAULT 0,
                roll_cost_usd REAL NOT NULL DEFAULT 0,
                intelligence_cost_usd REAL NOT NULL DEFAULT 0,
                net_pnl_usd REAL NOT NULL, opened_at TEXT NOT NULL,
                closed_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_ibkr_round_trips_book_closed
                ON ibkr_paper_round_trips(book, closed_at);
        """)
        await self._db.execute("UPDATE schema_version SET version = 30")
        await self._db.commit()
        log.info("database.migrated", from_version=29, to_version=30)

    async def _migrate_v30_to_v31(self) -> None:
        """Add the interim-manager proposal queue."""
        await self._db.execute(
            """CREATE TABLE IF NOT EXISTS manager_proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                venue TEXT NOT NULL,
                market_id TEXT NOT NULL,
                side TEXT NOT NULL,
                fair_prob REAL NOT NULL,
                stake_usd REAL NOT NULL,
                thesis TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                decided_at TEXT
            )""")
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_manager_proposals_status "
            "ON manager_proposals(status, created_at)")
        await self._db.execute("UPDATE schema_version SET version = 31")
        await self._db.commit()
        log.info("database.migrated", from_version=30, to_version=31)

    async def _migrate_v31_to_v32(self) -> None:
        """Structured thesis columns for the interim-manager compiler contract."""
        for column in (
            "thesis_class TEXT NOT NULL DEFAULT 'unclassified'",
            "confidence_lo REAL", "confidence_hi REAL", "max_entry_price REAL",
            "catalyst TEXT NOT NULL DEFAULT ''",
            "invalidation TEXT NOT NULL DEFAULT ''",
            "sunset_at TEXT", "robust_edge REAL", "decision_price REAL",
        ):
            try:
                await self._db.execute(
                    f"ALTER TABLE manager_proposals ADD COLUMN {column}")
            except Exception:  # noqa: BLE001 — column already present
                pass
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_manager_proposals_class "
            "ON manager_proposals(thesis_class, status)")
        await self._db.execute("UPDATE schema_version SET version = 32")
        await self._db.commit()
        log.info("database.migrated", from_version=31, to_version=32)

    async def _migrate_v32_to_v33(self) -> None:
        """Daily marks + research signal recordings (new tables only)."""
        await self._db.execute(
            """CREATE TABLE IF NOT EXISTS ibkr_paper_daily_marks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book TEXT NOT NULL, mark_date TEXT NOT NULL,
                equity_usd REAL NOT NULL,
                realized_cum_usd REAL NOT NULL DEFAULT 0,
                unrealized_usd REAL NOT NULL DEFAULT 0,
                marked_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(book, mark_date))""")
        await self._db.execute(
            """CREATE TABLE IF NOT EXISTS ibkr_research_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instrument_key TEXT NOT NULL, signal_date TEXT NOT NULL,
                signal_name TEXT NOT NULL, direction INTEGER NOT NULL,
                strength REAL NOT NULL DEFAULT 0,
                detail TEXT NOT NULL DEFAULT '',
                recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(instrument_key, signal_date, signal_name))""")
        await self._db.execute("UPDATE schema_version SET version = 33")
        await self._db.commit()
        log.info("database.migrated", from_version=32, to_version=33)

    async def _migrate_v33_to_v34(self) -> None:
        """Tag manager proposals with their author (operator vs auto)."""
        try:
            await self._db.execute(
                "ALTER TABLE manager_proposals ADD COLUMN "
                "proposer TEXT NOT NULL DEFAULT 'operator'")
        except Exception:  # noqa: BLE001 — column already present
            pass
        await self._db.execute("UPDATE schema_version SET version = 34")
        await self._db.commit()
        log.info("database.migrated", from_version=33, to_version=34)

    async def _migrate_v34_to_v35(self) -> None:
        """Local LLM tier tables (distilled_claims, distill_progress,
        local_llm_calls)."""
        # TABLES has already created the additive tables; only advance the
        # version so older live databases converge without destructive DDL.
        await self._db.execute("UPDATE schema_version SET version = 35")
        await self._db.commit()
        log.info("database.migrated", from_version=34, to_version=35)

    async def _migrate_v28_to_v29(self) -> None:
        """Add immutable strategy-research and CLV accounting tables."""
        # TABLES has already created the additive tables; only advance the
        # version so older live databases converge without destructive DDL.
        await self._db.execute("UPDATE schema_version SET version = 29")
        await self._db.commit()
        log.info("database.migrated", from_version=28, to_version=29)

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

    async def _migrate_v18_to_v19(self) -> None:
        """Add CLOB outcome-token columns to markets.

        Needed to resolve which SIDE a held token is: outcome labels like
        "Something"/"Nothing" aren't YES/NO, and the position syncer's
        YES-default marked such holdings at the wrong outcome's price.
        """
        for column_def in ("clob_token_yes TEXT DEFAULT ''", "clob_token_no TEXT DEFAULT ''"):
            try:
                await self._db.execute(f"ALTER TABLE markets ADD COLUMN {column_def}")
            except Exception:
                pass  # Column already exists
        await self._db.execute("UPDATE schema_version SET version = 19")
        await self._db.commit()
        log.info("database.migrated", from_version=18, to_version=19)

    async def _migrate_v19_to_v20(self) -> None:
        """Collapse casing-split cost_basis rows into canonical YES/NO tokens.

        Pre-v20 the live reconciler wrote the raw CLOB outcome ("Yes"/"No")
        while the fill and Kalshi paths wrote TokenType values ("YES"/"NO"),
        so one position could exist as two PK-distinct rows differing only by
        case — and the token-blind getters returned an arbitrary one. Merge
        each (market_id, is_paper, canonical-token) group into a single row:
        the newest ``updated_at`` wins for size/avg_cost/total_cost (the
        reconciler's CLOB ground truth), and realized_pnl is summed so no
        realized history is dropped. Genuine two-sided positions (distinct
        canonical YES and NO) are preserved as two rows. A full pre-migration
        snapshot is kept in ``cost_basis_backup_v20`` (reversible).
        """
        await self._db.execute("DROP TABLE IF EXISTS cost_basis_backup_v20")
        await self._db.execute(
            "CREATE TABLE cost_basis_backup_v20 AS SELECT * FROM cost_basis"
        )

        cursor = await self._db.execute(
            "SELECT market_id, token, token_id, size, avg_cost, total_cost,"
            " realized_pnl, is_paper, updated_at FROM cost_basis"
        )
        rows = await cursor.fetchall()

        def canon(t: object) -> str:
            return "NO" if str(t or "").strip().upper() == "NO" else "YES"

        groups: dict[tuple, list] = {}
        for r in rows:
            groups.setdefault((r["market_id"], r["is_paper"], canon(r["token"])), []).append(r)

        merged = 0
        for (market_id, is_paper, token), grp in groups.items():
            # Already-canonical singleton: nothing to rewrite.
            if len(grp) == 1 and grp[0]["token"] == token:
                continue
            # Newest write wins for current holdings; tiebreak on larger size.
            authoritative = max(
                grp, key=lambda r: (str(r["updated_at"] or ""), float(r["size"] or 0))
            )
            realized = sum(float(r["realized_pnl"] or 0) for r in grp)
            token_id = authoritative["token_id"] or next(
                (r["token_id"] for r in grp if r["token_id"]), ""
            )
            for r in grp:
                await self._db.execute(
                    "DELETE FROM cost_basis WHERE market_id = ? AND is_paper = ? AND token = ?",
                    (market_id, is_paper, r["token"]),
                )
            await self._db.execute(
                "INSERT INTO cost_basis (market_id, token, token_id, size, avg_cost,"
                " total_cost, realized_pnl, is_paper, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    market_id, token, token_id,
                    float(authoritative["size"] or 0),
                    float(authoritative["avg_cost"] or 0),
                    float(authoritative["total_cost"] or 0),
                    realized, is_paper, authoritative["updated_at"],
                ),
            )
            if len(grp) > 1:
                merged += 1

        await self._db.execute("UPDATE schema_version SET version = 20")
        await self._db.commit()
        log.info("database.migrated", from_version=19, to_version=20, merged_groups=merged)

    async def _migrate_v20_to_v21(self) -> None:
        """Register additive v21 schemas.

        ``TABLES`` runs before migrations and creates both the existing IBKR
        paper schema and the lineage/graduation tables. No populated table is
        rebuilt or indexed by this migration.
        """
        await self._db.execute("UPDATE schema_version SET version = 21")
        await self._db.commit()
        log.info("database.migrated", from_version=20, to_version=21)

    async def _migrate_v21_to_v22(self) -> None:
        """Register the additive lineage and information-graduation schema."""
        try:
            await self._db.execute(
                "ALTER TABLE source_fetches ADD COLUMN information_mode TEXT "
                "NOT NULL DEFAULT 'production'"
            )
        except Exception:
            pass
        await self._db.execute("UPDATE schema_version SET version = 22")
        await self._db.commit()
        log.info("database.migrated", from_version=21, to_version=22)

    async def _migrate_v22_to_v23(self) -> None:
        """Register isolated IBKR multi-asset paper accounting tables."""
        await self._db.execute("UPDATE schema_version SET version = 23")
        await self._db.commit()
        log.info("database.migrated", from_version=22, to_version=23)

    async def _migrate_v23_to_v24(self) -> None:
        """Record the market-data source used for each simulated mark/fill."""
        for table in ("ibkr_paper_positions", "ibkr_paper_fills"):
            try:
                await self._db.execute(
                    f"ALTER TABLE {table} ADD COLUMN price_source TEXT "
                    "NOT NULL DEFAULT 'ibkr_unknown'")
            except aiosqlite.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
        for table in ("ibkr_paper_positions", "ibkr_paper_fills"):
            columns = await self.fetchall(f"PRAGMA table_info({table})")
            if "price_source" not in {row["name"] for row in columns}:
                raise RuntimeError(f"migration did not add {table}.price_source")
        await self._db.execute("UPDATE schema_version SET version = 24")
        await self._db.commit()
        log.info("database.migrated", from_version=23, to_version=24)

    async def _migrate_v24_to_v25(self) -> None:
        """Persist enough instrument identity to manage catalog orphans."""
        try:
            await self._db.execute(
                "ALTER TABLE ibkr_paper_positions ADD COLUMN "
                "instrument_spec_json TEXT NOT NULL DEFAULT ''")
        except aiosqlite.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
        columns = await self.fetchall("PRAGMA table_info(ibkr_paper_positions)")
        if "instrument_spec_json" not in {row["name"] for row in columns}:
            raise RuntimeError("migration did not add instrument_spec_json")
        await self._db.execute("UPDATE schema_version SET version = 25")
        await self._db.commit()
        log.info("database.migrated", from_version=24, to_version=25)

    async def _migrate_v25_to_v26(self) -> None:
        """Register the persistent IBKR qualified-contract registry."""
        # Self-sufficient: do not depend on connect() having applied the
        # current TABLES script before migrations run.
        await self._db.execute("""CREATE TABLE IF NOT EXISTS ibkr_contract_registry (
    instrument_key TEXT PRIMARY KEY,
    book TEXT NOT NULL,
    kind TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    con_id INTEGER NOT NULL,
    local_symbol TEXT NOT NULL DEFAULT '',
    trading_class TEXT NOT NULL DEFAULT '',
    exchange TEXT NOT NULL DEFAULT '',
    currency TEXT NOT NULL DEFAULT '',
    multiplier REAL NOT NULL DEFAULT 1,
    status TEXT NOT NULL CHECK(status IN
        ('eligible', 'qualified_no_live_data', 'pending_approval',
         'quarantined', 'drifted')),
    approved INTEGER NOT NULL DEFAULT 0,
    approval_reason TEXT NOT NULL DEFAULT '',
    quote_source TEXT NOT NULL DEFAULT 'ibkr_unknown',
    has_history INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',
    qualified_at TEXT NOT NULL DEFAULT (datetime('now')),
    validated_at TEXT NOT NULL DEFAULT (datetime('now')),
    approved_at TEXT
)""")
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_ibkr_contract_registry_status "
            "ON ibkr_contract_registry(book, status, approved)")
        await self._db.execute("UPDATE schema_version SET version = 26")
        await self._db.commit()
        log.info("database.migrated", from_version=25, to_version=26)

    async def _migrate_v26_to_v27(self) -> None:
        """Persist immutable entry risk for IBKR paper positions."""
        for table in ("ibkr_paper_positions", "ibkr_etf_positions"):
            for name in ("stop_price", "initial_risk_usd"):
                try:
                    await self._db.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} REAL NOT NULL DEFAULT 0")
                except aiosqlite.OperationalError as exc:
                    if "duplicate column name" not in str(exc).lower():
                        raise
        await self._db.execute("UPDATE schema_version SET version = 27")
        await self._db.commit()
        log.info("database.migrated", from_version=26, to_version=27)

    async def _migrate_v27_to_v28(self) -> None:
        """Add the restart-safe, wallet-independent Kraken paper book."""
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS kraken_paper_positions (
                strategy TEXT NOT NULL DEFAULT 'llm', pair TEXT NOT NULL,
                quantity REAL NOT NULL, entry_price REAL NOT NULL,
                peak_gain_pct REAL NOT NULL DEFAULT 0,
                opened_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (strategy, pair)
            );
            CREATE INDEX IF NOT EXISTS idx_kraken_paper_positions_pair
                ON kraken_paper_positions(pair);
        """)
        await self._db.execute("UPDATE schema_version SET version = 28")
        await self._db.commit()
        log.info("database.migrated", from_version=27, to_version=28)

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
        try:
            await self.db.commit()
        except Exception as exc:  # noqa: BLE001 — narrow re-raise below
            if "no transaction is active" in str(exc).lower():
                # A transaction() adopter's COMMIT already landed these rows
                # (legacy commit interleaved with an explicit transaction on
                # the shared connection). The data is durable; only this
                # caller's notion of "its own" commit was stale. Full adopter
                # migration (contention plan phase 5) retires this path.
                log.debug("database.commit_already_landed")
                return
            raise

    async def rollback(self) -> None:
        await self.db.rollback()
