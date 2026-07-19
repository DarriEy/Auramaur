import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from auramaur.db.database import Database


@pytest.mark.asyncio
async def test_v23_migration_adds_verified_columns():
    path = Path("runtime/test-v23-migration.db")
    path.unlink(missing_ok=True)
    raw = sqlite3.connect(path)
    raw.executescript(
        """CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version VALUES (23);
        CREATE TABLE ibkr_paper_positions (
          book TEXT, instrument_key TEXT, con_id INTEGER, currency TEXT,
          quantity REAL, multiplier REAL, fx_to_usd REAL, avg_cost REAL,
          current_price REAL, unrealized_pnl_usd REAL, opened_at TEXT,
          updated_at TEXT, PRIMARY KEY (book, instrument_key));
        CREATE TABLE ibkr_paper_fills (
          id INTEGER PRIMARY KEY, book TEXT, instrument_key TEXT, con_id INTEGER,
          side TEXT, quantity REAL, multiplier REAL, price REAL, currency TEXT,
          fx_to_usd REAL, commission_usd REAL, fill_ref TEXT UNIQUE,
          filled_at TEXT);
        """)
    raw.close()

    db = Database(str(path))
    await db.connect()
    version = await db.fetchone("SELECT version FROM schema_version")
    position_columns = await db.fetchall("PRAGMA table_info(ibkr_paper_positions)")
    fill_columns = await db.fetchall("PRAGMA table_info(ibkr_paper_fills)")
    assert version["version"] == 27
    assert {row["name"] for row in position_columns} >= {
        "price_source", "instrument_spec_json", "stop_price", "initial_risk_usd"}
    assert "price_source" in {row["name"] for row in fill_columns}
    registry = await db.fetchall("PRAGMA table_info(ibkr_contract_registry)")
    assert {"instrument_key", "manifest_hash", "con_id", "status", "approved"} <= {
        row["name"] for row in registry}
    await db.close()
    path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_v23_migration_does_not_stamp_version_after_busy_error():
    db = Database(":memory:")
    await db.connect()
    await db.execute("UPDATE schema_version SET version = 23")
    await db.commit()
    connection = db._db

    class BusyConnection:
        async def execute(self, sql, params=()):
            if sql.startswith("ALTER TABLE"):
                raise aiosqlite.OperationalError("database is locked")
            return await connection.execute(sql, params)

    db._db = BusyConnection()
    with pytest.raises(aiosqlite.OperationalError, match="locked"):
        await db._migrate_v23_to_v24()
    version = await db.fetchone("SELECT version FROM schema_version")
    assert version["version"] == 23
    db._db = connection
    await db.close()
