#!/usr/bin/env python3
"""Consolidate a pre-cutover Auramaur SQLite DB with a fresh post-cutover DB.

The historical database is copied and migrated in place; neither input is
modified.  Every populated table in the current database must have an explicit
merge policy, otherwise the operation aborts before committing.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path


APPEND_WITH_NEW_ID = {
    "agent_trader_theses",
    "calibration",
    "fills",
    "forecast_snapshots",
    "ibkr_etf_openai_attempts",
    "kalshi_execution_samples",
    "orderbook_snapshots",
    "price_history",
    "slippage_log",
}

APPEND_BY_UNIQUE_KEY = {
    "evidence_observations",
    "ibkr_etf_ledger",
    "ibkr_paper_daily_marks",
    "ibkr_paper_fills",
    "ibkr_paper_ledger",
    "ibkr_research_signals",
    "ingestion_runs",
    "intraday_drift_obs",
    "pnl_ledger",
    "source_fetches",
}

# These are live snapshots, not historical journals.  Current state is
# authoritative even when it is empty.
REPLACE_WHOLE_TABLE = {
    "cost_basis",
    "ibkr_etf_cooldowns",
    "ibkr_etf_positions",
    "ibkr_etf_state",
    "ibkr_paper_positions",
    "ibkr_paper_state",
    "kraken_dir_signals",
    "kraken_paper_positions",
    "order_build_drops",
    "portfolio",
    "position_peaks",
    "signal_rejections",
    "strategy_heartbeats",
    "venue_balances",
    "venue_positions",
}

# Catalog/cache rows are retained from history, with current rows winning on
# primary-key collisions.
UPSERT_CURRENT = {
    "agent_trader_declines",
    "hydro_watch_seen",
    "ibkr_contract_registry",
    "information_graduation_state",
    "information_strategies",
    "lens_verdicts",
    "markets",
    "nlp_cache",
}

SUM_COUNTERS = {
    "agent_trader_costs": ("calls", "usd"),
    "llm_call_counter": ("claude_calls",),
}

SPECIAL = {
    "daily_stats",
    "decision_marks",
    "decision_snapshots",
    "schema_version",
    "signals",
    "trades",
}


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def table_columns(conn: sqlite3.Connection, schema: str, table: str) -> list[str]:
    return [
        row[1]
        for row in conn.execute(
            f"PRAGMA {quote(schema)}.table_info({quote(table)})"
        )
    ]


def primary_key(conn: sqlite3.Connection, schema: str, table: str) -> list[str]:
    rows = conn.execute(
        f"PRAGMA {quote(schema)}.table_info({quote(table)})"
    ).fetchall()
    return [row[1] for row in sorted(rows, key=lambda row: row[5]) if row[5]]


def copy_rows(
    conn: sqlite3.Connection,
    table: str,
    *,
    omit: set[str] | None = None,
    conflict: str = "",
) -> None:
    omit = omit or set()
    columns = [c for c in table_columns(conn, "main", table) if c not in omit]
    current_columns = set(table_columns(conn, "current", table))
    if not set(columns) <= current_columns:
        missing = sorted(set(columns) - current_columns)
        raise RuntimeError(f"{table}: current database lacks columns {missing}")
    names = ",".join(map(quote, columns))
    conn.execute(
        f"INSERT {conflict} INTO {quote(table)} ({names}) "
        f"SELECT {names} FROM current.{quote(table)}"
    )


def upsert_current(conn: sqlite3.Connection, table: str) -> None:
    pk = primary_key(conn, "main", table)
    if not pk:
        raise RuntimeError(f"{table}: UPSERT policy requires a primary key")
    columns = table_columns(conn, "main", table)
    names = ",".join(map(quote, columns))
    updates = [c for c in columns if c not in pk]
    if updates:
        assignment = ",".join(
            f"{quote(c)}=excluded.{quote(c)}" for c in updates
        )
        action = f"DO UPDATE SET {assignment}"
    else:
        action = "DO NOTHING"
    conn.execute(
        f"INSERT INTO {quote(table)} ({names}) "
        f"SELECT {names} FROM current.{quote(table)} WHERE true "
        f"ON CONFLICT ({','.join(map(quote, pk))}) {action}"
    )


def migrate_v38_to_v39(conn: sqlite3.Connection) -> None:
    version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
    if version == 39:
        return
    if version != 38:
        raise RuntimeError(f"historical schema is v{version}, expected v38 or v39")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS ibkr_execution_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            execution_ref TEXT NOT NULL UNIQUE,
            book TEXT NOT NULL, instrument_key TEXT NOT NULL,
            side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
            requested_quantity REAL NOT NULL,
            order_id TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'submitting',
            filled_quantity REAL NOT NULL DEFAULT 0,
            avg_fill_price REAL NOT NULL DEFAULT 0,
            accounted INTEGER NOT NULL DEFAULT 0,
            error TEXT NOT NULL DEFAULT '',
            submitted_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_ibkr_execution_unaccounted
            ON ibkr_execution_orders(
                book,instrument_key,side,accounted,updated_at
            );
        UPDATE schema_version SET version = 39;
        """
    )


def merge_daily_stats(conn: sqlite3.Connection) -> None:
    columns = table_columns(conn, "main", "daily_stats")
    names = ",".join(map(quote, columns))
    additive = {
        "total_pnl",
        "trades_count",
        "wins",
        "losses",
        "api_calls_claude",
        "api_cost_estimate",
    }
    assignments: list[str] = []
    for column in columns:
        if column == "date":
            continue
        if column in additive:
            assignments.append(
                f"{quote(column)}={quote('daily_stats')}.{quote(column)}"
                f"+excluded.{quote(column)}"
            )
        elif column == "max_drawdown":
            assignments.append(
                f"{quote(column)}=max({quote('daily_stats')}.{quote(column)},"
                f"excluded.{quote(column)})"
            )
        else:
            # Peak balance and any future non-additive daily state use the
            # current deployment's value.
            assignments.append(f"{quote(column)}=excluded.{quote(column)}")
    conn.execute(
        f"INSERT INTO daily_stats ({names}) SELECT {names} "
        f"FROM current.daily_stats WHERE true ON CONFLICT(date) DO UPDATE SET "
        + ",".join(assignments)
    )


def merge_sum_counter(
    conn: sqlite3.Connection, table: str, additive: tuple[str, ...]
) -> None:
    pk = primary_key(conn, "main", table)
    columns = table_columns(conn, "main", table)
    names = ",".join(map(quote, columns))
    assignments = [
        f"{quote(c)}={quote(table)}.{quote(c)}+excluded.{quote(c)}"
        if c in additive
        else f"{quote(c)}=excluded.{quote(c)}"
        for c in columns
        if c not in pk
    ]
    conn.execute(
        f"INSERT INTO {quote(table)} ({names}) "
        f"SELECT {names} FROM current.{quote(table)} WHERE true "
        f"ON CONFLICT ({','.join(map(quote, pk))}) DO UPDATE SET "
        + ",".join(assignments)
    )


def merge_signal_graph(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TEMP TABLE signal_id_map "
        "(old_id INTEGER PRIMARY KEY, new_id INTEGER NOT NULL)"
    )
    signal_columns = [c for c in table_columns(conn, "main", "signals") if c != "id"]
    signal_names = ",".join(map(quote, signal_columns))
    for row in conn.execute(
        f"SELECT id,{signal_names} FROM current.signals ORDER BY id"
    ).fetchall():
        cursor = conn.execute(
            f"INSERT INTO signals ({signal_names}) "
            f"VALUES ({','.join('?' for _ in signal_columns)})",
            row[1:],
        )
        conn.execute("INSERT INTO signal_id_map VALUES (?,?)", (row[0], cursor.lastrowid))

    trade_columns = [c for c in table_columns(conn, "main", "trades") if c != "id"]
    trade_select = [
        "m.new_id" if c == "signal_id" else f"t.{quote(c)}" for c in trade_columns
    ]
    conn.execute(
        f"INSERT INTO trades ({','.join(map(quote, trade_columns))}) "
        f"SELECT {','.join(trade_select)} FROM current.trades t "
        "LEFT JOIN signal_id_map m ON m.old_id=t.signal_id"
    )

    conn.execute(
        "CREATE TEMP TABLE decision_id_map "
        "(old_id INTEGER PRIMARY KEY, new_id INTEGER NOT NULL)"
    )
    decision_columns = [
        c for c in table_columns(conn, "main", "decision_snapshots") if c != "id"
    ]
    # decision_snapshots.signal_id is a separate text decision identifier; it
    # does not reference the integer signals.id key.
    select_columns = [f"d.{quote(c)}" for c in decision_columns]
    rows = conn.execute(
        f"SELECT d.id,{','.join(select_columns)} "
        "FROM current.decision_snapshots d ORDER BY d.id"
    ).fetchall()
    for row in rows:
        cursor = conn.execute(
            f"INSERT INTO decision_snapshots "
            f"({','.join(map(quote, decision_columns))}) "
            f"VALUES ({','.join('?' for _ in decision_columns)})",
            row[1:],
        )
        conn.execute(
            "INSERT INTO decision_id_map VALUES (?,?)", (row[0], cursor.lastrowid)
        )
    mark_columns = table_columns(conn, "main", "decision_marks")
    mark_select = [
        "m.new_id" if c == "decision_id" else f"d.{quote(c)}" for c in mark_columns
    ]
    conn.execute(
        f"INSERT INTO decision_marks ({','.join(map(quote, mark_columns))}) "
        f"SELECT {','.join(mark_select)} FROM current.decision_marks d "
        "JOIN decision_id_map m ON m.old_id=d.decision_id"
    )


def populated_tables(conn: sqlite3.Connection, schema: str) -> set[str]:
    tables = {
        row[0]
        for row in conn.execute(
            f"SELECT name FROM {quote(schema)}.sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    return {
        table
        for table in tables
        if conn.execute(
            f"SELECT EXISTS(SELECT 1 FROM {quote(schema)}.{quote(table)})"
        ).fetchone()[0]
    }


def consolidate(historical: Path, current: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    shutil.copy2(historical, output)
    output.chmod(0o600)
    conn = sqlite3.connect(output)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=OFF")
        migrate_v38_to_v39(conn)
        conn.execute("ATTACH DATABASE ? AS current", (str(current),))
        current_version = conn.execute(
            "SELECT version FROM current.schema_version"
        ).fetchone()[0]
        if current_version != 39:
            raise RuntimeError(f"current schema is v{current_version}, expected v39")

        handled = (
            APPEND_WITH_NEW_ID
            | APPEND_BY_UNIQUE_KEY
            | REPLACE_WHOLE_TABLE
            | UPSERT_CURRENT
            | set(SUM_COUNTERS)
            | SPECIAL
        )
        unhandled = populated_tables(conn, "current") - handled
        if unhandled:
            raise RuntimeError(f"populated current tables lack policy: {sorted(unhandled)}")

        conn.execute("BEGIN IMMEDIATE")
        for table in sorted(APPEND_WITH_NEW_ID):
            copy_rows(conn, table, omit={"id"})
        for table in sorted(APPEND_BY_UNIQUE_KEY):
            omit = {"id"} if "id" in table_columns(conn, "main", table) else set()
            copy_rows(conn, table, omit=omit, conflict="OR IGNORE")
        for table in sorted(REPLACE_WHOLE_TABLE):
            conn.execute(f"DELETE FROM {quote(table)}")
            copy_rows(conn, table)
        for table in sorted(UPSERT_CURRENT):
            upsert_current(conn, table)
        for table, columns in SUM_COUNTERS.items():
            merge_sum_counter(conn, table, columns)
        merge_daily_stats(conn)
        merge_signal_graph(conn)
        conn.commit()
        conn.execute("PRAGMA optimize")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    consolidate(args.historical, args.current, args.output)
