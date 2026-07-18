from __future__ import annotations

import sqlite3

from auramaur.health import _database_health


def test_database_health_requires_database(tmp_path):
    ok, detail = _database_health(tmp_path / "missing.db")
    assert ok is False
    assert detail == "database missing"


def test_database_health_reads_schema_version(tmp_path):
    path = tmp_path / "healthy.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO schema_version VALUES (24)")
    ok, detail = _database_health(path)
    assert ok is True
    assert detail == "schema version 24"


def test_database_health_rejects_uninitialized_sqlite(tmp_path):
    path = tmp_path / "empty.db"
    with sqlite3.connect(path):
        pass
    ok, detail = _database_health(path)
    assert ok is False
    assert "schema_version" in detail
