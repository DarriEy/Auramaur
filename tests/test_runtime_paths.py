from __future__ import annotations

from auramaur import runtime
from auramaur.db.database import Database


def test_runtime_paths_default_to_repo(monkeypatch):
    for name in (
        "AURAMAUR_STATE_DIR", "AURAMAUR_DB_PATH", "AURAMAUR_LOG_DIR",
        "AURAMAUR_KILL_SWITCH_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    assert runtime.db_path() == runtime.REPO_ROOT / "auramaur.db"
    assert runtime.kill_switch_path() == runtime.REPO_ROOT / "KILL_SWITCH"


def test_runtime_paths_are_environment_overridable(tmp_path, monkeypatch):
    monkeypatch.setenv("AURAMAUR_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("AURAMAUR_LOG_DIR", str(tmp_path / "logs"))
    assert runtime.db_path() == tmp_path / "state" / "auramaur.db"
    assert runtime.log_dir() == tmp_path / "logs"

    explicit = tmp_path / "other.db"
    monkeypatch.setenv("AURAMAUR_DB_PATH", str(explicit))
    assert runtime.db_path() == explicit
    assert Database().db_path == str(explicit)


def test_explicit_database_path_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("AURAMAUR_DB_PATH", str(tmp_path / "runtime.db"))
    assert Database(":memory:").db_path == ":memory:"


def test_log_file_path_follows_logging_setting(tmp_path, monkeypatch):
    # Container deployments point the structlog file elsewhere via the
    # LOGGING__FILE env override; log_file_path must follow it.
    override = tmp_path / "logs" / "auramaur.log"
    monkeypatch.setenv("LOGGING__FILE", str(override))
    assert runtime.log_file_path() == override


def test_log_file_path_matches_settings_default(monkeypatch):
    from pathlib import Path

    from config.settings import Settings

    monkeypatch.delenv("LOGGING__FILE", raising=False)
    # Whatever the effective logging.file setting is (defaults.yaml or local
    # overrides), log_file_path must agree with it — that is where the bot
    # actually writes.
    assert runtime.log_file_path() == Path(Settings().logging.file).expanduser()
