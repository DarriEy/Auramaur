"""Portable runtime paths shared by native and container deployments."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def state_dir() -> Path:
    return _path_from_env("AURAMAUR_STATE_DIR", REPO_ROOT)


def db_path() -> Path:
    return _path_from_env("AURAMAUR_DB_PATH", state_dir() / "auramaur.db")


def log_dir() -> Path:
    return _path_from_env("AURAMAUR_LOG_DIR", REPO_ROOT / "logs")


def log_file_path() -> Path:
    """Structlog file location — where the running bot actually writes.

    Resolved exactly the way logging setup does: the ``logging.file`` setting
    (``LOGGING__FILE`` env var, then local/tracked YAML, then the pydantic
    default ``auramaur.log``). Readers such as the readiness checks must use
    this instead of hardcoding a repo-root/CWD default, or a container run
    (which sets ``LOGGING__FILE=/app/logs/auramaur.log``) looks in the wrong
    place.
    """
    from config.settings import Settings  # local import keeps this module light

    return Path(Settings().logging.file).expanduser()


def kill_switch_path() -> Path:
    return _path_from_env("AURAMAUR_KILL_SWITCH_PATH", state_dir() / "KILL_SWITCH")
