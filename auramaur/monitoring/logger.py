"""Structured logging setup using structlog."""

from __future__ import annotations

import logging
import os
import sys

import structlog


class _RotatingWriter:
    """Size-based rotating file writer for structlog's WriteLoggerFactory.

    WriteLoggerFactory needs only a .write()/.flush() file-like object, so a
    stdlib RotatingFileHandler can't be plugged in directly. This keeps the
    existing factory (and thus the interactive console behavior) untouched
    while bounding the log file: when the active file passes ``max_bytes`` it
    is rolled to ``<path>.1``, older backups shifting up to ``<path>.<backups>``.
    """

    def __init__(self, path: str, max_bytes: int = 50 * 1024 * 1024, backups: int = 5):
        self._path = path
        self._max_bytes = max_bytes
        self._backups = backups
        self._file = open(path, "a", encoding="utf-8")

    def write(self, data: str) -> int:
        n = self._file.write(data)
        if self._file.tell() >= self._max_bytes:
            self._rotate()
        return n

    def flush(self) -> None:
        self._file.flush()

    def _rotate(self) -> None:
        self._file.flush()
        self._file.close()
        for i in range(self._backups - 1, 0, -1):
            src, dst = f"{self._path}.{i}", f"{self._path}.{i + 1}"
            if os.path.exists(src):
                os.replace(src, dst)
        if os.path.exists(self._path):
            os.replace(self._path, f"{self._path}.1")
        self._file = open(self._path, "a", encoding="utf-8")


def setup_logging(level: str = "INFO", json_format: bool = True, log_file: str | None = None):
    """Configure structlog for the application.

    When json_format=False (interactive terminal), structlog output goes to the
    log file only so it doesn't clutter the Rich console output.  The console
    receives only the curated Rich display calls.
    """

    log_level = getattr(logging, level, logging.INFO)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_format:
        # JSON to stdout (e.g. running as a service)
        structlog.configure(
            processors=[*shared_processors, structlog.processors.JSONRenderer()],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Interactive mode: structlog → file only, Rich handles the console
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter("%(message)s"))

            structlog.configure(
                processors=[*shared_processors, structlog.processors.JSONRenderer()],
                wrapper_class=structlog.make_filtering_bound_logger(log_level),
                context_class=dict,
                logger_factory=structlog.WriteLoggerFactory(file=_RotatingWriter(log_file)),
                cache_logger_on_first_use=True,
            )
        else:
            # No file, suppress structlog entirely in interactive mode
            structlog.configure(
                processors=[*shared_processors, structlog.processors.JSONRenderer()],
                wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
                context_class=dict,
                logger_factory=structlog.PrintLoggerFactory(),
                cache_logger_on_first_use=True,
            )

    # Silence stdlib logging from third-party libs in interactive mode
    stderr_level = log_level if json_format else logging.WARNING
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=stderr_level,
    )
