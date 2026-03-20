"""Structured logging setup using structlog."""

from __future__ import annotations

import logging
import sys

import structlog


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
                logger_factory=structlog.WriteLoggerFactory(file=open(log_file, "a")),
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
