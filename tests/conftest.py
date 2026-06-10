"""Shared test configuration.

Interpreter-exit fix: tests that build a Database(":memory:") and skip
``await db.close()`` — including any test that FAILS an assertion before
its close line — leak the aiosqlite Connection worker thread. Those
threads are non-daemon, so pytest finished in ~7s but the interpreter hung
forever at exit (diagnosed 2026-06-09: six lingering
``aiosqlite.core.Connection`` threads blocked on their command queues).

The hook below stops any surviving aiosqlite connections at session end:
``Connection.run()`` exits its loop on the next 0.1s queue-poll tick once
``_running`` is False. This is a backstop, not a license — tests should
still close their databases.
"""

from __future__ import annotations

import gc


def pytest_sessionfinish(session, exitstatus):
    try:
        import aiosqlite
    except ImportError:  # pragma: no cover
        return
    for obj in gc.get_objects():
        if isinstance(obj, aiosqlite.Connection):
            try:
                obj._running = False
            except Exception:  # pragma: no cover
                pass
