"""Shared test configuration.

Interpreter-exit fix: tests that build a Database(":memory:") and skip
``await db.close()`` — including any test that FAILS an assertion before
its close line — leak the aiosqlite Connection worker thread. Those
threads are non-daemon, so pytest finished in ~7s but the interpreter hung
forever at exit (diagnosed 2026-06-09: six lingering
``aiosqlite.core.Connection`` threads blocked on their command queues).

The hook below stops any surviving aiosqlite worker threads at session
end. 2026-08-04: the stop mechanism is version-dependent, and getting it
wrong hangs CI until the 6h job timeout — under aiosqlite <= 0.21 the
worker polls its queue on a 0.1s tick and honors ``_running = False``;
under >= 0.22 it blocks on an untimed queue get and only exits via the
sentinel that ``Connection.stop()`` enqueues (sync-safe by design — it is
what ``__del__`` uses). The old flag-flip alone became a silent no-op the
day the 0.17 -> 0.22.1 bump landed. This is a backstop, not a license —
tests must still close their databases. A leak now makes the session fail
after cleanup and reports both its owning test and construction stack.
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import traceback

import pytest

_AIOSQLITE_CREATION_STACKS: dict[int, tuple[str, str]] = {}
_ORIGINAL_AIOSQLITE_CONNECT = None
_CURRENT_TEST = "<pytest session>"

# Tests exercise order paths while an operator may intentionally have the real
# repository kill switch armed. Keep configuration and safety-file state local
# to each test process; dedicated kill-switch tests override the path directly.
os.environ["AURAMAUR_LOCAL_CONFIG"] = "/tmp/auramaur-test-no-local-config.yaml"


def pytest_configure():
    """Record where every test-owned aiosqlite connection was constructed."""
    global _ORIGINAL_AIOSQLITE_CONNECT
    import aiosqlite

    if _ORIGINAL_AIOSQLITE_CONNECT is not None:
        return
    _ORIGINAL_AIOSQLITE_CONNECT = aiosqlite.connect

    def tracked_connect(*args, **kwargs):
        connection = _ORIGINAL_AIOSQLITE_CONNECT(*args, **kwargs)
        _AIOSQLITE_CREATION_STACKS[id(connection)] = (
            _CURRENT_TEST,
            "".join(traceback.format_stack(limit=12)),
        )
        return connection

    aiosqlite.connect = tracked_connect


def pytest_unconfigure():
    """Restore the library function for in-process pytest embedders."""
    global _ORIGINAL_AIOSQLITE_CONNECT
    if _ORIGINAL_AIOSQLITE_CONNECT is None:
        return
    import aiosqlite

    aiosqlite.connect = _ORIGINAL_AIOSQLITE_CONNECT
    _ORIGINAL_AIOSQLITE_CONNECT = None


def pytest_runtest_setup(item):
    global _CURRENT_TEST
    _CURRENT_TEST = item.nodeid


@pytest.fixture(autouse=True)
def _isolate_kill_switch(monkeypatch, tmp_path):
    import auramaur.killswitch as killswitch

    monkeypatch.setattr(killswitch, "KILL_SWITCH_PATH", tmp_path / "KILL_SWITCH")
    # Exchange modules import the function directly, so patch those bound names
    # without changing the function exercised by tests/test_killswitch.py.
    for module in tuple(sys.modules.values()):
        if (module is not None and module is not killswitch
                and getattr(module, "__dict__", {}).get("kill_switch_present")
                is killswitch.kill_switch_present):
            monkeypatch.setattr(module, "kill_switch_present", lambda: False)


def _reap_stranded_connections() -> list[tuple[threading.Thread, str, str]]:
    try:
        import aiosqlite
    except ImportError:  # pragma: no cover
        return []
    stranded: list[tuple[threading.Thread, str, str]] = []
    for obj in gc.get_objects():
        if type(obj) is not aiosqlite.Connection:
            continue
        # Only connections whose worker is still parked matter; properly
        # closed connections may linger in gc but their thread is gone.
        thread = getattr(obj, "_thread", None)
        if thread is None and isinstance(obj, threading.Thread):
            thread = obj  # <= 0.21: Connection IS the thread
        if thread is None or not thread.is_alive():
            continue
        # close() clears the underlying sqlite handle before the worker exits.
        # On polling-worker releases (notably 0.17), that thread can remain
        # alive for one final 100ms tick. It is closing, not leaked.
        if getattr(obj, "_connection", object()) is None:
            thread.join(timeout=0.25)
            continue
        try:
            if hasattr(obj, "stop"):  # >= 0.22: sentinel-based, loop-safe
                obj.stop()
            else:  # <= 0.21: poll loop honors the flag on its next tick
                obj._running = False
            owner, stack = _AIOSQLITE_CREATION_STACKS.get(
                id(obj),
                ("<test owner unavailable>",
                 "<connection creation stack unavailable>"),
            )
            stranded.append((thread, owner, stack))
        except Exception:  # pragma: no cover
            pass
    for thread, _, _ in stranded:
        thread.join(timeout=2.0)
    return stranded


def pytest_sessionfinish(session, exitstatus):
    stranded = _reap_stranded_connections()
    if stranded:
        details = "\n\n".join(
            f"unclosed aiosqlite connection #{index} owned by {owner}, "
            f"created at:\n{stack}"
            for index, (_, owner, stack) in enumerate(stranded, start=1)
        )
        print(
            f"\n[conftest] reaped {len(stranded)} unclosed aiosqlite "
            f"connection(s) at session end:\n\n{details}",
            file=sys.stderr,
        )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
