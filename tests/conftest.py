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
tests should still close their databases, and the stderr line below names
the leak count so regressions stay visible.
"""

from __future__ import annotations

import gc
import os
import sys
import threading

import pytest

# Tests exercise order paths while an operator may intentionally have the real
# repository kill switch armed. Keep configuration and safety-file state local
# to each test process; dedicated kill-switch tests override the path directly.
os.environ["AURAMAUR_LOCAL_CONFIG"] = "/tmp/auramaur-test-no-local-config.yaml"


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


def pytest_sessionfinish(session, exitstatus):
    try:
        import aiosqlite
    except ImportError:  # pragma: no cover
        return
    stranded: list = []
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
        try:
            if hasattr(obj, "stop"):  # >= 0.22: sentinel-based, loop-safe
                obj.stop()
            else:  # <= 0.21: poll loop honors the flag on its next tick
                obj._running = False
            stranded.append(thread)
        except Exception:  # pragma: no cover
            pass
    for thread in stranded:
        thread.join(timeout=2.0)
    if stranded:
        print(
            f"\n[conftest] reaped {len(stranded)} unclosed aiosqlite "
            "connection(s) at session end — tests should close their "
            "databases",
            file=sys.stderr,
        )
