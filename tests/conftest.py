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
import os
import sys

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
    for obj in gc.get_objects():
        if type(obj) is aiosqlite.Connection:
            try:
                obj._running = False
            except Exception:  # pragma: no cover
                pass
