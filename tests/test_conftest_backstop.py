"""The sessionfinish backstop must actually stop a stranded aiosqlite
worker under the INSTALLED aiosqlite version.

The stop mechanism is version-dependent (see conftest.py's docstring):
the 0.17-era flag-flip became a silent no-op when 0.22.1 landed, and the
failure mode is not a red test but a green suite that hangs the
interpreter until CI's 6h job timeout. This test pins the contract to
whatever version the lockfile installs, so the next bump that changes
the worker's shutdown protocol fails here in seconds instead.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import aiosqlite
import pytest

from tests import conftest


def test_sessionfinish_marks_a_leaking_suite_failed(monkeypatch, capsys):
    stranded = [(SimpleNamespace(), "tests/test_owner.py::test_leak", "stack")]
    monkeypatch.setattr(
        conftest, "_reap_stranded_connections", lambda: stranded)
    session = SimpleNamespace(exitstatus=pytest.ExitCode.OK)

    conftest.pytest_sessionfinish(session=session, exitstatus=0)

    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert "tests/test_owner.py::test_leak" in capsys.readouterr().err


def test_sessionfinish_backstop_reaps_a_stranded_worker_thread():
    async def leak() -> aiosqlite.Connection:
        conn = await aiosqlite.connect(":memory:")
        await conn.execute("SELECT 1")
        return conn

    # asyncio.run closes its loop on return: the connection's worker is
    # now exactly the stranded, loop-less thread CI hung on.
    conn = asyncio.run(leak())
    thread = getattr(conn, "_thread", None)
    if thread is None and isinstance(conn, threading.Thread):
        thread = conn  # <= 0.21: Connection IS the thread
    assert thread is not None and thread.is_alive()

    stranded = conftest._reap_stranded_connections()

    thread.join(timeout=3.0)
    assert len(stranded) == 1
    assert not thread.is_alive(), (
        "backstop failed to stop a stranded aiosqlite worker — under this "
        "aiosqlite version the suite would hang at interpreter exit"
    )
