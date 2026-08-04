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

import aiosqlite

from tests import conftest


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

    conftest.pytest_sessionfinish(session=None, exitstatus=0)

    thread.join(timeout=3.0)
    assert not thread.is_alive(), (
        "backstop failed to stop a stranded aiosqlite worker — under this "
        "aiosqlite version the suite would hang at interpreter exit"
    )
