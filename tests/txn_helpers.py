"""Shared scaffolding for transaction-adoption tests (#353, phase 1).

Two instruments, both restore the patched method on exit:

* ``failing_on(db, needle)`` — makes ``Database.execute`` raise
  ``sqlite3.OperationalError("injected: ...")`` for any statement
  containing *needle*. Point it at the SECOND statement of a batch and
  the batch's atomicity contract becomes testable: after the failure,
  zero rows from ANY statement of the batch may exist.
* ``transaction_spy(db, events)`` — records ``(owner, "begin"/"end")``
  around every ``Database.transaction`` span, so tests can assert the
  writer's distinct owner label and that no awaited boundary (network,
  LLM) fires between begin and end.

These extend the idioms in ``tests/test_database_transaction.py``; the
crash-mid-batch pattern is the direct analogue of the injected-
SQLITE_BUSY test the db-contention plan specified for record_fill.
"""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager, contextmanager


@contextmanager
def failing_on(db, needle: str):
    """Raise on the first ``execute()`` whose SQL contains *needle*."""
    original = db.execute

    async def _wrapped(sql: str, params: tuple = ()):
        if needle in sql:
            raise sqlite3.OperationalError(f"injected: {needle}")
        return await original(sql, params)

    db.execute = _wrapped
    try:
        yield
    finally:
        db.execute = original


@contextmanager
def transaction_spy(db, events: list):
    """Append ``(owner, "begin")`` / ``(owner, "end")`` around each span.

    Joined (re-entrant) spans are invisible here by design — they never
    reach the patched entry point's BEGIN, exactly as in production.
    """
    original = db.transaction

    @asynccontextmanager
    async def _spy(owner: str | None = None):
        events.append((owner or "", "begin"))
        try:
            async with original(owner=owner) as handle:
                yield handle
        finally:
            events.append((owner or "", "end"))

    db.transaction = _spy
    try:
        yield
    finally:
        db.transaction = original


def span_owners(events: list) -> list:
    """The distinct owner labels that opened spans, in first-seen order."""
    seen: list = []
    for owner, phase in events:
        if phase == "begin" and owner not in seen:
            seen.append(owner)
    return seen
