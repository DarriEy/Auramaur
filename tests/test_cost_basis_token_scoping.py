"""Tests for cost_basis token scoping and the v20 casing-merge migration.

Locks in the fix for the token-blind cost-basis reads:

  1. ``TokenType.from_str`` normalizes any-case outcome strings to canonical
     "YES"/"NO" (the reconciler wrote raw CLOB "Yes"/"No", everything else
     wrote enum "YES"/"NO", which split one position into two PK rows).
  2. ``get_cost_basis`` / ``get_token_info`` disambiguate deterministically
     (largest open size, then most recent) instead of returning an arbitrary
     row when a market has more than one token row.
  3. The v19->v20 migration collapses casing-split duplicates into one
     canonical row (newest write wins for size/avg_cost, realized_pnl summed),
     preserves genuine two-sided positions, and keeps a reversible backup.
"""

from __future__ import annotations

import asyncio

from auramaur.broker.pnl import PnLTracker
from auramaur.db.database import Database
from auramaur.exchange.models import TokenType
from config.settings import Settings


def test_tokentype_from_str_normalizes_casing():
    assert TokenType.from_str("Yes") is TokenType.YES
    assert TokenType.from_str("yes") is TokenType.YES
    assert TokenType.from_str("YES") is TokenType.YES
    assert TokenType.from_str("No") is TokenType.NO
    assert TokenType.from_str("no") is TokenType.NO
    assert TokenType.from_str("NO") is TokenType.NO
    assert TokenType.from_str(" no ") is TokenType.NO
    # Unknown / empty defaults to YES (the historical default).
    assert TokenType.from_str("") is TokenType.YES
    assert TokenType.from_str(None) is TokenType.YES
    assert TokenType.from_str("Something") is TokenType.YES
    # .value is the canonical persisted form.
    assert TokenType.from_str("No").value == "NO"


async def _insert_cb(db, market_id, token, size, avg_cost, is_paper,
                     realized_pnl=0.0, updated_at="2026-01-01T00:00:00+00:00"):
    await db.execute(
        "INSERT INTO cost_basis (market_id, token, token_id, size, avg_cost,"
        " total_cost, realized_pnl, is_paper, updated_at)"
        " VALUES (?, ?, '', ?, ?, ?, ?, ?, ?)",
        (market_id, token, size, avg_cost, size * avg_cost, realized_pnl,
         is_paper, updated_at),
    )
    await db.commit()


def test_getters_disambiguate_two_sided_position():
    async def run():
        db = Database(":memory:")
        await db.connect()
        tracker = PnLTracker(db, Settings())
        mode = tracker._mode_flag()

        # A genuine two-sided position: small NO + large YES in the same mode.
        await _insert_cb(db, "m1", "NO", 5.0, 0.30, mode)
        await _insert_cb(db, "m1", "YES", 40.0, 0.62, mode)

        avg_cost, size = await tracker.get_cost_basis("m1")
        token, _ = await tracker.get_token_info("m1")
        # Largest open size wins, and the token returned matches it.
        assert size == 40.0
        assert abs(avg_cost - 0.62) < 1e-9
        assert token is TokenType.YES

    asyncio.run(run())


def test_getters_are_case_insensitive_for_legacy_rows():
    async def run():
        db = Database(":memory:")
        await db.connect()
        tracker = PnLTracker(db, Settings())
        mode = tracker._mode_flag()

        # A legacy title-case row (as the reconciler used to write) must still
        # parse to a valid TokenType rather than raising.
        await _insert_cb(db, "m2", "No", 12.0, 0.44, mode)
        token, _ = await tracker.get_token_info("m2")
        assert token is TokenType.NO

    asyncio.run(run())


def test_v19_to_v20_merges_casing_splits():
    async def run():
        db = Database(":memory:")
        await db.connect()

        # Casing-split duplicate of ONE position (live): stale "NO" from the
        # fill path + fresh "No" from the reconciler (ground-truth price).
        await _insert_cb(db, "split", "NO", 35.0, 0.835, 0,
                         realized_pnl=1.5, updated_at="2026-06-01T00:00:00+00:00")
        await _insert_cb(db, "split", "No", 35.0, 0.766, 0,
                         realized_pnl=0.0, updated_at="2026-06-23T00:00:00+00:00")
        # A genuine two-sided position must be preserved as two rows.
        await _insert_cb(db, "twosided", "YES", 20.0, 0.40, 0)
        await _insert_cb(db, "twosided", "NO", 10.0, 0.55, 0)
        # Same market id, different mode — must stay independent.
        await _insert_cb(db, "split", "Yes", 8.0, 0.20, 1)

        await db._migrate_v19_to_v20()

        # The casing split collapsed to ONE canonical NO row, newest wins.
        rows = await db.fetchall(
            "SELECT token, size, avg_cost, realized_pnl FROM cost_basis"
            " WHERE market_id = 'split' AND is_paper = 0"
        )
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["token"] == "NO"
        assert abs(row["avg_cost"] - 0.766) < 1e-9          # reconciler ground truth
        assert abs(row["realized_pnl"] - 1.5) < 1e-9        # realized history summed, not lost

        # No non-canonical casing remains anywhere.
        bad = await db.fetchall(
            "SELECT 1 FROM cost_basis WHERE token NOT IN ('YES', 'NO')"
        )
        assert bad == []

        # Genuine two-sided position preserved as two rows.
        ts = await db.fetchall(
            "SELECT token FROM cost_basis WHERE market_id = 'twosided' ORDER BY token"
        )
        assert [dict(r)["token"] for r in ts] == ["NO", "YES"]

        # Paper-mode row for the same id is untouched (and recased).
        paper = await db.fetchall(
            "SELECT token, size FROM cost_basis WHERE market_id = 'split' AND is_paper = 1"
        )
        assert len(paper) == 1
        assert dict(paper[0])["token"] == "YES"

        # Reversible backup holds every pre-migration row.
        backup = await db.fetchall("SELECT count(*) AS n FROM cost_basis_backup_v20")
        assert dict(backup[0])["n"] == 5

    asyncio.run(run())
