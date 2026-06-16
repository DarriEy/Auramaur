"""Tests for correlation-scan window rotation + resolved-relationship pruning.

The relationship detector used to re-feed only the top-by-volume head every
cycle, so niche conditional/entailment pairs were never discovered, and
market_relationships was never pruned, so it filled with resolved markets.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from auramaur.bot import AuramaurBot
from auramaur.db.database import Database


# ---------------------------------------------------------------------------
# _rotate_scan_window — pure rolling window
# ---------------------------------------------------------------------------

def test_rotate_window_advances_and_wraps():
    u = list(range(25))
    w1, off1 = AuramaurBot._rotate_scan_window(u, 0, 10)
    assert w1 == list(range(0, 10)) and off1 == 10
    w2, off2 = AuramaurBot._rotate_scan_window(u, off1, 10)
    assert w2 == list(range(10, 20)) and off2 == 20
    # Past the end → tail slice, offset advances beyond len
    w3, off3 = AuramaurBot._rotate_scan_window(u, off2, 10)
    assert w3 == list(range(20, 25)) and off3 == 30
    # Next call wraps back to the start
    w4, off4 = AuramaurBot._rotate_scan_window(u, off3, 10)
    assert w4 == list(range(0, 10)) and off4 == 10


def test_rotate_window_empty_universe():
    assert AuramaurBot._rotate_scan_window([], 50, 10) == ([], 0)


# ---------------------------------------------------------------------------
# _prune_resolved_relationships — drop pairs with a resolved leg
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prune_drops_relationships_with_resolved_leg():
    db = Database(":memory:")
    await db.connect()
    try:
        # Two live markets, one resolved (active=0).
        for mid, active in (("LIVE_A", 1), ("LIVE_B", 1), ("DEAD", 0)):
            await db.execute(
                """INSERT INTO markets (id, exchange, question, category, active,
                   outcome_yes_price, outcome_no_price, last_updated)
                   VALUES (?, 'polymarket', 'q', 'other', ?, 0.5, 0.5, datetime('now'))""",
                (mid, active),
            )
        # Relationship between two live markets (keep) + one touching the dead one (drop).
        await db.execute(
            """INSERT INTO market_relationships
               (market_id_a, market_id_b, relationship_type, strength, description, detected_at)
               VALUES ('LIVE_A','LIVE_B','conditional',1.0,'x',datetime('now'))"""
        )
        await db.execute(
            """INSERT INTO market_relationships
               (market_id_a, market_id_b, relationship_type, strength, description, detected_at)
               VALUES ('LIVE_A','DEAD','conditional',1.0,'x',datetime('now'))"""
        )
        await db.commit()

        bot = AuramaurBot(settings=MagicMock())
        removed = await bot._prune_resolved_relationships(db)

        assert removed == 1
        remaining = await db.fetchall("SELECT market_id_a, market_id_b FROM market_relationships")
        assert [(r["market_id_a"], r["market_id_b"]) for r in remaining] == [("LIVE_A", "LIVE_B")]
    finally:
        await db.close()
