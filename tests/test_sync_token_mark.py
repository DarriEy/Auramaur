"""Position marks must price the token actually held.

Outcome labels like "Something"/"Nothing" aren't YES/NO, and the syncer's
old YES-default marked such holdings at the first outcome's price: the
Obama "Something" token (worth ~$0.105) was marked at "Nothing"'s $0.885 —
a phantom +$77 whose fantasy-priced exit the bot chased for two days.

Side resolution ladder: token_id match against the market's CLOB token ids
(authoritative) → literal YES/NO label → price the held token off its own
book.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from auramaur.broker.sync import PositionSyncer
from auramaur.db.database import Database
from auramaur.exchange.models import OrderBook, OrderBookLevel, TokenType


async def _seed_market(db: Database, market_id: str, yes_price: float,
                       clob_yes: str = "", clob_no: str = "") -> None:
    await db.execute(
        """INSERT INTO markets (id, question, outcome_yes_price, outcome_no_price,
                                clob_token_yes, clob_token_no, last_updated)
           VALUES (?, 'Q?', ?, ?, ?, ?, datetime('now'))""",
        (market_id, yes_price, 1.0 - yes_price, clob_yes, clob_no),
    )


async def _seed_holding(db: Database, market_id: str, token_label: str,
                        token_id: str, size: float = 100.0, avg_cost: float = 0.126) -> None:
    await db.execute(
        """INSERT INTO cost_basis (market_id, token, token_id, size, avg_cost, total_cost, is_paper)
           VALUES (?, ?, ?, ?, ?, ?, 0)""",
        (market_id, token_label, token_id, size, avg_cost, size * avg_cost),
    )


def _syncer(db: Database, book: OrderBook | None = None) -> PositionSyncer:
    syncer = PositionSyncer.__new__(PositionSyncer)
    syncer._settings = MagicMock()
    syncer._db = db
    syncer._exchange = SimpleNamespace(
        get_order_book=AsyncMock(return_value=book if book is not None else OrderBook()),
    )
    syncer._paper = MagicMock()
    syncer._pnl = MagicMock()
    return syncer


def test_token_id_match_beats_label_default():
    """Held token matches clob_token_no -> NO side, NO price — even though
    the label ("Something") would have defaulted to YES before."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            await _seed_market(db, "m1", yes_price=0.895,
                               clob_yes="tok-nothing", clob_no="tok-something")
            await _seed_holding(db, "m1", "Something", "tok-something")
            await db.commit()

            positions = await _syncer(db)._sync_live()
            assert len(positions) == 1
            assert positions[0].token == TokenType.NO
            assert abs(positions[0].current_price - 0.105) < 1e-9
        finally:
            await db.close()

    asyncio.run(run())


def test_unknown_label_priced_from_own_book():
    """No CLOB token ids stored and a non-YES/NO label -> mark from the held
    token's own book midpoint, not the first outcome's price."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            await _seed_market(db, "m1", yes_price=0.895)
            await _seed_holding(db, "m1", "Something", "tok-something")
            await db.commit()

            book = OrderBook(
                bids=[OrderBookLevel(price=0.07, size=100)],
                asks=[OrderBookLevel(price=0.14, size=100)],
            )
            positions = await _syncer(db, book)._sync_live()
            assert len(positions) == 1
            assert abs(positions[0].current_price - 0.105) < 1e-9
        finally:
            await db.close()

    asyncio.run(run())


def test_plain_yes_label_unchanged():
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            await _seed_market(db, "m1", yes_price=0.60)
            await _seed_holding(db, "m1", "YES", "tok-1")
            await db.commit()

            positions = await _syncer(db)._sync_live()
            assert len(positions) == 1
            assert positions[0].token == TokenType.YES
            assert abs(positions[0].current_price - 0.60) < 1e-9
        finally:
            await db.close()

    asyncio.run(run())


def test_unknown_label_empty_book_falls_back_to_yes_price():
    """Self-heal unavailable (no book at all) -> old behavior, with a warning
    logged, rather than a zero mark."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            await _seed_market(db, "m1", yes_price=0.895)
            await _seed_holding(db, "m1", "Something", "tok-something")
            await db.commit()

            positions = await _syncer(db, OrderBook())._sync_live()
            assert len(positions) == 1
            assert abs(positions[0].current_price - 0.895) < 1e-9
        finally:
            await db.close()

    asyncio.run(run())


def test_markets_table_has_clob_token_columns():
    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            cols = {r["name"] for r in await db.fetchall("SELECT name FROM pragma_table_info('markets')")}
            assert "clob_token_yes" in cols
            assert "clob_token_no" in cols
        finally:
            await db.close()

    asyncio.run(run())
