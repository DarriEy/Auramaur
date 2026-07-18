"""Coinbase read-only shadow book tests."""

from types import SimpleNamespace

import pytest

from auramaur.db.database import Database
from auramaur.exchange.coinbase import CoinbaseQuote
from auramaur.exchange.models import OrderSide
from auramaur.treasury.coinbase_paper import CoinbasePaperBook
from config.settings import Settings


class FakeCoinbase:
    async def get_quote(self, product):
        assert product == "BTC-USD"
        return CoinbaseQuote(bid=99.0, ask=101.0)


@pytest.mark.asyncio
async def test_shadow_book_uses_ask_bid_and_keeps_separate_attribution():
    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    book = CoinbasePaperBook(settings, FakeCoinbase(), db)

    await book.record_shadow_fill("XBTUSDC", OrderSide.BUY, 0.1)
    await book.record_shadow_fill("XBTUSDC", OrderSide.SELL, 0.1)

    fills = await db.fetchall(
        "SELECT market_id, side, price, fee, is_paper FROM fills ORDER BY id")
    assert [(r["side"], r["price"]) for r in fills] == [("BUY", 101.0), ("SELL", 99.0)]
    assert all(r["market_id"] == "coinbase:BTC-USD" for r in fills)
    assert all(r["is_paper"] == 1 for r in fills)
    assert fills[0]["fee"] == pytest.approx(10.1 * 0.006)

    ledger = await db.fetchone(
        "SELECT venue, category, strategy_source, pnl FROM pnl_ledger")
    assert dict(ledger) | {} == {
        "venue": "coinbase", "category": "coinbase_spot",
        "strategy_source": "coinbase_paper",
        "pnl": pytest.approx((99.0 - 101.0) * 0.1 - 9.9 * 0.006),
    }
    await db.close()


@pytest.mark.asyncio
async def test_unknown_pair_is_ignored_without_market_call():
    client = SimpleNamespace(get_quote=None)
    book = CoinbasePaperBook(Settings(), client, None)
    await book.record_shadow_fill("DOGEUSDC", OrderSide.BUY, 10.0)
