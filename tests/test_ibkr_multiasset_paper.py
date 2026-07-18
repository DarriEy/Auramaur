import time

import pytest

from auramaur.db.database import Database
from auramaur.exchange.ibkr_instruments import ContractKind, IBKRBook
from auramaur.exchange.ibkr_market_data import MarketDataQuote
from auramaur.strategy.ibkr_multiasset_paper import IBKRMultiAssetPaperBook
from config.settings import Settings


class FakeMarketData:
    async def get_quote(self, spec):
        price = 2.0 if spec.kind is ContractKind.OPTION else 100.0
        return MarketDataQuote(spec.key, price, price * 1.0001, time.time(),
                               abs(hash(spec.key)) % 100000 + 1,
                               spec.currency, spec.multiplier)

    async def get_daily_bars(self, spec, duration="3 M"):
        return [(f"2026-06-{day:02d}", 80 + day) for day in range(1, 22)]

    async def get_fx_to_usd(self, currency):
        return 1.0


@pytest.mark.asyncio
async def test_all_six_books_write_only_isolated_paper_tables():
    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.multiasset_paper_enabled = True
    settings.ibkr.multiasset_refreshes_per_cycle = 1
    for cfg in settings.ibkr.multiasset_books.values():
        cfg.max_position_pct = 40
        cfg.max_deployment_pct = 60
    for book in IBKRBook:
        pillar = IBKRMultiAssetPaperBook(settings, FakeMarketData(), db, book)
        pillar.market_open = lambda now=None: True
        assert await pillar.run_once() == 1
    rows = await db.fetchall(
        "SELECT DISTINCT book FROM ibkr_paper_positions ORDER BY book")
    assert {row["book"] for row in rows} == {book.value for book in IBKRBook}
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM ibkr_paper_fills"))["n"] == 6
    # The shared prediction-market wallet is not touched.
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM cost_basis"))["n"] == 0
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM pnl_ledger"))["n"] == 0
    await db.close()


@pytest.mark.asyncio
async def test_daily_loss_gate_blocks_entries():
    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.multiasset_paper_enabled = True
    await db.execute(
        "INSERT INTO ibkr_paper_ledger (book, kind, pnl_usd, source_ref) "
        "VALUES ('global_etf', 'trade', -101, 'loss')")
    await db.commit()
    pillar = IBKRMultiAssetPaperBook(
        settings, FakeMarketData(), db, IBKRBook.GLOBAL_ETF)
    pillar.market_open = lambda now=None: True
    assert await pillar.run_once() == 0
    assert (await db.fetchone(
        "SELECT COUNT(*) AS n FROM ibkr_paper_positions"))["n"] == 0
    await db.close()


def test_books_declare_paper_simulated_mode_and_asset_calendars():
    settings = Settings()
    for book in IBKRBook:
        pillar = IBKRMultiAssetPaperBook(settings, None, None, book)
        assert pillar.execution_mode.value == "paper_simulated"
