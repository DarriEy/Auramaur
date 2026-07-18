import time
from datetime import datetime, timezone

import pytest

from auramaur.db.database import Database
from auramaur.exchange.ibkr_instruments import ContractKind, IBKRBook
from auramaur.exchange.ibkr_market_data import MarketDataQuote
from auramaur.strategy.ibkr_multiasset_paper import IBKRMultiAssetPaperBook
from config.settings import Settings


class FakeMarketData:
    con_ids_requested = []

    async def get_quote(self, spec):
        price = 2.0 if spec.kind is ContractKind.OPTION else 100.0
        return MarketDataQuote(spec.key, price, price * 1.0001, time.time(),
                               abs(hash(spec.key)) % 100000 + 1,
                               spec.currency, spec.multiplier)

    async def get_daily_bars(self, spec, duration="3 M"):
        return [(f"2026-06-{day:02d}", 80 + day) for day in range(1, 22)]

    async def get_fx_to_usd(self, currency):
        return 1.0

    async def get_quote_by_con_id(self, spec, con_id):
        self.con_ids_requested.append(con_id)
        quote = await self.get_quote(spec)
        return MarketDataQuote(quote.key, quote.bid, quote.ask, quote.timestamp,
                               con_id, quote.currency, quote.multiplier,
                               source="held_contract")


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
    assert (await db.fetchone(
        "SELECT COUNT(*) AS n FROM ibkr_paper_fills WHERE price_source='ibkr'"))["n"] == 6
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


@pytest.mark.asyncio
async def test_intracycle_commission_tightens_loss_gate():
    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.multiasset_paper_enabled = True
    settings.ibkr.multiasset_refreshes_per_cycle = 2
    cfg = settings.ibkr.multiasset_books["global_etf"]
    cfg.daily_loss_limit_usd = 0.5
    cfg.max_positions = 2
    pillar = IBKRMultiAssetPaperBook(
        settings, FakeMarketData(), db, IBKRBook.GLOBAL_ETF)
    pillar.market_open = lambda now=None: True
    assert await pillar.run_once() == 1
    assert (await db.fetchone(
        "SELECT COUNT(*) AS n FROM ibkr_paper_positions"))["n"] == 1
    await db.close()


@pytest.mark.asyncio
async def test_stale_quote_cannot_create_fill():
    class StaleMarketData(FakeMarketData):
        async def get_quote(self, spec):
            quote = await super().get_quote(spec)
            return MarketDataQuote(quote.key, quote.bid, quote.ask, time.time() - 3600,
                                   quote.con_id, quote.currency, quote.multiplier)

    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.multiasset_paper_enabled = True
    settings.ibkr.multiasset_refreshes_per_cycle = 1
    pillar = IBKRMultiAssetPaperBook(
        settings, StaleMarketData(), db, IBKRBook.GLOBAL_ETF)
    pillar.market_open = lambda now=None: True
    assert await pillar.run_once() == 0
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM ibkr_paper_fills"))["n"] == 0
    await db.close()


@pytest.mark.asyncio
async def test_open_position_is_marked_by_original_contract_id():
    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.multiasset_paper_enabled = True
    settings.ibkr.multiasset_refreshes_per_cycle = 1
    client = FakeMarketData()
    client.con_ids_requested = []
    pillar = IBKRMultiAssetPaperBook(
        settings, client, db, IBKRBook.FUTURES)
    pillar.market_open = lambda now=None: True
    settings.ibkr.multiasset_books["futures"].max_position_pct = 40
    await pillar.run_once()
    row = await db.fetchone(
        "SELECT con_id FROM ibkr_paper_positions WHERE book='futures'")
    assert row is not None
    await pillar.run_once()
    assert int(row["con_id"]) in client.con_ids_requested
    await db.close()


def test_books_declare_paper_simulated_mode_and_asset_calendars():
    settings = Settings()
    for book in IBKRBook:
        pillar = IBKRMultiAssetPaperBook(settings, None, None, book)
        assert pillar.execution_mode.value == "paper_simulated"


def test_option_fallback_pricer_respects_intrinsic_value():
    from auramaur.exchange.ibkr_market_data import IBKRReadOnlyMarketData

    call = IBKRReadOnlyMarketData._black_scholes(110, 100, 30 / 365, 0.20, "C")
    put = IBKRReadOnlyMarketData._black_scholes(90, 100, 30 / 365, 0.20, "P")
    assert call >= 10
    assert put >= 9.5


def test_futures_calendar_closes_friday_and_reopens_sunday_evening():
    pillar = IBKRMultiAssetPaperBook(
        Settings(), None, None, IBKRBook.FUTURES)
    assert not pillar.market_open(datetime(2026, 7, 17, 22, tzinfo=timezone.utc))
    assert not pillar.market_open(datetime(2026, 7, 19, 21, tzinfo=timezone.utc))
    assert pillar.market_open(datetime(2026, 7, 19, 23, tzinfo=timezone.utc))
