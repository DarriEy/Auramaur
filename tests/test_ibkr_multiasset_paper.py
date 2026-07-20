import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from auramaur.db.database import Database
from auramaur.exchange.ibkr_instruments import BY_BOOK, ContractKind, IBKRBook
from auramaur.exchange.ibkr_market_data import MarketDataQuote
from auramaur.strategy.ibkr_multiasset_paper import IBKRMultiAssetPaperBook
from config.settings import Settings


class FakeMarketData:
    con_ids_requested = []

    async def get_quote(self, spec):
        price = (2.0 if spec.kind is ContractKind.OPTION else
                 1.0 if spec.kind is ContractKind.FOREX else 100.0)
        return MarketDataQuote(spec.key, price, price * 1.0001, time.time(),
                               abs(hash(spec.key)) % 100000 + 1,
                               spec.currency, spec.multiplier)

    async def get_daily_bars(self, spec, duration="3 M"):
        return [(f"session-{day:03d}", 80 + day * 0.2) for day in range(121)]

    async def get_fx_to_usd(self, currency):
        return 1.0

    async def is_market_open(self, spec, *, con_id=0, now=None):
        return True

    async def get_daily_bars_by_con_id(self, spec, con_id, duration="3 M"):
        return await self.get_daily_bars(spec, duration)

    async def get_quote_by_con_id(self, spec, con_id):
        self.con_ids_requested.append(con_id)
        quote = await self.get_quote(spec)
        return MarketDataQuote(quote.key, quote.bid, quote.ask, quote.timestamp,
                               con_id, quote.currency, quote.multiplier)


@pytest.mark.asyncio
async def test_all_six_books_write_only_isolated_paper_tables():
    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.multiasset_paper_enabled = True
    settings.ibkr.multiasset_registry_required = False
    settings.ibkr.multiasset_refreshes_per_cycle = 1
    for cfg in settings.ibkr.multiasset_books.values():
        cfg.enabled = True  # exercise isolation plumbing, including gated books
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
        "SELECT COUNT(*) AS n FROM ibkr_paper_fills WHERE price_source='ibkr_live'"))["n"] == 6
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
    settings.ibkr.multiasset_registry_required = False
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
    settings.ibkr.multiasset_registry_required = False
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
async def test_completed_position_records_one_net_round_trip():
    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    book = IBKRMultiAssetPaperBook(
        settings, FakeMarketData(), db, IBKRBook.GLOBAL_ETF)
    spec = BY_BOOK[IBKRBook.GLOBAL_ETF][0]
    buy = MarketDataQuote(spec.key, 99.0, 100.0, time.time(), 42,
                          spec.currency, spec.multiplier)
    await book._fill(spec, buy, "BUY", 1, 1.0, stop_price=95,
                     initial_risk_usd=5)
    position = await db.fetchone(
        "SELECT * FROM ibkr_paper_positions WHERE book='global_etf'")
    sell = MarketDataQuote(spec.key, 110.0, 111.0, time.time(), 42,
                           spec.currency, spec.multiplier)
    await book._fill(spec, sell, "SELL", 1, 1.0,
                     entry_price=float(position["avg_cost"]))
    result = await db.fetchone("SELECT * FROM ibkr_paper_round_trips")
    assert result is not None
    assert result["entry_fill_ref"]
    assert result["exit_fill_ref"] != result["entry_fill_ref"]
    assert result["net_pnl_usd"] == pytest.approx(
        result["gross_pnl_usd"] - result["entry_commission_usd"]
        - result["exit_commission_usd"])
    assert (await db.fetchone(
        "SELECT COUNT(*) AS n FROM ibkr_paper_round_trips"))["n"] == 1
    await db.close()


@pytest.mark.asyncio
async def test_asset_class_risk_cap_bounds_correlated_entries():
    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.multiasset_paper_enabled = True
    settings.ibkr.multiasset_registry_required = False
    settings.ibkr.multiasset_refreshes_per_cycle = 2
    cfg = settings.ibkr.multiasset_books["global_etf"]
    cfg.max_positions = 2
    cfg.max_position_pct = 40
    cfg.max_deployment_pct = 80
    cfg.risk_per_position_pct = 0.25
    cfg.max_asset_class_risk_pct = 0.25
    pillar = IBKRMultiAssetPaperBook(
        settings, FakeMarketData(), db, IBKRBook.GLOBAL_ETF)
    await pillar.run_once()
    row = await db.fetchone(
        "SELECT SUM(initial_risk_usd) AS risk FROM ibkr_paper_positions "
        "WHERE book='global_etf'")
    assert float(row["risk"] or 0) <= 12.5
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
async def test_delayed_quote_cannot_create_fill():
    class DelayedMarketData(FakeMarketData):
        async def get_quote(self, spec):
            quote = await super().get_quote(spec)
            return MarketDataQuote(quote.key, quote.bid, quote.ask, quote.timestamp,
                                   quote.con_id, quote.currency, quote.multiplier,
                                   "ibkr_delayed")

    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.multiasset_paper_enabled = True
    pillar = IBKRMultiAssetPaperBook(
        settings, DelayedMarketData(), db, IBKRBook.GLOBAL_ETF)
    assert await pillar.run_once() == 0
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM ibkr_paper_fills"))["n"] == 0
    await db.close()


@pytest.mark.asyncio
async def test_stop_executes_during_spread_and_history_dislocation():
    class DislocatedMarketData(FakeMarketData):
        async def get_quote_by_con_id(self, spec, con_id):
            return MarketDataQuote(spec.key, 80, 120, time.time(), con_id,
                                   spec.currency, spec.multiplier)

        async def get_fx_to_usd(self, currency):
            return None

        async def get_daily_bars_by_con_id(self, spec, con_id, duration="3 M"):
            raise AssertionError("hard stops must not wait for history")

    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.multiasset_paper_enabled = True
    spec = BY_BOOK[IBKRBook.GLOBAL_ETF][0]
    seed = IBKRMultiAssetPaperBook(settings, FakeMarketData(), db, IBKRBook.GLOBAL_ETF)
    quote = await FakeMarketData().get_quote(spec)
    await seed._fill(spec, quote, "BUY", 1, 1.0)
    pillar = IBKRMultiAssetPaperBook(
        settings, DislocatedMarketData(), db, IBKRBook.GLOBAL_ETF)
    await pillar.run_once()
    assert await db.fetchone(
        "SELECT 1 FROM ibkr_paper_positions WHERE instrument_key = ?", (spec.key,)) is None
    await db.close()


@pytest.mark.asyncio
async def test_disabled_held_instrument_is_still_managed_and_exited():
    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.multiasset_paper_enabled = True
    spec = BY_BOOK[IBKRBook.GLOBAL_ETF][0]
    seed = IBKRMultiAssetPaperBook(settings, FakeMarketData(), db, IBKRBook.GLOBAL_ETF)
    quote = await FakeMarketData().get_quote(spec)
    await seed._fill(spec, quote, "BUY", 1, 1.0)
    settings.ibkr.multiasset_disabled_instruments = [spec.key]

    class StopMarketData(FakeMarketData):
        async def get_quote_by_con_id(self, held_spec, con_id):
            return MarketDataQuote(held_spec.key, 80, 81, time.time(), con_id,
                                   held_spec.currency, held_spec.multiplier)

    pillar = IBKRMultiAssetPaperBook(
        settings, StopMarketData(), db, IBKRBook.GLOBAL_ETF)
    await pillar.run_once()
    assert await db.fetchone(
        "SELECT 1 FROM ibkr_paper_positions WHERE instrument_key = ?", (spec.key,)) is None
    await db.close()


@pytest.mark.asyncio
async def test_open_position_is_marked_by_original_contract_id():
    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.multiasset_paper_enabled = True
    settings.ibkr.multiasset_registry_required = False
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
    marked = await db.fetchone(
        "SELECT updated_at, con_id FROM ibkr_paper_positions WHERE book='futures'")
    assert int(marked["con_id"]) == int(row["con_id"])
    await db.close()


def test_books_declare_paper_simulated_mode_and_asset_calendars():
    settings = Settings()
    assert not settings.ibkr.multiasset_books["options"].enabled
    assert not settings.ibkr.multiasset_books["bonds"].enabled
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


@pytest.mark.asyncio
async def test_broker_calendar_honours_holiday_and_split_sessions():
    from auramaur.exchange.ibkr_instruments import BY_BOOK
    from auramaur.exchange.ibkr_market_data import IBKRReadOnlyMarketData

    class FakeIB:
        async def reqContractDetailsAsync(self, contract):
            return [SimpleNamespace(
                liquidHours=("20260720:CLOSED;"
                             "20260721:0930-20260721:1130,"
                             "20260721:1230-20260721:1600"),
                tradingHours="", timeZoneId="America/New_York")]

    client = IBKRReadOnlyMarketData(Settings())
    client._ib = FakeIB()
    client._connected = True
    contract = SimpleNamespace(conId=123)

    async def resolve(spec):
        return contract

    client.resolve = resolve
    spec = BY_BOOK[IBKRBook.GLOBAL_ETF][0]
    assert not await client.is_market_open(
        spec, now=datetime(2026, 7, 20, 15, tzinfo=timezone.utc))
    assert await client.is_market_open(
        spec, now=datetime(2026, 7, 21, 14, tzinfo=timezone.utc))
    assert not await client.is_market_open(
        spec, now=datetime(2026, 7, 21, 16, tzinfo=timezone.utc))
    assert await client.is_market_open(
        spec, now=datetime(2026, 7, 21, 18, tzinfo=timezone.utc))


# ---- FX audit follow-ups (2026-07-20) --------------------------------------

class RankedFakeMarketData(FakeMarketData):
    """Distinct momentum per instrument so entry ordering is observable."""

    def __init__(self, momenta):
        self._momenta = momenta

    async def get_daily_bars(self, spec, duration="3 M"):
        # Construct a price path whose normalized momentum ordering follows
        # the configured per-key slope: later closes grow by slope per step.
        slope = self._momenta.get(spec.key, 0.0)
        return [(f"s-{d:03d}", 100 * (1 + slope) ** d) for d in range(121)]


@pytest.mark.asyncio
async def test_entries_rank_strongest_momentum_first():
    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.multiasset_paper_enabled = True
    settings.ibkr.multiasset_registry_required = False
    settings.ibkr.multiasset_refreshes_per_cycle = 12
    cfg = settings.ibkr.multiasset_books["fx"]
    cfg.enabled = True
    cfg.max_positions = 1  # one slot: the strongest signal must win it
    cfg.max_position_pct = 40
    cfg.max_deployment_pct = 60
    momenta = {spec.key: 0.001 for spec in BY_BOOK[IBKRBook.FX]}
    momenta["GBPJPY"] = 0.004  # clearly strongest trend
    pillar = IBKRMultiAssetPaperBook(
        settings, RankedFakeMarketData(momenta), db, IBKRBook.FX)
    pillar.market_open = lambda now=None: True
    assert await pillar.run_once() == 1
    row = await db.fetchone(
        "SELECT instrument_key FROM ibkr_paper_positions WHERE book='fx'")
    assert row["instrument_key"] == "GBPJPY"
    await db.close()


@pytest.mark.asyncio
async def test_daily_mark_upserts_one_row_per_day():
    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.multiasset_paper_enabled = True
    settings.ibkr.multiasset_registry_required = False
    settings.ibkr.multiasset_refreshes_per_cycle = 1
    settings.ibkr.multiasset_books["futures"].max_position_pct = 40
    pillar = IBKRMultiAssetPaperBook(
        settings, FakeMarketData(), db, IBKRBook.FUTURES)
    pillar.market_open = lambda now=None: True
    await pillar.run_once()
    await pillar.run_once()  # same day: must update, not duplicate
    rows = await db.fetchall(
        "SELECT * FROM ibkr_paper_daily_marks WHERE book='futures'")
    assert len(rows) == 1
    assert rows[0]["equity_usd"] == pytest.approx(
        rows[0]["realized_cum_usd"] + rows[0]["unrealized_usd"])
    await db.close()


@pytest.mark.asyncio
async def test_fx_research_recorder_records_trend_and_carry_once_daily():
    class Rates:
        async def rate(self, currency):
            return {"GBP": 0.05, "JPY": 0.001}.get(currency, 0.03)

    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.multiasset_paper_enabled = True
    settings.ibkr.multiasset_registry_required = False
    pillar = IBKRMultiAssetPaperBook(
        settings, RankedFakeMarketData({s.key: 0.002 for s in BY_BOOK[IBKRBook.FX]}),
        db, IBKRBook.FX, rates_provider=Rates())
    await pillar._record_research_signals()
    await pillar._record_research_signals()  # second call same day: no-op
    trend = await db.fetchall(
        "SELECT * FROM ibkr_research_signals WHERE signal_name='trend_normalized'")
    carry = await db.fetchall(
        "SELECT * FROM ibkr_research_signals WHERE signal_name='fx_carry_trend'")
    assert len(trend) == len(BY_BOOK[IBKRBook.FX])
    assert len(carry) == len(BY_BOOK[IBKRBook.FX])
    gbpjpy = next(r for r in carry if r["instrument_key"] == "GBPJPY")
    assert gbpjpy["direction"] == 1  # positive carry + uptrend agree
    await db.close()
