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
    # The shared prediction-market wallet is not touched. The books now DO
    # write cost_basis/pnl_ledger — that is how they reach the graduation
    # ladder — so the guard is no longer "these tables are empty" but "every
    # row is inside the ibkr: namespace, which PaperTrader excludes". The
    # balance assertion below is the one that actually matters.
    from auramaur.exchange.paper import PaperTrader

    wallet = PaperTrader(db, initial_balance=1_000.0)
    assert await wallet._compute_balance() == 1_000.0
    for table in ("cost_basis", "pnl_ledger"):
        rows = await db.fetchall(f"SELECT market_id FROM {table}")
        assert rows, f"{table} should now carry ladder evidence"
        assert all(r["market_id"].startswith("ibkr:") for r in rows)
    # Nothing may reach the account-wide LIVE daily-loss gate.
    assert (await db.fetchone(
        "SELECT COUNT(*) AS n FROM pnl_ledger WHERE is_paper = 0"))["n"] == 0
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


def test_books_declare_gated_multiasset_mode_and_asset_calendars():
    """These books are paper by DEFAULT, not paper-only.

    `_fill` places through IBKRMultiAssetExecution once a book graduates,
    behind its own confirm-live + is_live + environment + kill-switch chain.
    They were declared PAPER_SIMULATED, which the enum defines as "no order
    path exists at all" — a declaration that told a reviewer real money was
    impossible here.
    """
    settings = Settings()
    assert not settings.ibkr.multiasset_books["options"].enabled
    assert not settings.ibkr.multiasset_books["bonds"].enabled
    for book in IBKRBook:
        pillar = IBKRMultiAssetPaperBook(settings, None, None, book)
        assert pillar.execution_mode.value == "direct_multiasset"


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


def test_stranded_positions_in_disabled_book_warn():
    """Positions in a book that is not running must be surfaced — a disabled
    book never cycles, so its positions are never re-marked or exit-managed."""
    import asyncio
    from unittest.mock import patch
    from auramaur.db.database import Database
    from auramaur.strategy.ibkr_multiasset_paper import warn_stranded_positions

    async def run():
        db = Database(":memory:")
        await db.connect()
        try:
            await db.execute(
                """INSERT INTO ibkr_paper_positions
                   (book, instrument_key, con_id, currency, quantity, multiplier,
                    fx_to_usd, avg_cost, current_price, price_source,
                    instrument_spec_json, stop_price, initial_risk_usd,
                    entry_commission_usd, entry_fill_ref)
                   VALUES ('international_equity', 'SHEL.L', 1, 'GBP', 0.1, 1.0,
                           1.35, 3200.0, 3200.0, 'ibkr_live', '{}', 3100.0,
                           12.0, 1.0, 'ref-1')""")
            await db.execute(
                """INSERT INTO ibkr_paper_positions
                   (book, instrument_key, con_id, currency, quantity, multiplier,
                    fx_to_usd, avg_cost, current_price, price_source,
                    instrument_spec_json, stop_price, initial_risk_usd,
                    entry_commission_usd, entry_fill_ref)
                   VALUES ('fx', 'USDCAD', 2, 'CAD', 1.0, 1000.0, 0.71,
                           1.40, 1.40, 'ibkr_live', '{}', 1.39, 5.0, 0.2,
                           'ref-2')""")
            await db.commit()
            with patch("auramaur.strategy.ibkr_multiasset_paper.log") as mock_log:
                stranded = await warn_stranded_positions(db, {"fx"})
            assert stranded == ["international_equity"]
            warns = [c for c in mock_log.warning.call_args_list
                     if c.args and c.args[0] == "ibkr_multiasset.stranded_positions"]
            assert len(warns) == 1
            assert warns[0].kwargs["book"] == "international_equity"
            assert warns[0].kwargs["count"] == 1

            # All books enabled -> nothing stranded, no warning.
            with patch("auramaur.strategy.ibkr_multiasset_paper.log") as mock_log:
                stranded = await warn_stranded_positions(
                    db, {"fx", "international_equity"})
            assert stranded == []
            assert not mock_log.warning.called
        finally:
            await db.close()
    asyncio.run(run())


@pytest.mark.asyncio
async def test_leveraged_instrument_is_capped_on_capital_not_notional():
    """The cap is a capital budget; cost_basis stores notional.

    _exceeds_market_cap compared `size * price` against ibkr.paper_budget_usd.
    For a local-currency or leveraged instrument those are different units:
    35.8 shares of 7203.T at Y2,501 reads as Y89,615 against a $5,000 cap and
    would have blocked an entry the book sizes at ~$578. Order.capital_ratio
    converts, and is 1.0 for everything priced in USD at multiplier 1 — so no
    prediction-market order changes.
    """
    from auramaur.exchange.ibkr_intent import prepare_instrument_order
    from auramaur.exchange.ibkr_instruments import BY_KEY
    from auramaur.exchange.models import OrderSide

    spec = next(iter(BY_KEY.values()))

    # Yen-priced equity: 35.8 shares at Y2,501, FX 1/155, no multiplier.
    jpy = prepare_instrument_order(
        spec, side=OrderSide.BUY, quantity=35.82, price=2501.0, is_live=False,
        strategy_source="ibkr_international_equity_paper",
        usd_per_point=1 / 155.0, usd_capital_per_unit=2501.0 / 155.0)
    assert jpy.notional == pytest.approx(578.0, abs=2.0)      # USD, not Y89,615
    assert jpy.capital_ratio == pytest.approx(1.0, abs=1e-6)  # unleveraged

    # FX: notional per unit ~$1,081, committed capital ~$100.
    fx = prepare_instrument_order(
        spec, side=OrderSide.BUY, quantity=7.0, price=1.081, is_live=False,
        strategy_source="ibkr_fx_paper",
        usd_per_point=1000.0, usd_capital_per_unit=100.0)
    assert fx.notional == pytest.approx(7567.0, abs=2.0)
    # Capital-basis exposure is what the budget bounds: 7 x $100.
    assert fx.notional * fx.capital_ratio == pytest.approx(700.0, abs=1.0)

    # A USD prediction-style order is untouched: ratio exactly 1.0.
    plain = prepare_instrument_order(
        spec, side=OrderSide.BUY, quantity=10.0, price=0.42, is_live=False,
        strategy_source="llm")
    assert plain.capital_ratio == 1.0
    assert plain.notional == pytest.approx(4.2)


@pytest.mark.asyncio
async def test_usd_pnl_is_booked_not_local_currency_pnl():
    """record_fill realizes (price - avg) * size with no multiplier or FX.

    Booking a raw local price recorded $0.02 where the FX book records $21.61 —
    a 1000x understatement fed straight to the ladder's net-P&L bar.
    """
    from auramaur.exchange.ibkr_intent import prepare_instrument_order
    from auramaur.exchange.ibkr_instruments import BY_KEY
    from auramaur.exchange.models import OrderSide

    spec = next(iter(BY_KEY.values()))
    entry = prepare_instrument_order(
        spec, side=OrderSide.BUY, quantity=2.0, price=1.0810, is_live=False,
        strategy_source="ibkr_fx_paper", usd_per_point=1000.0)
    exit_ = prepare_instrument_order(
        spec, side=OrderSide.SELL, quantity=2.0, price=1.0918, is_live=False,
        strategy_source="ibkr_fx_paper", usd_per_point=1000.0)
    # What PnLTracker will realize: (exit - entry) * size, on USD prices.
    realized = (exit_.price - entry.price) * 2.0
    assert realized == pytest.approx(21.6, abs=0.2)
    # The local-price arithmetic the old path would have booked.
    assert (1.0918 - 1.0810) * 2.0 == pytest.approx(0.0216, abs=0.001)


@pytest.mark.asyncio
async def test_a_live_multiasset_fill_is_not_booked_to_the_ledger():
    """is_paper=0 rows feed the account-wide live daily-loss gate.

    Booking a live IBKR fill would add its P&L to a LIVE RISK GATE's inputs;
    booking it as paper would be a lie in the ladder's evidence. Neither is a
    decision this increment makes, so a live fill is skipped and logged.
    """
    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    pillar = IBKRMultiAssetPaperBook(settings, FakeMarketData(), db, IBKRBook.FX)
    spec = BY_BOOK[IBKRBook.FX][0]

    await pillar._book_fill(spec, side="BUY", quantity=2.0, price=1.081,
                            fx=1.0, multiplier=1000.0, fee=0.2,
                            fill_ref="live-ref", was_live=True)
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM pnl_ledger"))["n"] == 0
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM cost_basis"))["n"] == 0

    await pillar._book_fill(spec, side="BUY", quantity=2.0, price=1.081,
                            fx=1.0, multiplier=1000.0, fee=0.2,
                            fill_ref="paper-ref", was_live=False)
    assert (await db.fetchone("SELECT COUNT(*) AS n FROM cost_basis"))["n"] == 1
    await db.close()


@pytest.mark.asyncio
async def test_entries_off_blocks_new_risk_but_keeps_managing_held():
    """The kill lever (2026-08-03): entries_enabled=False must stop NEW
    positions while held ones keep being marked and exit-managed, so a
    killed book DRAINS instead of stranding — the failure mode
    enabled:false has (warn_stranded_positions) and the lever the
    market_maker demotion lacked."""
    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.multiasset_paper_enabled = True
    settings.ibkr.multiasset_registry_required = False
    cfg = settings.ibkr.multiasset_books["global_etf"]
    cfg.max_positions = 1
    seed = IBKRMultiAssetPaperBook(
        settings, FakeMarketData(), db, IBKRBook.GLOBAL_ETF)
    seed.market_open = lambda now=None: True
    assert await seed.run_once() == 1

    cfg.entries_enabled = False
    cfg.max_positions = 8  # room exists — the lever alone must block
    pillar = IBKRMultiAssetPaperBook(
        settings, FakeMarketData(), db, IBKRBook.GLOBAL_ETF)
    pillar.market_open = lambda now=None: True
    assert await pillar.run_once() == 0
    rows = await db.fetchall(
        "SELECT current_price FROM ibkr_paper_positions")
    assert len(rows) == 1  # no new risk, held position not stranded
    assert rows[0]["current_price"] is not None  # still being marked
    await db.close()


@pytest.mark.parametrize(
    "bid, expect_sell",
    [
        # Round-trip commissions straddle take_profit_pct (10%). Entry 100.0,
        # 1 unit: $1.00 charged on the way in and $1.00 on the way out.
        (111.0, False),  # gross +11.0%, net +9.0% — under the threshold
        (113.0, True),   # gross +13.0%, net +11.0% — clears it
    ],
)
@pytest.mark.asyncio
async def test_take_profit_waits_for_commissions_to_clear_the_threshold(
        bid, expect_sell):
    """IBKR take-profit measures the NET gain, not the quoted one.

    Both commissions are known exactly here — the entry's is stored on the
    position and the exit's is priced off the executable bid — so a gross
    comparison sells a winner the round trip has not actually earned. The two
    bids bracket the threshold, which a single-sided case cannot.
    """
    class _BidQuotes(FakeMarketData):
        async def get_quote_by_con_id(self, spec, con_id):
            return MarketDataQuote(spec.key, bid, bid * 1.0001, time.time(),
                                   con_id, spec.currency, spec.multiplier)

    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.multiasset_paper_enabled = True
    settings.ibkr.multiasset_registry_required = False
    cfg = settings.ibkr.multiasset_books["global_etf"]
    cfg.entries_enabled = False  # isolate exit management from new risk
    cfg.take_profit_pct = 10.0
    spec = BY_BOOK[IBKRBook.GLOBAL_ETF][0]
    await db.execute(
        """INSERT INTO ibkr_paper_positions
           (book, instrument_key, con_id, currency, quantity, multiplier,
            fx_to_usd, avg_cost, current_price, stop_price,
            entry_commission_usd, entry_fill_ref)
           VALUES ('global_etf', ?, 42, ?, 1, ?, 1, 100.0, 100.0, 0, 1.0,
                   'seed')""",
        (spec.key, spec.currency, spec.multiplier))
    await db.commit()

    pillar = IBKRMultiAssetPaperBook(settings, _BidQuotes(), db, IBKRBook.GLOBAL_ETF)
    pillar.market_open = lambda now=None: True
    await pillar.run_once()

    sells = await db.fetchone(
        "SELECT COUNT(*) AS n FROM ibkr_paper_fills WHERE side = 'SELL'")
    assert bool(sells["n"]) is expect_sell
    held = await db.fetchone(
        "SELECT COUNT(*) AS n FROM ibkr_paper_positions WHERE book='global_etf'")
    assert bool(held["n"]) is not expect_sell
    await db.close()
