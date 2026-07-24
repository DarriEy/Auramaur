"""Tests for the Alpaca IEX multiasset quote bridge.

Without IBKR market-data subscriptions, USD stock instruments sat
`qualified_no_live_data` and the global_etf book accrued no evidence. The
bridge substitutes free real-time IEX bid/ask for PAPER fills (provenance
`alpaca_iex`), scoped to USD STOCK instruments only."""

import asyncio
import time

from auramaur.exchange.alpaca_multiasset import AlpacaMultiAssetQuotes
from auramaur.exchange.ibkr_instruments import BY_KEY
from auramaur.exchange.ibkr_market_data import MarketDataQuote
from auramaur.exchange.ibkr_equity import EquityQuote


SPY = BY_KEY["SPY"]
GBPJPY = BY_KEY["GBPJPY"]


class FakeIBKR:
    readonly = True

    def __init__(self, quote):
        self._quote = quote
        self.closed = False

    async def get_quote(self, spec):
        if isinstance(self._quote, Exception):
            raise self._quote
        return self._quote

    async def get_daily_bars(self, spec, duration="3 M"):
        return ["bar"] * 30

    async def close(self):
        self.closed = True


class FakeAlpaca:
    def __init__(self, quote):
        self._quote = quote
        self.requested = []
        self.closed = False

    async def get_quote(self, symbol):
        self.requested.append(symbol)
        return self._quote

    async def close(self):
        self.closed = True


def _ibkr_quote(source, key="SPY"):
    return MarketDataQuote(key, 100.0, 100.1, time.time(), 756733, "USD", 1.0,
                           source=source)


def test_delayed_stock_quote_falls_back_to_alpaca():
    async def run():
        alpaca = FakeAlpaca(EquityQuote(628.10, 628.15, time.time(), "alpaca_iex"))
        bridge = AlpacaMultiAssetQuotes(FakeIBKR(_ibkr_quote("ibkr_delayed")), alpaca)
        quote = await bridge.get_quote(SPY)
        assert quote.source == "alpaca_iex"
        assert quote.key == "SPY" and quote.currency == "USD"
        assert quote.bid == 628.10 and quote.ask == 628.15
        assert alpaca.requested == ["SPY"]
    asyncio.run(run())


def test_live_ibkr_quote_never_touches_alpaca():
    async def run():
        alpaca = FakeAlpaca(EquityQuote(1, 2, time.time(), "alpaca_iex"))
        bridge = AlpacaMultiAssetQuotes(FakeIBKR(_ibkr_quote("ibkr_live")), alpaca)
        quote = await bridge.get_quote(SPY)
        assert quote.source == "ibkr_live"
        assert alpaca.requested == []
    asyncio.run(run())


def test_non_stock_instruments_never_fall_back():
    async def run():
        alpaca = FakeAlpaca(EquityQuote(1, 2, time.time(), "alpaca_iex"))
        bridge = AlpacaMultiAssetQuotes(
            FakeIBKR(_ibkr_quote("ibkr_delayed", key="GBPJPY")), alpaca)
        quote = await bridge.get_quote(GBPJPY)
        assert quote.source == "ibkr_delayed"  # passed through untouched
        assert alpaca.requested == []
    asyncio.run(run())


def test_alpaca_miss_returns_original_ibkr_quote_and_delegation_works():
    async def run():
        bridge = AlpacaMultiAssetQuotes(
            FakeIBKR(_ibkr_quote("ibkr_delayed")), FakeAlpaca(None))
        quote = await bridge.get_quote(SPY)
        assert quote.source == "ibkr_delayed"
        # __getattr__ delegation: bars + isolation attributes come from IBKR.
        assert len(await bridge.get_daily_bars(SPY)) == 30
        assert bridge.readonly is True
        assert not hasattr(bridge, "place_order")
    asyncio.run(run())


def test_quote_fresh_accepts_alpaca_but_not_delayed():
    from auramaur.strategy.ibkr_multiasset_paper import IBKRMultiAssetPaperBook
    from config.settings import Settings

    book = IBKRMultiAssetPaperBook.__new__(IBKRMultiAssetPaperBook)
    book._settings = Settings()
    now = time.time()
    assert book._quote_fresh(_ibkr_quote("alpaca_iex")) is True
    assert book._quote_fresh(_ibkr_quote("ibkr_live")) is True
    assert book._quote_fresh(_ibkr_quote("ibkr_delayed")) is False
    stale = MarketDataQuote("SPY", 1, 2, now - 10_000, 0, "USD", 1.0,
                            source="alpaca_iex")
    assert book._quote_fresh(stale) is False


def test_registry_marks_alpaca_sourced_stock_eligible(tmp_path):
    async def run():
        from auramaur.db.database import Database
        from auramaur.exchange.ibkr_registry import record_validation
        from types import SimpleNamespace

        db = Database(str(tmp_path / "reg.db"))
        await db.connect()
        try:
            contract = SimpleNamespace(conId=756733, exchange="SMART",
                                       currency="USD", multiplier=1.0)
            await record_validation(db, SPY, contract,
                                    quote_source="alpaca_iex", has_history=True)
            row = await db.fetchone(
                "SELECT status, quote_source FROM ibkr_contract_registry "
                "WHERE instrument_key='SPY'")
            assert row["status"] == "eligible"
            assert row["quote_source"] == "alpaca_iex"
        finally:
            await db.close()
    asyncio.run(run())


def test_quote_fresh_tolerates_small_forward_clock_skew():
    from auramaur.strategy.ibkr_multiasset_paper import IBKRMultiAssetPaperBook
    from config.settings import Settings

    book = IBKRMultiAssetPaperBook.__new__(IBKRMultiAssetPaperBook)
    book._settings = Settings()
    now = time.time()
    ahead = MarketDataQuote("SPY", 1, 2, now + 1.0, 0, "USD", 1.0,
                            source="alpaca_iex")
    assert book._quote_fresh(ahead) is True  # ~1s server-ahead stamp is normal
    absurd = MarketDataQuote("SPY", 1, 2, now + 60.0, 0, "USD", 1.0,
                             source="alpaca_iex")
    assert book._quote_fresh(absurd) is False


def test_hung_ibkr_quote_times_out_and_falls_back_to_alpaca(monkeypatch):
    """An unentitled instrument's IBKR ticker can await forever — before the
    bound, the first eligible stock quote hung the entire multiasset task
    (2026-07-24, heartbeats silent 2h+). The bridge must time out and serve
    Alpaca instead."""
    async def run():
        class HangingIBKR(FakeIBKR):
            async def get_quote(self, spec):
                await asyncio.Event().wait()  # never resolves

        monkeypatch.setattr(
            AlpacaMultiAssetQuotes, "_IBKR_QUOTE_TIMEOUT_SECONDS", 0.05)
        alpaca = FakeAlpaca(EquityQuote(628.10, 628.15, time.time(), "alpaca_iex"))
        bridge = AlpacaMultiAssetQuotes(HangingIBKR(None), alpaca)
        quote = await asyncio.wait_for(bridge.get_quote(SPY), timeout=5)
        assert quote is not None and quote.source == "alpaca_iex"
        # Non-stock instruments time out to None rather than hanging.
        quote = await asyncio.wait_for(bridge.get_quote(GBPJPY), timeout=5)
        assert quote is None
    asyncio.run(run())
