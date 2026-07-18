import time
import asyncio

import pytest

from auramaur.db.database import Database
from auramaur.exchange.ibkr_market_data import MarketDataQuote
from auramaur.monitoring.ibkr_multiasset_preflight import preflight
from config.settings import Settings


class ReadyClient:
    readonly = True

    async def get_quote(self, spec):
        return MarketDataQuote(spec.key, 99.0, 100.0, time.time(), 42,
                               spec.currency, spec.multiplier)

    async def get_daily_bars(self, spec):
        return [(f"2026-06-{day:02d}", float(day)) for day in range(1, 22)]


@pytest.mark.asyncio
async def test_preflight_proves_all_six_books_and_isolation():
    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.enabled = True
    report = await preflight(settings, db, client=ReadyClient())
    assert report.ready
    assert {result.book for result in report.results} >= {
        "isolation", "database", "feature gate", "global_etf", "fx",
        "futures", "international_equity", "options", "bonds"}
    await db.close()


@pytest.mark.asyncio
async def test_preflight_blocks_client_with_order_surface_and_missing_data():
    class UnsafeClient(ReadyClient):
        readonly = False

        async def place_order(self):
            pass

        async def get_quote(self, spec):
            return None

    db = Database(":memory:")
    await db.connect()
    report = await preflight(Settings(), db, client=UnsafeClient())
    assert not report.ready
    blocked = {result.book for result in report.results
               if result.severity == "BLOCK"}
    assert "isolation" in blocked
    assert set(("global_etf", "fx", "futures", "international_equity",
                "options", "bonds")) <= blocked
    await db.close()


@pytest.mark.asyncio
async def test_preflight_blocks_stale_quotes():
    class StaleClient(ReadyClient):
        async def get_quote(self, spec):
            return MarketDataQuote(spec.key, 99.0, 100.0, time.time() - 3600, 42,
                                   spec.currency, spec.multiplier)

    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.enabled = True
    report = await preflight(settings, db, client=StaleClient())
    blocked = {result.book for result in report.results if result.severity == "BLOCK"}
    assert {"global_etf", "fx", "futures", "international_equity",
            "options", "bonds"} <= blocked
    await db.close()


@pytest.mark.asyncio
async def test_preflight_blocks_synthetic_quotes_as_non_executable():
    class SyntheticClient(ReadyClient):
        async def get_quote(self, spec):
            return MarketDataQuote(spec.key, 99.0, 100.0, time.time(), 42,
                                   spec.currency, spec.multiplier, "synthetic_option")

    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.enabled = True
    report = await preflight(settings, db, client=SyntheticClient())
    assert not report.ready
    await db.close()


@pytest.mark.asyncio
async def test_preflight_accepts_closed_venue_with_contract_history():
    class ClosedClient(ReadyClient):
        async def is_market_open(self, spec):
            return False

        async def get_quote(self, spec):
            return None

    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.enabled = True
    report = await preflight(settings, db, client=ClosedClient())
    assert report.ready
    book_results = [r for r in report.results if r.book in {
        "global_etf", "fx", "futures", "international_equity", "options", "bonds"}]
    assert all(result.severity == "WARN" for result in book_results)
    await db.close()


@pytest.mark.asyncio
async def test_preflight_retries_transient_pacing_errors():
    class PacingClient(ReadyClient):
        def __init__(self):
            self.attempts = set()

        async def get_quote(self, spec):
            if spec.key not in self.attempts:
                self.attempts.add(spec.key)
                raise RuntimeError("Error 162: pacing violation")
            return await super().get_quote(spec)

    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.enabled = True
    settings.ibkr.multiasset_preflight_retry_seconds = 0
    report = await preflight(settings, db, client=PacingClient())
    assert report.ready
    await db.close()


@pytest.mark.asyncio
async def test_preflight_bounds_concurrency_and_retries_pacing_errors():
    class PacingClient(ReadyClient):
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.attempts = {}

        async def get_quote(self, spec):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0)
            self.active -= 1
            count = self.attempts.get(spec.key, 0)
            self.attempts[spec.key] = count + 1
            if count == 0:
                raise RuntimeError("IBKR error 162: historical data pacing violation")
            return await super().get_quote(spec)

    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.enabled = True
    settings.ibkr.multiasset_preflight_concurrency = 2
    settings.ibkr.multiasset_preflight_retry_seconds = 0
    client = PacingClient()
    report = await preflight(settings, db, client=client)
    assert report.ready
    assert client.max_active <= 2
    assert all(attempts == 2 for attempts in client.attempts.values())
    await db.close()


@pytest.mark.asyncio
async def test_edge_evidence_is_net_of_commissions_and_financing():
    db = Database(":memory:")
    await db.connect()
    await db.execute(
        "INSERT INTO ibkr_paper_ledger (book, kind, pnl_usd, source_ref) VALUES "
        "('global_etf', 'trade', 10, 'trade-1'), "
        "('global_etf', 'commission', -1.5, 'fee-1'), "
        "('global_etf', 'financing', -0.5, 'finance-1')")
    await db.commit()
    settings = Settings()
    settings.ibkr.enabled = True
    report = await preflight(settings, db, client=ReadyClient())
    edge = next(result for result in report.results if result.book == "global_etf:edge")
    assert "1 exits" in edge.detail
    assert "$8.00 net P&L" in edge.detail
    await db.close()
