import time

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
