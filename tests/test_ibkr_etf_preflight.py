"""IBKR ETF operator preflight tests."""

import time

import pytest

from auramaur.db.database import Database
from auramaur.exchange.ibkr_equity import EquityQuote
from auramaur.monitoring.ibkr_etf_preflight import preflight
from config.settings import Settings


class ReadyClient:
    _force_paper_readonly = True

    async def get_quote(self, symbol):
        return EquityQuote(99.9, 100.0, time.time())

    async def get_adjusted_daily_closes(self, symbol):
        return [("2026-07-16", 99.0), ("2026-07-17", 100.0)]


async def models_available(api_key, models, timeout):
    return {model: None for model in models}


@pytest.mark.asyncio
async def test_ready_preflight_checks_all_dependencies():
    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    settings.ibkr.enabled = True
    settings.ibkr.etf_paper_enabled = True
    settings.openai_api_key = "test-key"
    report = await preflight(
        settings, db, client=ReadyClient(), model_checker=models_available)
    assert report.ready
    assert {result.name for result in report.results} == {
        "feature gates", "paper isolation", "market quote", "adjusted bars",
        "database schema", "OpenAI models", "experiment cells"}
    await db.close()


@pytest.mark.asyncio
async def test_preflight_fails_closed_on_routing_data_and_models():
    class UnsafeClient(ReadyClient):
        _force_paper_readonly = False

        async def get_quote(self, symbol):
            return EquityQuote(99.9, 100.0, time.time() - 10_000)

        async def get_adjusted_daily_closes(self, symbol):
            return []

    async def missing_models(api_key, models, timeout):
        return {model: "not found" for model in models}

    db = Database(":memory:")
    await db.connect()
    settings = Settings()
    report = await preflight(
        settings, db, client=UnsafeClient(), model_checker=missing_models)
    assert not report.ready
    blocked = {result.name for result in report.results if result.severity == "BLOCK"}
    assert {"feature gates", "paper isolation", "market quote", "adjusted bars",
            "OpenAI models"}.issubset(blocked)
    await db.close()
