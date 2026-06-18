"""Hydrology-market watcher: precise matcher + alert-once dedup."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from auramaur.db.database import Database
from auramaur.exchange.models import Market
from auramaur.monitoring.hydro_market_watch import HydroMarketWatcher, is_hydro_market
from config.settings import Settings


def test_matcher_hits_real_hydro_terms():
    for q in [
        "Will Lake Mead drop below 1000 ft by December?",
        "Will there be a drought emergency in California in 2026?",
        "Will the Colorado River reach flood stage in July?",
        "Will Sierra snowpack exceed 150% of normal?",
        "Will streamflow at the gauge exceed 5000 cfs?",
        "Will the reservoir level fall below capacity?",
    ]:
        assert is_hydro_market(q), q


def test_matcher_rejects_false_positives():
    # the exact substrings that broke the naive scan + the live false positive
    for q in [
        "Will Oscar Piastri get pole at the 2026 F1 Austrian Grand Prix?",
        "Rainbow Six Siege: 100 Thieves vs Shopify Rebellion",
        "Will it be a snowboard gold for Team USA?",   # 'snow' substring, not hydro
        "Will Nithya Raman win the LA mayoral election?",
        "Will there be a runoff election in the Georgia Senate race?",  # ELECTION runoff
    ]:
        assert not is_hydro_market(q), q


def test_matcher_only_checks_question_not_description():
    """Description boilerplate ('a runoff election will be held') must not match —
    the watcher scopes to the question; the matcher is what's exercised here."""
    assert not is_hydro_market("a majority of the vote, a runoff election will be held")


def _settings():
    s = Settings()
    s.hydro_watch.enabled = True
    s.hydro_watch.min_liquidity = 100.0
    return s


def _mkt(mid, q, liq=5000.0):
    return Market(id=mid, exchange="kalshi", question=q, liquidity=liq, volume=liq)


def test_alerts_once_per_market():
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            disc = MagicMock()
            disc.get_markets = AsyncMock(return_value=[
                _mkt("h1", "Will Lake Mead drop below 1000 ft?"),
                _mkt("x1", "Will the F1 Austrian GP be wet?"),          # not hydro
                _mkt("h2", "Drought emergency declared?", liq=10.0),    # below min_liq
            ])
            alerts = MagicMock(); alerts.send = AsyncMock()
            w = HydroMarketWatcher(db, _settings(), {"kalshi": disc}, alerts)
            assert await w.run_once() == 1            # only h1 qualifies
            alerts.send.assert_awaited_once()
            # second cycle: already seen -> no new alert
            assert await w.run_once() == 0
            assert alerts.send.await_count == 1
        finally:
            await db.close()
    asyncio.run(run())
