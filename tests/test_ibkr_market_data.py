from datetime import datetime, timezone
from types import SimpleNamespace
import asyncio
import sys
import types

import pytest

from auramaur.exchange.ibkr_instruments import BY_BOOK, IBKRBook
from auramaur.exchange.ibkr_market_data import IBKRReadOnlyMarketData
from config.settings import Settings


@pytest.mark.asyncio
async def test_ibkr_liquid_hours_drive_holidays_and_sessions():
    client = IBKRReadOnlyMarketData(Settings())
    contract = SimpleNamespace(conId=756733)

    async def resolve(spec):
        return contract

    client.resolve = resolve
    client._ib = SimpleNamespace(reqContractDetailsAsync=lambda c: _details())
    client._connected = True

    spec = BY_BOOK[IBKRBook.GLOBAL_ETF][0]
    assert await client.is_market_open(
        spec, now=datetime(2026, 7, 17, 15, tzinfo=timezone.utc))
    assert not await client.is_market_open(
        spec, now=datetime(2026, 7, 17, 22, tzinfo=timezone.utc))
    assert not await client.is_market_open(
        spec, now=datetime(2026, 7, 18, 15, tzinfo=timezone.utc))


async def _details():
    return [SimpleNamespace(
        liquidHours="20260717:0930-20260717:1600;20260718:CLOSED",
        tradingHours="", timeZoneId="America/New_York")]


@pytest.mark.asyncio
async def test_preflight_client_id_does_not_collide_with_daemon(monkeypatch):
    captured = {}

    class FakeIB:
        async def connectAsync(self, **kwargs):
            captured.update(kwargs)

        def reqMarketDataType(self, value):
            pass

    module = types.ModuleType("ib_async")
    module.IB = FakeIB
    monkeypatch.setitem(sys.modules, "ib_async", module)
    settings = Settings()
    client = IBKRReadOnlyMarketData(
        settings, client_id=settings.ibkr.multiasset_preflight_client_id)
    await client._ensure_connected()
    assert captured["clientId"] == 97
    assert captured["clientId"] != settings.ibkr.multiasset_client_id
    assert captured["readonly"] is True


@pytest.mark.asyncio
async def test_concurrent_probes_establish_only_one_socket(monkeypatch):
    calls = 0

    class FakeIB:
        async def connectAsync(self, **kwargs):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)

        def reqMarketDataType(self, value):
            pass

    module = types.ModuleType("ib_async")
    module.IB = FakeIB
    monkeypatch.setitem(sys.modules, "ib_async", module)
    client = IBKRReadOnlyMarketData(Settings(), client_id=97)
    await asyncio.gather(*(client._ensure_connected() for _ in range(20)))
    assert calls == 1


@pytest.mark.asyncio
async def test_stale_socket_reconnects(monkeypatch):
    instances = []

    class FakeIB:
        def __init__(self):
            self.alive = False
            instances.append(self)

        async def connectAsync(self, **kwargs):
            self.alive = True

        def isConnected(self):
            return self.alive

        def reqMarketDataType(self, value):
            pass

    module = types.ModuleType("ib_async")
    module.IB = FakeIB
    monkeypatch.setitem(sys.modules, "ib_async", module)
    client = IBKRReadOnlyMarketData(Settings(), client_id=97)
    await client._ensure_connected()
    instances[0].alive = False
    await client._ensure_connected()
    assert len(instances) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("market_data_type,source", [
    (1, "ibkr_live"), (2, "ibkr_frozen"), (3, "ibkr_delayed"),
    (4, "ibkr_delayed_frozen"),
])
async def test_quote_provenance_comes_from_ibkr_market_data_type(
        market_data_type, source):
    now = datetime.now(timezone.utc)
    ticker = SimpleNamespace(bid=100, ask=101, time=now,
                             marketDataType=market_data_type)

    class FakeIB:
        async def reqTickersAsync(self, contract):
            return [ticker]

    client = IBKRReadOnlyMarketData(Settings())
    client._ib = FakeIB()
    spec = BY_BOOK[IBKRBook.GLOBAL_ETF][0]
    quote = await client._quote_contract(spec, SimpleNamespace(conId=123))
    assert quote.source == source
