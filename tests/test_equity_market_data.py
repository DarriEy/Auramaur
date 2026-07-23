from types import SimpleNamespace
from unittest.mock import AsyncMock
import httpx
from auramaur.exchange.equity_market_data import AlpacaIEXMarketData, ResearchEquityMarketData
from auramaur.exchange.ibkr_equity import EquityQuote

async def test_router_prefers_execution_ready_ibkr():
    ibkr = SimpleNamespace(get_quote=AsyncMock(return_value=EquityQuote(100, 101, 1, "ibkr_live")))
    alpaca = SimpleNamespace(get_quote=AsyncMock())
    router = ResearchEquityMarketData(ibkr, alpaca)
    assert (await router.get_quote("SPY")).source == "ibkr_live"
    alpaca.get_quote.assert_not_awaited()

async def test_router_replaces_delayed_ibkr_with_iex():
    ibkr = SimpleNamespace(get_quote=AsyncMock(return_value=EquityQuote(100, 101, 1, "ibkr_delayed")))
    alpaca = SimpleNamespace(get_quote=AsyncMock(return_value=EquityQuote(100, 101, 2, "alpaca_iex")))
    assert (await ResearchEquityMarketData(ibkr, alpaca).get_quote("SPY")).source == "alpaca_iex"

async def test_alpaca_quote_is_research_not_execution_ready():
    def handler(request):
        assert request.url.params["feed"] == "iex"
        return httpx.Response(200, json={"quote": {"bp": 100, "ap": 100.05, "t": "2026-07-22T15:30:00Z"}})
    source = AlpacaIEXMarketData("key", "secret", base_url="https://example.test")
    await source._client.aclose()
    source._client = httpx.AsyncClient(base_url="https://example.test", transport=httpx.MockTransport(handler))
    quote = await source.get_quote("SPY")
    assert quote.research_ready and not quote.execution_ready
    await source.close()
