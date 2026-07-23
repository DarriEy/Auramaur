"""Provenance-aware US equity data for research and local paper fills."""
from __future__ import annotations
from datetime import datetime
import math
import httpx
import structlog
from auramaur.exchange.ibkr_equity import EquityQuote

log = structlog.get_logger()


class AlpacaIEXMarketData:
    """Free-plan IEX quotes: research-ready, never execution-ready."""

    def __init__(self, key: str, secret: str, *, base_url: str, timeout: float = 10):
        self._enabled = bool(key and secret)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout,
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        )

    async def get_quote(self, symbol: str) -> EquityQuote | None:
        if not self._enabled:
            return None
        try:
            response = await self._client.get(
                f"/v2/stocks/{symbol}/quotes/latest", params={"feed": "iex"})
            response.raise_for_status()
            quote = response.json().get("quote") or {}
            bid, ask = float(quote.get("bp") or 0), float(quote.get("ap") or 0)
            timestamp = datetime.fromisoformat(
                str(quote.get("t", "")).replace("Z", "+00:00")).timestamp()
            if (not all(math.isfinite(v) for v in (bid, ask, timestamp))
                    or bid <= 0 or ask <= 0 or bid > ask):
                return None
            return EquityQuote(bid, ask, timestamp, "alpaca_iex")
        except Exception as exc:
            log.warning("alpaca_iex.quote_error", symbol=symbol, error=str(exc)[:120])
            return None

    async def close(self) -> None:
        await self._client.aclose()


class ResearchEquityMarketData:
    """Prefer consolidated IBKR; fall back to free IEX for paper research."""

    def __init__(self, ibkr, alpaca: AlpacaIEXMarketData):
        self._ibkr, self._alpaca = ibkr, alpaca

    async def get_quote(self, symbol: str) -> EquityQuote | None:
        try:
            quote = await self._ibkr.get_quote(symbol)
        except Exception as exc:
            log.warning("equity_data.ibkr_quote_error", symbol=symbol,
                        error=str(exc)[:120])
            quote = None
        if quote is not None and quote.execution_ready:
            return quote
        return await self._alpaca.get_quote(symbol)

    async def get_adjusted_daily_closes(self, symbol: str):
        return await self._ibkr.get_adjusted_daily_closes(symbol)

    async def close(self) -> None:
        await self._alpaca.close()
        await self._ibkr.close()
