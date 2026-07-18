"""Read-only Coinbase public market data for the spot paper comparator."""

from __future__ import annotations

from dataclasses import dataclass

import aiohttp


@dataclass(frozen=True)
class CoinbaseQuote:
    bid: float
    ask: float


class CoinbasePublicClient:
    """Tiny unauthenticated adapter; deliberately has no order-placement API."""

    _API = "https://api.exchange.coinbase.com"

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"User-Agent": "auramaur-coinbase-paper/1.0"},
            )
        return self._session

    async def get_quote(self, product_id: str) -> CoinbaseQuote | None:
        session = await self._get_session()
        async with session.get(f"{self._API}/products/{product_id}/ticker") as response:
            if response.status != 200:
                return None
            data = await response.json()
        try:
            bid, ask = float(data["bid"]), float(data["ask"])
        except (KeyError, TypeError, ValueError):
            return None
        if bid <= 0 or ask <= 0 or bid > ask:
            return None
        return CoinbaseQuote(bid=bid, ask=ask)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
