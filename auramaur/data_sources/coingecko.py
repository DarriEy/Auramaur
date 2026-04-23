"""CoinGecko crypto price + trending source.

Free tier, no API key required (rate-limited to ~30 req/min globally).
Surfaces current price / 24h change for major coins plus a trending-coins
list so crypto markets get structured numeric context.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import aiohttp
import structlog

from auramaur.data_sources.base import NewsItem

logger = structlog.get_logger(__name__)

_BASE = "https://api.coingecko.com/api/v3"

# Map common ticker/name tokens to CoinGecko coin ids.
_COIN_IDS: dict[str, str] = {
    "bitcoin": "bitcoin", "btc": "bitcoin",
    "ethereum": "ethereum", "eth": "ethereum",
    "solana": "solana", "sol": "solana",
    "ripple": "ripple", "xrp": "ripple",
    "cardano": "cardano", "ada": "cardano",
    "dogecoin": "dogecoin", "doge": "dogecoin",
    "polygon": "polygon-ecosystem-token", "matic": "polygon-ecosystem-token",
    "chainlink": "chainlink", "link": "chainlink",
    "avalanche": "avalanche-2", "avax": "avalanche-2",
}


class CoinGeckoSource:
    """Current price, 24h change, and trending list — served as NewsItems."""

    source_name: str = "coingecko"
    categories: set[str] | None = {"crypto"}

    async def _fetch_prices(self, session: aiohttp.ClientSession, ids: list[str]) -> list[NewsItem]:
        if not ids:
            return []
        url = f"{_BASE}/simple/price"
        params = {
            "ids": ",".join(ids),
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_market_cap": "true",
        }
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
        except Exception as e:
            logger.debug("coingecko.prices_error", error=str(e)[:120])
            return []

        now = datetime.now(timezone.utc)
        items: list[NewsItem] = []
        for coin_id, payload in data.items():
            price = payload.get("usd")
            change = payload.get("usd_24h_change")
            mcap = payload.get("usd_market_cap")
            if price is None:
                continue
            title = f"{coin_id} price ${price:,.2f} ({change:+.2f}% 24h)" if change is not None else f"{coin_id} price ${price:,.2f}"
            content = f"Price {price}. 24h change {change}%. Market cap ${mcap:,.0f}." if mcap else title
            items.append(NewsItem(
                id=hashlib.md5(f"coingecko:price:{coin_id}:{now.isoformat()[:13]}".encode()).hexdigest(),
                source="coingecko",
                title=title,
                content=content,
                url=f"https://www.coingecko.com/en/coins/{coin_id}",
                published_at=now,
                relevance_score=2.0,
            ))
        return items

    async def _fetch_trending(self, session: aiohttp.ClientSession) -> list[NewsItem]:
        url = f"{_BASE}/search/trending"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
        except Exception as e:
            logger.debug("coingecko.trending_error", error=str(e)[:120])
            return []

        now = datetime.now(timezone.utc)
        items: list[NewsItem] = []
        for entry in (data.get("coins") or [])[:7]:
            coin = entry.get("item") or {}
            name = coin.get("name") or coin.get("symbol") or "unknown"
            items.append(NewsItem(
                id=hashlib.md5(f"coingecko:trending:{coin.get('id', name)}:{now.isoformat()[:13]}".encode()).hexdigest(),
                source="coingecko",
                title=f"Trending: {name} ({coin.get('symbol', '').upper()})",
                content=f"Trending on CoinGecko. Market cap rank: {coin.get('market_cap_rank', 'n/a')}.",
                url=f"https://www.coingecko.com/en/coins/{coin.get('id', '')}",
                published_at=now,
                relevance_score=1.5,
            ))
        return items

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        # Pick coin ids referenced by the query, fall back to a default basket.
        q_lower = (query or "").lower()
        matched_ids: list[str] = []
        for token, coin_id in _COIN_IDS.items():
            if token in q_lower and coin_id not in matched_ids:
                matched_ids.append(coin_id)
        if not matched_ids:
            # Default to BTC + ETH so crypto-category queries always have
            # macro price context even when the question is about a
            # smaller coin we can't resolve.
            matched_ids = ["bitcoin", "ethereum"]

        async with aiohttp.ClientSession() as session:
            prices = await self._fetch_prices(session, matched_ids[:5])
            trending = await self._fetch_trending(session)

        combined = prices + trending
        return combined[:limit]

    async def close(self) -> None:
        return None
