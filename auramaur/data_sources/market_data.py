"""Financial market data source using yfinance (no API key required).

Provides real-time and historical price data for stocks, indices,
commodities, crypto, and currencies.  Returns structured data as
NewsItem objects so it plugs into the existing aggregator pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone

import structlog

from auramaur.data_sources.base import NewsItem

logger = structlog.get_logger(__name__)

# Symbols to check for any query — covers most Polymarket-relevant assets
_WATCHLIST: dict[str, list[str]] = {
    # Keywords → yfinance tickers
    "sp500 s&p stock market equit": ["^GSPC", "^VIX"],
    "nasdaq tech": ["^IXIC", "QQQ"],
    "dow jones": ["^DJI"],
    "bitcoin btc crypto": ["BTC-USD"],
    "ethereum eth": ["ETH-USD"],
    "solana sol": ["SOL-USD"],
    "gold": ["GC=F"],
    "oil crude brent": ["CL=F", "BZ=F"],
    "natural gas": ["NG=F"],
    "dollar dxy usd": ["DX-Y.NYB"],
    "treasury bond yield": ["^TNX", "^FVX"],
    "tesla tsla": ["TSLA"],
    "apple aapl": ["AAPL"],
    "nvidia nvda": ["NVDA"],
    "amazon amzn": ["AMZN"],
    "meta facebook": ["META"],
    "google alphabet": ["GOOGL"],
    "microsoft msft": ["MSFT"],
}

# Friendly names for tickers
_TICKER_NAMES: dict[str, str] = {
    "^GSPC": "S&P 500",
    "^VIX": "CBOE VIX (Volatility Index)",
    "^IXIC": "NASDAQ Composite",
    "^DJI": "Dow Jones Industrial",
    "QQQ": "QQQ ETF (NASDAQ 100)",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "GC=F": "Gold Futures",
    "CL=F": "WTI Crude Oil",
    "BZ=F": "Brent Crude Oil",
    "NG=F": "Natural Gas",
    "DX-Y.NYB": "US Dollar Index (DXY)",
    "^TNX": "10-Year Treasury Yield",
    "^FVX": "5-Year Treasury Yield",
    "TSLA": "Tesla",
    "AAPL": "Apple",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "GOOGL": "Alphabet (Google)",
    "MSFT": "Microsoft",
}


def _match_tickers(query: str) -> list[str]:
    """Find relevant tickers based on query keywords."""
    query_lower = query.lower()
    tickers: list[str] = []

    for keywords, symbols in _WATCHLIST.items():
        if any(kw in query_lower for kw in keywords.split()):
            tickers.extend(symbols)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


class MarketDataSource:
    """Financial market data as structured NewsItem objects."""

    source_name: str = "market_data"

    def _fetch_sync(self, query: str, limit: int) -> list[NewsItem]:
        import yfinance as yf  # type: ignore[import-untyped]

        tickers = _match_tickers(query)
        if not tickers:
            return []

        items: list[NewsItem] = []

        for ticker_symbol in tickers[:8]:  # Cap at 8 tickers per query
            try:
                ticker = yf.Ticker(ticker_symbol)
                hist = ticker.history(period="5d")

                if hist.empty:
                    continue

                name = _TICKER_NAMES.get(ticker_symbol, ticker_symbol)
                latest = hist.iloc[-1]
                price = latest["Close"]

                # Calculate changes
                if len(hist) >= 2:
                    prev = hist.iloc[-2]["Close"]
                    day_change = ((price - prev) / prev) * 100
                    day_str = f"{day_change:+.2f}%"
                else:
                    day_str = "N/A"

                if len(hist) >= 5:
                    week_ago = hist.iloc[0]["Close"]
                    week_change = ((price - week_ago) / week_ago) * 100
                    week_str = f"{week_change:+.2f}%"
                else:
                    week_str = "N/A"

                # Build structured content
                content = (
                    f"{name} ({ticker_symbol}): ${price:,.2f}\n"
                    f"1-day change: {day_str}\n"
                    f"5-day change: {week_str}\n"
                    f"5-day high: ${hist['High'].max():,.2f}\n"
                    f"5-day low: ${hist['Low'].min():,.2f}\n"
                    f"Volume: {latest['Volume']:,.0f}"
                )

                item_id = hashlib.sha256(
                    f"mkt:{ticker_symbol}:{datetime.now().date()}".encode()
                ).hexdigest()[:16]

                items.append(
                    NewsItem(
                        id=item_id,
                        source=self.source_name,
                        title=f"{name}: ${price:,.2f} ({day_str} today)",
                        content=content,
                        url=f"https://finance.yahoo.com/quote/{ticker_symbol}",
                        published_at=datetime.now(timezone.utc),
                        relevance_score=2.0,  # High relevance — structured data
                        keywords=[ticker_symbol, name.lower()],
                    )
                )

            except Exception as e:
                logger.debug("market_data_ticker_failed", ticker=ticker_symbol, error=str(e)[:60])

        logger.info("market_data_fetched", count=len(items), tickers=tickers[:8])
        return items[:limit]

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        """Run the blocking yfinance calls in a thread-pool executor."""
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, self._fetch_sync, query, limit)
        except Exception:
            logger.exception("market_data_fetch_failed", query=query)
            return []

    async def close(self) -> None:
        pass
