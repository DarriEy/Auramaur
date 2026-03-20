"""FRED economic data source (sync API wrapped in executor)."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone

import structlog
from fredapi import Fred  # type: ignore[import-untyped]

from auramaur.data_sources.base import DataSource, NewsItem

logger = structlog.get_logger(__name__)

_DEFAULT_SERIES: dict[str, str] = {
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "Consumer Price Index (CPI)",
    "GDP": "Gross Domestic Product (GDP)",
    "FEDFUNDS": "Federal Funds Rate",
}


class FREDSource:
    """Treat FRED economic data releases as news items."""

    source_name: str = "fred"

    def __init__(self, api_key: str) -> None:
        self._fred = Fred(api_key=api_key)

    def _fetch_sync(self, query: str, limit: int) -> list[NewsItem]:
        items: list[NewsItem] = []
        query_lower = query.lower()

        for series_id, description in _DEFAULT_SERIES.items():
            # Optionally filter by query relevance
            if query_lower and not any(
                kw in description.lower() or kw in series_id.lower()
                for kw in query_lower.split()
            ):
                # Still include if query is generic / short
                if len(query_lower) > 3:
                    continue

            try:
                observations = self._fred.get_series(series_id).dropna().tail(limit)
            except Exception:
                logger.exception("fred_series_failed", series_id=series_id)
                continue

            for date, value in observations.items():
                ts = date.to_pydatetime() if hasattr(date, "to_pydatetime") else datetime.now(timezone.utc)
                item_id = hashlib.sha256(
                    f"fred:{series_id}:{date}".encode()
                ).hexdigest()[:16]
                items.append(
                    NewsItem(
                        id=item_id,
                        source=self.source_name,
                        title=f"{description}: {value}",
                        content=f"FRED series {series_id} recorded {value} on {date}.",
                        url=f"https://fred.stlouisfed.org/series/{series_id}",
                        published_at=ts,
                        keywords=[series_id, description.lower()],
                    )
                )

        items.sort(key=lambda n: n.published_at, reverse=True)
        logger.info("fred_fetched", count=len(items[:limit]))
        return items[:limit]

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        """Run the blocking FRED API call in a thread-pool executor."""
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, self._fetch_sync, query, limit)
        except Exception:
            logger.exception("fred_fetch_failed", query=query)
            return []

    async def close(self) -> None:
        # fredapi has no persistent connection to close
        pass
