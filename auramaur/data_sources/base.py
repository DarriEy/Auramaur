"""Base protocol and models for data sources."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class NewsItem(BaseModel):
    id: str
    source: str
    title: str
    content: str = ""
    url: str = ""
    published_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # exact: publisher supplied; inferred/provider_seen/unknown: progressively
    # weaker time semantics. Unknown dates must not masquerade as breaking news.
    timestamp_quality: str = "exact"
    relevance_score: float = 0.0
    keywords: list[str] = Field(default_factory=list)
    ingestion_run_id: str = ""
    information_mode: str = "production"  # production | shadow | paired


@runtime_checkable
class DataSource(Protocol):
    """Protocol that all data sources must implement.

    ``categories`` gates when the source fires:
      * ``None`` (or attr missing): fires for every query — use for broad
        sources like general news, web search, or top-of-funnel RSS.
      * A ``set[str]`` of market categories: fires only when the market's
        category is in the set. Domain sources (e.g. USGS for seismic)
        declare their scope this way so they don't add latency to
        unrelated market analyses.
    """

    source_name: str
    categories: set[str] | None

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        """Fetch news items matching the query."""
        ...

    async def close(self) -> None:
        """Clean up resources."""
        ...
