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
    relevance_score: float = 0.0
    keywords: list[str] = Field(default_factory=list)


@runtime_checkable
class DataSource(Protocol):
    """Protocol that all data sources must implement."""

    source_name: str

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        """Fetch news items matching the query."""
        ...

    async def close(self) -> None:
        """Clean up resources."""
        ...
