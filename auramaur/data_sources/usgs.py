"""USGS earthquake data source — seismic events in near-real-time.

No API key required. The USGS publishes GeoJSON feeds of earthquakes by
time window and magnitude threshold. Used for markets that predict
earthquake occurrence, magnitude, or casualties.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import aiohttp
import structlog

from auramaur.data_sources.base import NewsItem

logger = structlog.get_logger(__name__)

# GeoJSON feeds. Past 30 days, M4.5+ strikes a balance between noise and
# relevance for market-moving events.
_FEED_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_month.geojson"


class USGSSource:
    """Surface recent significant earthquakes as evidence items."""

    source_name: str = "usgs"
    # Fire on seismic-adjacent markets only. "weather" is where the classifier
    # currently drops earthquake markets (it's more generic "natural events").
    categories: set[str] | None = {"weather", "science"}

    def __init__(self, min_magnitude: float = 4.5) -> None:
        self._min_magnitude = min_magnitude

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        items: list[NewsItem] = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(_FEED_URL, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.warning("usgs.fetch_error", status=resp.status)
                        return []
                    data = await resp.json()
        except Exception as e:
            logger.warning("usgs.fetch_exception", error=str(e)[:120])
            return []

        features = data.get("features", []) or []
        query_tokens = [t for t in (query or "").lower().split() if len(t) > 3]

        # Score each feature: +1 if it matches a query token, else 0. Then
        # return matches first (by score desc, time desc), filling with most
        # recent up to the limit. For seismic markets we want recent quake
        # activity globally even when the question is about a specific place
        # — absence of activity elsewhere is also a signal.
        scored: list[tuple[int, int, dict]] = []  # (score, time, feat)
        for feat in features:
            props = feat.get("properties") or {}
            mag = props.get("mag")
            if mag is None or mag < self._min_magnitude:
                continue
            place = (props.get("place") or "").lower()
            title = (props.get("title") or "").lower()
            score = sum(1 for tok in query_tokens if tok in title or tok in place)
            ts = props.get("time") or 0
            scored.append((score, ts, feat))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        for _score, _ts, feat in scored[:limit]:
            props = feat.get("properties") or {}
            mag = props.get("mag")
            place = props.get("place") or ""
            title = props.get("title") or f"M{mag} - {place}"
            url = props.get("url") or ""
            ts = props.get("time")
            try:
                published = datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc)
            except Exception:
                published = datetime.now(timezone.utc)

            item_id = hashlib.md5(f"usgs:{feat.get('id', title)}".encode()).hexdigest()
            items.append(NewsItem(
                id=item_id,
                source="usgs",
                title=title,
                content=f"Magnitude {mag} earthquake at {place}. Tsunami: {bool(props.get('tsunami'))}.",
                url=url,
                published_at=published,
            ))

        return items

    async def close(self) -> None:
        return None
