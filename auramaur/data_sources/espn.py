"""ESPN scoreboards + news via the public ``site.api.espn.com`` endpoints.

No authentication. ESPN exposes JSON at ``site.api.espn.com/apis/site/v2/sports/<league>/scoreboard``
and ``/news`` that cover scores, event status, and top headlines.
We surface today's scores + top headlines for the major US leagues so
sports markets get structured context.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone

import aiohttp
import structlog

from auramaur.data_sources.base import NewsItem

logger = structlog.get_logger(__name__)

# league_path → human short name. We probe each; failures (off-season, API
# hiccups) are silently skipped.
_LEAGUES: dict[str, str] = {
    "football/nfl": "NFL",
    "basketball/nba": "NBA",
    "baseball/mlb": "MLB",
    "hockey/nhl": "NHL",
    "soccer/usa.1": "MLS",
    "basketball/wnba": "WNBA",
    "basketball/mens-college-basketball": "NCAAM",
    "football/college-football": "NCAAF",
}

_BASE = "https://site.api.espn.com/apis/site/v2/sports"


class ESPNSource:
    """Today's scores + top headlines across major leagues, filtered by query."""

    source_name: str = "espn"
    categories: set[str] | None = {"sports"}

    async def _fetch_one(self, session: aiohttp.ClientSession, league_path: str, short: str) -> list[NewsItem]:
        items: list[NewsItem] = []

        # Scoreboard
        try:
            url = f"{_BASE}/{league_path}/scoreboard"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status == 200:
                    sb = await resp.json()
                    for event in (sb.get("events") or [])[:5]:
                        name = event.get("name") or ""
                        status = (event.get("status") or {}).get("type", {}).get("description", "")
                        comp = (event.get("competitions") or [{}])[0]
                        teams = comp.get("competitors") or []
                        score_line = ""
                        if len(teams) == 2 and teams[0].get("score") is not None:
                            a = teams[0]; b = teams[1]
                            score_line = f"{a.get('team', {}).get('displayName', '?')} {a.get('score', '?')} - {b.get('team', {}).get('displayName', '?')} {b.get('score', '?')}"
                        items.append(NewsItem(
                            id=hashlib.md5(f"espn:{event.get('id')}".encode()).hexdigest(),
                            source="espn",
                            title=f"[{short}] {name} — {status}",
                            content=score_line or f"{short} event: {status}",
                            url=(event.get("links") or [{}])[0].get("href", "") if event.get("links") else "",
                            published_at=datetime.now(timezone.utc),
                            relevance_score=1.5,
                        ))
        except Exception as e:
            logger.debug("espn.scoreboard_error", league=league_path, error=str(e)[:120])

        # News headlines
        try:
            url = f"{_BASE}/{league_path}/news"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status == 200:
                    news = await resp.json()
                    for a in (news.get("articles") or [])[:5]:
                        headline = a.get("headline") or ""
                        if not headline:
                            continue
                        pub_raw = a.get("published") or ""
                        try:
                            published = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
                        except Exception:
                            published = datetime.now(timezone.utc)
                        link_href = ""
                        for link in a.get("links", {}).get("web", {}).values() if isinstance(a.get("links", {}).get("web", {}), dict) else []:
                            if isinstance(link, dict) and link.get("href"):
                                link_href = link["href"]
                                break
                        items.append(NewsItem(
                            id=hashlib.md5(f"espn:news:{a.get('id', headline)}".encode()).hexdigest(),
                            source="espn",
                            title=f"[{short}] {headline}",
                            content=a.get("description") or "",
                            url=link_href,
                            published_at=published,
                            relevance_score=1.2,
                        ))
        except Exception as e:
            logger.debug("espn.news_error", league=league_path, error=str(e)[:120])

        return items

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        q_tokens = [t for t in (query or "").lower().split() if len(t) > 2]

        async with aiohttp.ClientSession() as session:
            batches = await asyncio.gather(
                *(self._fetch_one(session, path, short) for path, short in _LEAGUES.items())
            )

        all_items: list[NewsItem] = []
        for b in batches:
            all_items.extend(b)

        # If the query has tokens, prefer items whose title/content mentions them;
        # otherwise return all up to the limit, freshest first.
        def _score(it: NewsItem) -> int:
            hay = f"{it.title} {it.content}".lower()
            return sum(1 for t in q_tokens if t in hay)

        all_items.sort(key=lambda it: (_score(it), it.published_at), reverse=True)
        return all_items[:limit]

    async def close(self) -> None:
        return None
