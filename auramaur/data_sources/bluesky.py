"""Bluesky (AT Protocol) post search.

The public appview (``public.api.bsky.app``) began 403-ing
unauthenticated/datacenter search traffic — measured 2026-08-04: 29,743
fetches over 14 days, every one returning zero items while the old code
swallowed the 403 at DEBUG level and reported ok/empty upstream. Two
consequences fixed here:

* **Auth**: with ``bluesky_identifier`` + ``bluesky_app_password`` set, a
  session is created against ``bsky.social`` and searches carry the access
  JWT (refreshed once on 401/403, recreated on expiry). Without
  credentials the source still runs unauthenticated — it will likely 403,
  and that is now VISIBLE:
* **No silent failure**: a non-200 raises, so the aggregator's fetch
  telemetry records status='error' instead of a healthy-looking empty. A
  source failing 100% of the time must look broken in the funnel.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone

import aiohttp
import structlog

from auramaur.data_sources.base import NewsItem

logger = structlog.get_logger(__name__)

_PUBLIC_SEARCH = "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"
_AUTH_BASE = "https://bsky.social/xrpc"
# A post that is nothing but URL(s) and whitespace/ellipsis — no analyzable
# content of its own (observed day one: a bare "youtu.be/..." ranked into
# an analysis context).
_LINK_ONLY_RE = re.compile(r"(?:\s|\.{3}|…|https?://\S+|[\w.-]+\.\w{2,6}/\S*)+")


class BlueskyFetchError(RuntimeError):
    """Non-200 from the search endpoint — must surface in fetch telemetry."""


class BlueskySource:
    """Search Bluesky posts for recent takes on a topic."""

    source_name: str = "bluesky"
    # Category-agnostic — fires for any query, like the rest of the text
    # news sources. Bluesky discussion cuts across every topic we trade.
    categories: set[str] | None = None

    def __init__(self, identifier: str = "", app_password: str = "") -> None:
        self._identifier = identifier.strip()
        self._app_password = app_password
        self._jwt: str = ""

    @property
    def _authed(self) -> bool:
        return bool(self._identifier and self._app_password)

    async def _create_session(self, session: aiohttp.ClientSession) -> None:
        async with session.post(
                f"{_AUTH_BASE}/com.atproto.server.createSession",
                json={"identifier": self._identifier,
                      "password": self._app_password},
                timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                raise BlueskyFetchError(
                    f"createSession returned {resp.status}")
            data = await resp.json()
            self._jwt = data.get("accessJwt") or ""
            if not self._jwt:
                raise BlueskyFetchError("createSession returned no accessJwt")

    async def _search(self, session: aiohttp.ClientSession, params: dict,
                      *, retried: bool = False) -> dict:
        if self._authed:
            if not self._jwt:
                await self._create_session(session)
            url = f"{_AUTH_BASE}/app.bsky.feed.searchPosts"
            headers = {"Authorization": f"Bearer {self._jwt}"}
        else:
            url, headers = _PUBLIC_SEARCH, {}
        async with session.get(url, params=params, headers=headers,
                               timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status in (401, 403) and self._authed and not retried:
                # Expired/invalidated token: recreate the session once.
                self._jwt = ""
                return await self._search(session, params, retried=True)
            if resp.status != 200:
                raise BlueskyFetchError(f"search returned {resp.status}")
            return await resp.json()

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        if not query or len(query.strip()) < 3:
            return []

        params = {
            "q": query,
            "limit": str(min(limit, 50)),
            "sort": "latest",
            # Freshness bound (2026-08-04): latest-sort already biases
            # recent, but a sparse query happily returns months-old takes
            # (evergreen FDR quotes ranked into context on day one). Ten
            # days covers every horizon the evidence pipeline cares about.
            "since": (datetime.now(timezone.utc)
                      - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)) as session:
            data = await self._search(session, params)

        items: list[NewsItem] = []
        seen_text: set[str] = set()
        for post in (data.get("posts") or [])[:limit]:
            record = post.get("record") or {}
            text = (record.get("text") or "").strip()
            if not text:
                continue
            # Link-only posts carry no analyzable content of their own.
            if _LINK_ONLY_RE.fullmatch(text):
                continue
            # Reposts/quote-chains: same words, different URIs. One context
            # slot per distinct take (first wins = the latest, per sort).
            normalized = " ".join(text.casefold().split())
            if normalized in seen_text:
                continue
            seen_text.add(normalized)

            author = (post.get("author") or {}).get("handle") or "unknown"
            uri = post.get("uri") or ""
            created_at = record.get("createdAt") or ""
            try:
                published = datetime.fromisoformat(
                    created_at.replace("Z", "+00:00"))
            except Exception:
                published = datetime.now(timezone.utc)

            likes = post.get("likeCount") or 0
            reposts = post.get("repostCount") or 0
            replies = post.get("replyCount") or 0
            engagement = likes + reposts + replies

            # Noise floor (2026-08-04, tightened from <2-and-<80): drive-by
            # one-liners ("Winning is the only thing.") cleared the old gate
            # on 2 likes. Substantial text passes regardless; short text now
            # needs REAL engagement — which deliberately keeps the case that
            # matters, a terse breaking post from a followed account.
            if len(text) < 40 and engagement < 10:
                continue
            if len(text) < 80 and engagement < 2:
                continue

            # Convert at:// URI to web URL (bsky.app post permalinks).
            web_url = ""
            if uri.startswith("at://"):
                parts = uri[5:].split("/")
                if len(parts) >= 3:
                    web_url = (f"https://bsky.app/profile/{parts[0]}"
                               f"/post/{parts[-1]}")

            # Relevance = freshness × engagement, not engagement alone
            # (2026-08-04): the old /50 engagement-only score gave nearly
            # every post ~0.0, so the ranker could not tell bluesky's gold
            # from its mud — and it priced a 5-minute-old breaking post at
            # exactly zero, the moment such a post is MOST informative.
            # Fresh-but-quiet now floors at ~0.25; stale-and-quiet decays
            # toward zero; viral still caps at 2.5 (HN caps at 3.0 — same
            # scale family).
            age_days = max(0.0, (datetime.now(timezone.utc)
                                 - published).total_seconds() / 86400.0)
            freshness = max(0.2, 1.0 - age_days / 10.0)
            engagement_part = min((likes + reposts * 2) / 50.0, 2.0)
            items.append(NewsItem(
                id=hashlib.md5(f"bsky:{uri or text[:60]}".encode()).hexdigest(),
                source="bluesky",
                title=text.split("\n", 1)[0][:160],
                content=(f"@{author}: {text[:400]} (likes:{likes} "
                         f"reposts:{reposts} replies:{replies})"),
                url=web_url,
                published_at=published,
                relevance_score=round(
                    min(freshness * (0.25 + engagement_part), 2.5), 3),
            ))

        return items

    async def close(self) -> None:
        return None
