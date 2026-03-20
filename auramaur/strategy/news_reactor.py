"""News-reactive trading — monitors RSS for breaking news and triggers fast analysis."""

from __future__ import annotations

import asyncio
import re

import structlog

from auramaur.data_sources.base import NewsItem
from auramaur.data_sources.rss import RSSSource
from auramaur.db.database import Database
from auramaur.exchange.models import Market
from auramaur.exchange.protocols import MarketDiscovery
from auramaur.strategy.engine import TradingEngine

log = structlog.get_logger()

# Common English stop words to strip from headlines before building search terms.
_STOP_WORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "not", "no", "nor", "so", "if", "then", "than", "that", "this",
    "these", "those", "what", "which", "who", "whom", "how", "when",
    "where", "why", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "only", "own", "same", "very",
    "just", "about", "above", "after", "again", "against", "before",
    "below", "between", "during", "into", "through", "under", "until",
    "over", "out", "up", "down", "off", "here", "there", "new", "says",
    "said", "also", "as", "can", "get", "got", "one", "two", "now",
    "still", "s", "t", "re", "ve", "d", "ll", "m",
}

# Words that are generic news fluff — not useful for Polymarket search.
_NEWS_FLUFF: set[str] = {
    "breaking", "update", "live", "latest", "report", "reports",
    "exclusive", "developing", "just", "watch", "opinion", "analysis",
    "sources", "source", "officials", "official", "according",
}

# Regex to detect capitalised proper-noun-like tokens.
_PROPER_NOUN_RE = re.compile(r"[A-Z][a-z]{2,}")


class NewsReactor:
    """Monitor RSS feeds for breaking news and trigger fast market analysis.

    The reactor polls RSS feeds on a short interval, extracts search terms from
    new headlines, searches Polymarket for related markets, and immediately
    triggers the full analysis pipeline on matches.  This gives us first-mover
    advantage — we can trade before the broader market adjusts to the news.
    """

    def __init__(
        self,
        rss_source: RSSSource,
        discovery: MarketDiscovery,
        engine: TradingEngine,
        db: Database,
        *,
        max_markets_per_story: int = 3,
        min_liquidity: float = 500.0,
    ) -> None:
        self._rss = rss_source
        self._discovery = discovery
        self._engine = engine
        self._db = db
        self._seen_ids: set[str] = set()
        self._max_markets_per_story = max_markets_per_story
        self._min_liquidity = min_liquidity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check_for_news(self) -> list[dict]:
        """Check RSS feeds for new stories and find related markets.

        Returns a list of analysis results (same shape as
        ``TradingEngine.analyze_market``).
        """
        # Fetch all RSS items without a keyword filter — we want everything.
        items = await self._rss.fetch(query="", limit=50)

        new_items = self._filter_new(items)
        if not new_items:
            return []

        log.info(
            "news_reactor.new_stories",
            count=len(new_items),
            titles=[i.title[:80] for i in new_items[:5]],
        )

        results: list[dict] = []
        for item in new_items:
            try:
                matches = await self._find_related_markets(item)
                for market in matches:
                    try:
                        result = await self._engine.analyze_market(market)
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        log.error(
                            "news_reactor.analysis_error",
                            market_id=market.id,
                            error=str(e),
                        )
            except Exception as e:
                log.error(
                    "news_reactor.search_error",
                    title=item.title[:80],
                    error=str(e),
                )

        if results:
            log.info(
                "news_reactor.cycle_complete",
                stories=len(new_items),
                analyses=len(results),
                trades=sum(1 for r in results if r.get("order")),
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_new(self, items: list[NewsItem]) -> list[NewsItem]:
        """Return only items we haven't processed yet."""
        new: list[NewsItem] = []
        for item in items:
            if item.id not in self._seen_ids:
                self._seen_ids.add(item.id)
                new.append(item)

        # Cap memory — keep only the last 2000 IDs.
        if len(self._seen_ids) > 2000:
            excess = len(self._seen_ids) - 2000
            # sets are unordered; trim arbitrary elements
            for _ in range(excess):
                self._seen_ids.pop()

        return new

    def _extract_search_terms(self, title: str) -> list[str]:
        """Extract key search terms from a news headline.

        Strategy:
        1. Find proper nouns (capitalised words) — these are usually the most
           informative tokens (names, countries, organisations).
        2. Keep remaining meaningful words after stripping stop words.
        3. Combine into 2-3 short search queries suitable for the Gamma API.

        Examples
        --------
        "Trump signs executive order on AI" -> ["Trump AI", "executive order AI"]
        "Fed raises interest rates by 50 basis points" -> ["Fed interest rates", "Fed basis points"]
        "Russia launches new offensive in Ukraine" -> ["Russia Ukraine", "Russia offensive Ukraine"]
        """
        # Tokenise, stripping punctuation.
        raw_tokens = re.findall(r"[A-Za-z']+", title)

        proper_nouns: list[str] = []
        keywords: list[str] = []

        for token in raw_tokens:
            lower = token.lower()
            if lower in _STOP_WORDS or lower in _NEWS_FLUFF:
                continue
            if _PROPER_NOUN_RE.fullmatch(token) or token.isupper():
                proper_nouns.append(token)
            else:
                keywords.append(token)

        # Build search queries.
        queries: list[str] = []

        # Query 1: proper nouns together (most targeted).
        if proper_nouns:
            queries.append(" ".join(proper_nouns[:4]))

        # Query 2: first proper noun + first couple of keywords.
        if proper_nouns and keywords:
            queries.append(f"{proper_nouns[0]} {' '.join(keywords[:2])}")

        # Query 3: just keywords if no proper nouns.
        if not proper_nouns and keywords:
            queries.append(" ".join(keywords[:3]))

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique: list[str] = []
        for q in queries:
            q_lower = q.lower()
            if q_lower not in seen:
                seen.add(q_lower)
                unique.append(q)

        return unique or [" ".join(raw_tokens[:3])]

    async def _find_related_markets(self, item: NewsItem) -> list[Market]:
        """Search Polymarket for markets related to a news item."""
        search_terms = self._extract_search_terms(item.title)

        all_markets: dict[str, Market] = {}  # deduplicate by market id

        for query in search_terms:
            markets = await self._discovery.search_markets(query, limit=10)
            for market in markets:
                if (
                    market.active
                    and market.liquidity >= self._min_liquidity
                    and market.id not in all_markets
                ):
                    all_markets[market.id] = market

        # Rank by liquidity (most liquid = most reliable pricing).
        ranked = sorted(all_markets.values(), key=lambda m: m.liquidity, reverse=True)

        selected = ranked[: self._max_markets_per_story]
        if selected:
            log.info(
                "news_reactor.markets_found",
                headline=item.title[:80],
                queries=search_terms,
                market_count=len(selected),
                market_ids=[m.id for m in selected],
            )

        return selected
