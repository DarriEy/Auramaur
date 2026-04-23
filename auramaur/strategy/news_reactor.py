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

# Common words that start sentences and get spuriously classified as proper
# nouns because the RSS headline title-cases them. Matching against these
# lets clickbait like "Cheap stuff that doesn't suck" promote "Cheap" to a
# proper noun and turn into a false market match. Keep lowercase.
_GENERIC_TITLE_WORDS: set[str] = {
    "cheap", "best", "good", "great", "new", "old", "big", "small",
    "hot", "top", "bad", "nice", "fun", "cool", "easy", "hard", "fast",
    "slow", "free", "rich", "poor", "young", "full", "short", "long",
    "yes", "no", "now", "soon", "next", "last", "first", "final",
    "early", "late", "happy", "sad", "real", "fake", "true", "false",
    "every", "any", "much", "many", "some", "few", "less", "more",
    "here", "there", "why", "how", "who", "what", "when", "where",
    "thanks", "sorry", "hello", "goodbye", "welcome", "please",
    "cheap", "stuff", "things", "ways", "reasons", "signs", "tips",
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
        discoveries: dict[str, MarketDiscovery],
        engines: dict[str, TradingEngine],
        db: Database,
        *,
        max_markets_per_story: int = 3,
        min_liquidity: float = 2000.0,
        min_proper_nouns: int = 1,
    ) -> None:
        """Initialize the reactor.

        ``discoveries`` and ``engines`` are keyed by exchange name. Each news
        story is searched across all exchanges concurrently; matches on any
        exchange get flagged on the corresponding engine so the next
        strategic batch evaluates them. We no longer call analyze_market
        per-match — that spent two Claude calls per headline on junk
        matches like clickbait RSS entries.

        ``min_proper_nouns`` controls how strict headline filtering is.
        Headlines that don't yield at least this many real proper nouns
        are skipped (drops low-signal clickbait).
        """
        self._rss = rss_source
        self._discoveries = discoveries
        self._engines = engines
        self._db = db
        self._seen_ids: set[str] = set()
        self._max_markets_per_story = max_markets_per_story
        self._min_liquidity = min_liquidity
        self._min_proper_nouns = min_proper_nouns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check_for_news(self) -> list[dict]:
        """Check RSS feeds for new stories and flag related markets.

        Matched markets are registered with their exchange's engine so the
        next strategic batch picks them up. Returns a list of descriptors
        (for callers that want to see what got flagged) — NOT analysis
        results, since no Claude calls happen here anymore.
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

        flagged: list[dict] = []
        for item in new_items:
            try:
                matches_by_exchange = await self._find_related_markets(item)
                for exchange_name, markets in matches_by_exchange.items():
                    engine = self._engines.get(exchange_name)
                    if engine is None:
                        continue
                    for market in markets:
                        engine.flag_market_from_news(market.id)
                        flagged.append({
                            "exchange": exchange_name,
                            "market_id": market.id,
                            "question": market.question,
                            "headline": item.title,
                        })
            except Exception as e:
                log.error(
                    "news_reactor.search_error",
                    title=item.title[:80],
                    error=str(e),
                )

        if flagged:
            log.info(
                "news_reactor.flagged",
                stories=len(new_items),
                markets_flagged=len(flagged),
            )

        return flagged

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

        # The first token of a headline is almost always title-cased even if
        # it's a generic word ("Cheap", "Best", "New"), so ignore its casing
        # when deciding whether it's a real proper noun — fall through to the
        # generic-title-words filter below.
        for idx, token in enumerate(raw_tokens):
            lower = token.lower()
            if lower in _STOP_WORDS or lower in _NEWS_FLUFF:
                continue
            # Don't treat sentence-starting title-case as a proper noun.
            treat_as_capital = (_PROPER_NOUN_RE.fullmatch(token) or (token.isupper() and len(token) >= 2))
            if treat_as_capital and lower in _GENERIC_TITLE_WORDS:
                treat_as_capital = False
            if treat_as_capital:
                proper_nouns.append(token)
            elif lower not in _GENERIC_TITLE_WORDS:
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

    def _extract_proper_nouns(self, title: str) -> list[str]:
        """Return just the real proper nouns in a headline, for match validation."""
        proper_nouns: list[str] = []
        for token in re.findall(r"[A-Za-z']+", title):
            lower = token.lower()
            if lower in _STOP_WORDS or lower in _NEWS_FLUFF:
                continue
            if lower in _GENERIC_TITLE_WORDS:
                continue
            if _PROPER_NOUN_RE.fullmatch(token) or (token.isupper() and len(token) >= 2):
                proper_nouns.append(token)
        return proper_nouns

    @staticmethod
    def _market_text(m: Market) -> str:
        return f"{m.question or ''} {m.description or ''}".lower()

    def _match_is_credible(self, item: NewsItem, market: Market) -> bool:
        """Require at least one headline proper noun to appear in the market's
        text before accepting a match.

        This is the guard that stops clickbait RSS entries like
        "Cheap stuff that doesn't suck, take 3" from being routed to
        unrelated markets just because the Gamma API search returned
        fuzzy results.
        """
        proper_nouns = self._extract_proper_nouns(item.title)
        if not proper_nouns:
            return False
        mtext = self._market_text(market)
        return any(p.lower() in mtext for p in proper_nouns)

    async def _find_related_markets(
        self, item: NewsItem,
    ) -> dict[str, list[Market]]:
        """Search all configured exchanges for markets related to a news item.

        Returns a mapping of exchange name → ranked list of matching markets
        (top ``max_markets_per_story`` per exchange, liquidity-sorted).
        Headlines with too few proper nouns are skipped entirely.
        """
        proper_nouns = self._extract_proper_nouns(item.title)
        if len(proper_nouns) < self._min_proper_nouns:
            log.debug(
                "news_reactor.headline_skip_low_signal",
                title=item.title[:80],
                proper_nouns=proper_nouns,
            )
            return {}

        search_terms = self._extract_search_terms(item.title)

        async def _search_one(name: str, disc: MarketDiscovery) -> tuple[str, list[Market]]:
            all_markets: dict[str, Market] = {}
            for query in search_terms:
                try:
                    markets = await disc.search_markets(query, limit=10)
                except Exception as e:
                    log.debug("news_reactor.search_exchange_error", exchange=name, error=str(e))
                    continue
                for m in markets:
                    # Kalshi reports thin top-of-book liquidity and high volume;
                    # fall back to volume when liquidity is zero so we don't
                    # filter out active Kalshi markets.
                    activity = max(m.liquidity or 0, m.volume or 0)
                    if (m.active
                            and activity >= self._min_liquidity
                            and m.id not in all_markets
                            and self._match_is_credible(item, m)):
                        all_markets[m.id] = m
            ranked = sorted(all_markets.values(), key=lambda m: m.liquidity, reverse=True)
            return (name, ranked[: self._max_markets_per_story])

        results = await asyncio.gather(
            *[_search_one(n, d) for n, d in self._discoveries.items()]
        )
        matches: dict[str, list[Market]] = {name: markets for name, markets in results if markets}

        if matches:
            log.info(
                "news_reactor.markets_found",
                headline=item.title[:80],
                queries=search_terms,
                total_matches=sum(len(v) for v in matches.values()),
                by_exchange={k: [m.id for m in v] for k, v in matches.items()},
            )

        return matches
