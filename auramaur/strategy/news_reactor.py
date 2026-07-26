"""News-reactive trading — monitors RSS for breaking news and triggers fast analysis."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from auramaur.experiments.strategies.news_reactor import (
    NewsMarketCandidate,
    NewsReactorRules,
    extract_proper_nouns,
    extract_search_terms,
    form_news_market_proposal,
)

from auramaur.strategy.protocols import ExecutionMode

from auramaur.data_sources.base import NewsItem
from auramaur.data_sources.rss import RSSSource
from auramaur.db.database import Database
from auramaur.exchange.models import Market
from auramaur.exchange.protocols import MarketDiscovery
from auramaur.strategy.engine import TradingEngine

if TYPE_CHECKING:
    from auramaur.nlp.news_triage import MaterialityTriage

log = structlog.get_logger()


class NewsReactor:
    """Monitor RSS feeds for breaking news and trigger fast market analysis.

    The reactor polls RSS feeds on a short interval, extracts search terms from
    new headlines, searches Polymarket for related markets, and immediately
    triggers the full analysis pipeline on matches.  This gives us first-mover
    advantage — we can trade before the broader market adjusts to the news.
    """

    name = "news_reactor"
    execution_mode = ExecutionMode.GATEWAY_SINGLE

    # Categories news-reaction structurally cannot price: a breaking headline
    # ("Team X beats Team Y") doesn't predict a LIVE MATCH OUTCOME the way it
    # predicts a policy/geopolitical event, yet the keyword search readily
    # matches team/player names. The news_speed paper book lost ~$65 over 3
    # esports events (Counter-Strike, Dota) doing exactly this. Block the
    # live-event-outcome categories so the reactor never flags them.
    BLOCKED_CATEGORIES = frozenset({"esports", "sports"})

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
        fast_analysis: bool = False,
        triage: "MaterialityTriage | None" = None,
    ) -> None:
        """Initialize the reactor.

        ``discoveries`` and ``engines`` are keyed by exchange name. Each news
        story is searched across all exchanges concurrently; matches on any
        exchange get flagged on the corresponding engine so the next
        strategic batch evaluates them.

        When ``fast_analysis`` is True (hybrid mode), the reactor additionally
        triggers a direct Claude analysis on the single highest-liquidity match
        per cycle, giving a speed advantage on breaking news.

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
        self._fast_analysis = fast_analysis
        self._analysis_semaphore = asyncio.Semaphore(1)
        # Optional local-LLM materiality pre-screen (fail-open: None or any
        # screen error preserves today's flag-everything behavior).
        self._triage = triage

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

        if self._triage is not None:
            self._triage.reset_cycle()

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
                        # Skip live-event-outcome categories news-reaction can't
                        # price (esports/sports). Classify off the stored label
                        # when present, else the question text — the loser
                        # esports markets carried the category, but match-ups can
                        # also be caught structurally.
                        if self._is_blocked_category(market):
                            log.info(
                                "news_reactor.category_skip",
                                market_id=market.id,
                                category=getattr(market, "category", "") or "",
                            )
                            continue
                        if self._triage is not None:
                            try:
                                score = await self._triage.screen(
                                    item.id, item.title, market.question)
                            except Exception as e:  # noqa: BLE001 — fail open
                                log.debug("news_reactor.triage_error",
                                          error=str(e)[:120])
                                score = None
                            if score is not None and score < self._triage.threshold:
                                log.info(
                                    "news_reactor.triage_skip",
                                    market_id=market.id,
                                    score=round(score, 2),
                                    headline=item.title[:80],
                                )
                                continue
                        engine.flag_market_from_news(market.id)
                        flagged.append({
                            "exchange": exchange_name,
                            "market_id": market.id,
                            "question": market.question,
                            "headline": item.title,
                            "liquidity": getattr(market, "liquidity", 0.0),
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

        # Hybrid fast-path: trigger immediate Claude analysis on the single
        # highest-liquidity flagged market (one per cycle, semaphore-guarded).
        if self._fast_analysis and flagged:
            best = max(flagged, key=lambda f: self._get_market_liquidity(f))
            engine = self._engines.get(best["exchange"])
            if engine and self._analysis_semaphore._value > 0:
                try:
                    async with self._analysis_semaphore:
                        market = await self._discoveries[best["exchange"]].get_market(best["market_id"])
                        if market:
                            log.info(
                                "news_reactor.fast_analysis",
                                market_id=market.id,
                                headline=best["headline"][:80],
                            )
                            result = await engine.analyze_market(
                                market, strategy_source="news_speed",
                            )
                            if result and result.get("order"):
                                best["fast_analysis"] = True
                except Exception as e:
                    log.error("news_reactor.fast_analysis_error", error=str(e))

        return flagged

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_market_liquidity(flagged_entry: dict) -> float:
        """Extract liquidity from a flagged entry for ranking."""
        return flagged_entry.get("liquidity", 0.0)

    @classmethod
    def _is_blocked_category(cls, market: Market) -> bool:
        """True if the market is a live-event-outcome (esports/sports) the
        reactor must not trade. Category-only on purpose: the proven loser
        markets all carried the label, and a fuzzy ``A vs B`` text fallback
        would false-block legitimate political head-to-heads (Trump vs Biden)."""
        category = (getattr(market, "category", "") or "").strip().lower()
        return category in cls.BLOCKED_CATEGORIES

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
        return extract_search_terms(title)

    def _extract_proper_nouns(self, title: str) -> list[str]:
        """Return just the real proper nouns in a headline, for match validation."""
        return extract_proper_nouns(title)

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
        return form_news_market_proposal(
            item.title,
            NewsMarketCandidate(
                market_id=market.id,
                question=market.question or "",
                description=market.description or "",
                category="",
                active=True,
                liquidity=max(float(market.liquidity or 0), self._min_liquidity),
                volume=float(market.volume or 0),
            ),
            NewsReactorRules(
                min_liquidity=self._min_liquidity,
                min_proper_nouns=self._min_proper_nouns,
                blocked_categories=frozenset(),
            ),
        ) is not None

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
                    proposal = form_news_market_proposal(
                        item.title,
                        NewsMarketCandidate(
                            market_id=m.id,
                            question=m.question or "",
                            description=m.description or "",
                            category=getattr(m, "category", "") or "",
                            active=bool(m.active),
                            liquidity=float(m.liquidity or 0),
                            volume=float(m.volume or 0),
                        ),
                        NewsReactorRules(
                            min_liquidity=self._min_liquidity,
                            min_proper_nouns=self._min_proper_nouns,
                            blocked_categories=self.BLOCKED_CATEGORIES,
                        ),
                    )
                    if proposal is not None and m.id not in all_markets:
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
