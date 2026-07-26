"""Pure market-routing decisions for news-reactor experiments."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "not", "no",
    "nor", "so", "if", "then", "than", "that", "this", "these", "those",
    "what", "which", "who", "whom", "how", "when", "where", "why", "all",
    "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "very", "just", "about", "above", "after", "again",
    "against", "before", "below", "between", "during", "into", "through", "under",
    "until", "over", "out", "up", "down", "off", "here", "there", "new", "says",
    "said", "also", "as", "can", "get", "got", "one", "two", "now", "still",
    "s", "t", "re", "ve", "d", "ll", "m",
})
NEWS_FLUFF = frozenset({
    "breaking", "update", "live", "latest", "report", "reports", "exclusive",
    "developing", "just", "watch", "opinion", "analysis", "sources", "source",
    "officials", "official", "according",
})
GENERIC_TITLE_WORDS = frozenset({
    "cheap", "best", "good", "great", "new", "old", "big", "small", "hot",
    "top", "bad", "nice", "fun", "cool", "easy", "hard", "fast", "slow",
    "free", "rich", "poor", "young", "full", "short", "long", "yes", "no",
    "now", "soon", "next", "last", "first", "final", "early", "late", "happy",
    "sad", "real", "fake", "true", "false", "every", "any", "much", "many",
    "some", "few", "less", "more", "here", "there", "why", "how", "who",
    "what", "when", "where", "thanks", "sorry", "hello", "goodbye", "welcome",
    "please", "stuff", "things", "ways", "reasons", "signs", "tips",
})
PROPER_NOUN_RE = re.compile(r"[A-Z][a-z]{2,}")


@dataclass(frozen=True)
class NewsMarketCandidate:
    market_id: str
    question: str
    description: str
    category: str
    active: bool
    liquidity: float
    volume: float


@dataclass(frozen=True)
class NewsReactorRules:
    min_liquidity: float = 2000.0
    min_proper_nouns: int = 1
    blocked_categories: frozenset[str] = frozenset({"esports", "sports"})


@dataclass(frozen=True)
class NewsMarketProposal:
    market_id: str
    headline: str
    activity: float
    matched_proper_nouns: tuple[str, ...]
    rationale: str


def extract_proper_nouns(title: str) -> list[str]:
    result: list[str] = []
    for token in re.findall(r"[A-Za-z']+", title):
        lower = token.lower()
        if lower in STOP_WORDS or lower in NEWS_FLUFF or lower in GENERIC_TITLE_WORDS:
            continue
        if PROPER_NOUN_RE.fullmatch(token) or (token.isupper() and len(token) >= 2):
            result.append(token)
    return result


def extract_search_terms(title: str) -> list[str]:
    raw_tokens = re.findall(r"[A-Za-z']+", title)
    proper_nouns: list[str] = []
    keywords: list[str] = []
    for token in raw_tokens:
        lower = token.lower()
        if lower in STOP_WORDS or lower in NEWS_FLUFF:
            continue
        capital = bool(PROPER_NOUN_RE.fullmatch(token) or (token.isupper() and len(token) >= 2))
        if capital and lower in GENERIC_TITLE_WORDS:
            capital = False
        if capital:
            proper_nouns.append(token)
        elif lower not in GENERIC_TITLE_WORDS:
            keywords.append(token)
    queries: list[str] = []
    if proper_nouns:
        queries.append(" ".join(proper_nouns[:4]))
    if proper_nouns and keywords:
        queries.append(f"{proper_nouns[0]} {' '.join(keywords[:2])}")
    if not proper_nouns and keywords:
        queries.append(" ".join(keywords[:3]))
    unique: list[str] = []
    seen: set[str] = set()
    for query in queries:
        if query.lower() not in seen:
            seen.add(query.lower())
            unique.append(query)
    return unique or [" ".join(raw_tokens[:3])]


def form_news_market_proposal(
    headline: str,
    candidate: NewsMarketCandidate,
    rules: NewsReactorRules,
) -> NewsMarketProposal | None:
    """Return a reproducible routing proposal, failing closed on invalid input."""
    if (
        not headline.strip()
        or not candidate.market_id
        or rules.min_proper_nouns < 0
        or not math.isfinite(rules.min_liquidity)
        or rules.min_liquidity < 0.0
        or not math.isfinite(candidate.liquidity)
        or not math.isfinite(candidate.volume)
        or candidate.liquidity < 0.0
        or candidate.volume < 0.0
        or not candidate.active
        or candidate.category.strip().lower() in rules.blocked_categories
    ):
        return None
    proper_nouns = extract_proper_nouns(headline)
    if len(proper_nouns) < rules.min_proper_nouns:
        return None
    activity = max(candidate.liquidity, candidate.volume)
    if activity < rules.min_liquidity:
        return None
    market_text = f"{candidate.question} {candidate.description}".lower()
    matched = tuple(noun for noun in proper_nouns if noun.lower() in market_text)
    if not matched:
        return None
    return NewsMarketProposal(
        market_id=candidate.market_id,
        headline=headline,
        activity=activity,
        matched_proper_nouns=matched,
        rationale=f"news match: {', '.join(matched)}",
    )
