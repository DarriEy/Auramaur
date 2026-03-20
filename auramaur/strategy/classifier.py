"""Market category classification for exposure tracking."""

from __future__ import annotations

import re

# --- Priority patterns: checked first, before keyword scoring ---
# These categories have distinctive markers that should not fall through
# to generic keyword matching (e.g. "win" matching politics).

SPORTS_KEYWORDS: list[str] = [
    "nba", "nfl", "mlb", "soccer", "football", "championship", "super bowl",
    "world cup", "vs.", "o/u", "spread", "winner", "game", "match", "fc ",
    "win on 202", "league", "premier", "serie a", "bundesliga", "ligue 1",
    "eredivisie", "champions league", "europa league", "nhl", "mls",
]

WEATHER_KEYWORDS: list[str] = [
    "temperature", "°c", "°f", "weather", "forecast", "rain", "snow", "wind",
]

ESPORTS_KEYWORDS: list[str] = [
    "lol:", "cs2", "dota", "valorant", "esport", "game 1 winner",
    "game 2 winner", "game 3 winner",
]

GAMBLING_KEYWORDS: list[str] = [
    "o/u", "spread:", "moneyline", "over/under",
]

# Priority categories — checked in order before the general scoring.
PRIORITY_CATEGORIES: list[tuple[str, list[str]]] = [
    ("esports", ESPORTS_KEYWORDS),
    ("sports", SPORTS_KEYWORDS),
    ("weather", WEATHER_KEYWORDS),
    ("gambling", GAMBLING_KEYWORDS),
]

# --- General keyword scoring (used when no priority category matches) ---
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "politics_us": ["president", "congress", "senate", "house", "democrat", "republican", "biden", "trump", "election", "vote", "primary", "gop"],
    "politics_intl": ["ukraine", "russia", "china", "eu", "nato", "war", "un ", "geopoliti"],
    "economics": ["gdp", "inflation", "fed ", "interest rate", "unemployment", "recession", "cpi", "jobs report", "treasury"],
    "crypto": ["bitcoin", "ethereum", "crypto", "btc", "eth", "defi", "nft", "token"],
    "tech": ["ai ", "artificial intelligence", "openai", "google", "apple", "meta", "microsoft", "tech"],
    "entertainment": ["oscar", "grammy", "movie", "film", "tv ", "show", "celebrity"],
    "science": ["climate", "nasa", "space", "vaccine", "covid", "health", "fda"],
    "legal": ["supreme court", "lawsuit", "trial", "verdict", "indictment", "ruling"],
}


def classify_market(question: str, description: str = "") -> str:
    """Classify a market question into a category.

    Priority categories (sports, weather, esports, gambling) are checked
    first using distinctive markers so they don't get misclassified by
    generic keyword overlap (e.g. "win" matching politics keywords).
    """
    text = f"{question} {description}".lower()

    # 1. Check priority categories first
    for category, keywords in PRIORITY_CATEGORIES:
        if any(kw in text for kw in keywords):
            return category

    # 2. Fall through to general keyword scoring
    scores: dict[str, int] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[category] = score

    if not scores:
        return "other"

    return max(scores, key=scores.get)
