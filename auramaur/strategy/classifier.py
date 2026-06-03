"""Market category classification for exposure tracking."""

from __future__ import annotations

import re

# --- Priority patterns: checked first, before keyword scoring ---
# These categories have distinctive markers that should not fall through
# to generic keyword matching (e.g. "win" matching politics).

SPORTS_KEYWORDS: list[str] = [
    "nba", "nfl", "mlb", "ncaa", "soccer", "football", "championship", "super bowl",
    "world cup", "vs.", "o/u", "spread", "game", "match", "fc",
    "league", "premier", "serie a", "bundesliga", "ligue 1",
    "eredivisie", "champions league", "europa league", "nhl", "mls",
    # Individual-sport / event markets — these often lack a team-vs marker and
    # used to rely on the removed "winner" keyword, so without explicit markers
    # they fall to "other" (which is NOT blocked) and the bot would trade them.
    "golf", "pga", "masters", "ryder cup", "tennis", "wimbledon", "grand slam",
    "us open", "australian open", "french open", "atp", "wta",
    "grand prix", "nascar", "tour de france", "ufc", "boxing",
]
# Note: "winner" is intentionally NOT a sports marker — it appears in the
# resolution boilerplate of nearly every election/award market ("resolve
# according to the winner of the ... election"), which is exactly why priority
# matching below runs against the question, not the description.
# Sports markets phrased "<team> win on <date>", e.g. "Will Poland win on
# 2026-03-26?". A plain keyword can't express the year, so it's a pattern.
SPORTS_PATTERNS: list[str] = [r"win on 20\d\d", r"\bolympics?\b", r"\bf1\b"]

# Checked BEFORE sports so award-show markets ("Eurovision ... Jury Winner",
# "win the Oscar") aren't stolen by the sports "winner"/"win" markers.
ENTERTAINMENT_PRIORITY: list[str] = [
    "eurovision", "song contest", "oscar", "grammy", "academy award",
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

# --- General keyword scoring (used when no priority category matches) ---
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "economics": ["gdp", "inflation", "fed", "interest rate", "unemployment", "recession", "cpi", "jobs report", "treasury"],
    "crypto": ["bitcoin", "ethereum", "crypto", "btc", "eth", "defi", "nft", "token"],
    "tech": ["ai", "artificial intelligence", "openai", "google", "apple", "meta", "microsoft", "tech"],
    "entertainment": ["oscar", "grammy", "movie", "film", "tv", "show", "celebrity"],
    "science": ["climate", "nasa", "space", "vaccine", "covid", "health", "fda"],
    "legal": ["supreme court", "lawsuit", "trial", "verdict", "indictment", "ruling"],
}

# Politics is split US vs. international by COUNTRY CONTEXT, not handled in the
# generic dict above. Generic governance terms (president, election, vote, ...)
# describe both, so they must not by themselves pull a market into politics_us —
# that previously misfiled every foreign election (a French presidential race, a
# Bolivian president question) as politics_us, because politics_intl carried no
# governance or country markers to compete. We score US-specific markers,
# foreign country/demonym markers, and shared governance terms separately, then
# let country context decide the bucket (see classify_market).
POLITICS_US_KEYWORDS: list[str] = [
    "congress", "senate", "house", "democrat", "republican", "biden", "trump",
    "gop", "white house", "american", "governor", "midterm",
]
POLITICS_INTL_KEYWORDS: list[str] = [
    "ukraine", "russia", "china", "eu", "nato", "war", "un", "geopoliti",
    "france", "french", "germany", "german", "uk", "britain", "british",
    "canada", "canadian", "mexico", "mexican", "brazil", "brazilian",
    "india", "indian", "japan", "japanese", "israel", "israeli",
    "iran", "iranian", "venezuela", "argentina", "bolivia", "bolivian",
    "poland", "polish", "italy", "italian", "spain", "spanish",
    "turkey", "turkish", "taiwan", "pakistan", "nigeria",
    "south korea", "north korea",
]
POLITICS_GOVERNANCE_KEYWORDS: list[str] = [
    "president", "presidential", "election", "vote", "primary", "parliament",
    "prime minister", "referendum", "chancellor", "coalition",
]


def _compile_one(keyword: str) -> re.Pattern:
    """Compile a keyword into a word-boundary regex.

    Boundaries are anchored only where the keyword edge is alphanumeric, so
    short tokens can't match inside longer words ("nfl" no longer matches
    "NFLX", "eu" no longer matches "Eurovision", "war" no longer matches
    "Warren") while keywords with punctuation/whitespace edges ("spread:",
    "o/u", "°c") still match. Leading/trailing whitespace — historically used
    as a manual word boundary — is stripped and replaced by a real one.
    """
    k = keyword.strip()
    left = r"\b" if k[:1].isalnum() else ""
    right = r"\b" if k[-1:].isalnum() else ""
    return re.compile(left + re.escape(k) + right, re.IGNORECASE)


def _compile_many(keywords: list[str], extra_patterns: list[str] | None = None) -> list[re.Pattern]:
    res = [_compile_one(k) for k in keywords if k.strip()]
    res += [re.compile(p, re.IGNORECASE) for p in (extra_patterns or [])]
    return res


_ESPORTS_RE = _compile_many(ESPORTS_KEYWORDS)
_ENTERTAINMENT_PRIORITY_RE = _compile_many(ENTERTAINMENT_PRIORITY)
_SPORTS_RE = _compile_many(SPORTS_KEYWORDS, SPORTS_PATTERNS)
_WEATHER_RE = _compile_many(WEATHER_KEYWORDS)
_GAMBLING_RE = _compile_many(GAMBLING_KEYWORDS)

# Priority categories — checked in order before the general scoring.
_PRIORITY_CATEGORIES: list[tuple[str, list[re.Pattern]]] = [
    ("esports", _ESPORTS_RE),
    ("entertainment", _ENTERTAINMENT_PRIORITY_RE),
    ("sports", _SPORTS_RE),
    ("weather", _WEATHER_RE),
    ("gambling", _GAMBLING_RE),
]

_CATEGORY_RES: dict[str, list[re.Pattern]] = {
    cat: _compile_many(kws) for cat, kws in CATEGORY_KEYWORDS.items()
}

_POLITICS_US_RE = _compile_many(POLITICS_US_KEYWORDS)
_POLITICS_INTL_RE = _compile_many(POLITICS_INTL_KEYWORDS)
_POLITICS_GOV_RE = _compile_many(POLITICS_GOVERNANCE_KEYWORDS)


def classify_market(question: str, description: str = "") -> str:
    """Classify a market question into a category.

    Priority categories (esports, entertainment, sports, weather, gambling)
    are checked first using distinctive markers so they don't get
    misclassified by generic keyword overlap. All matching is on word
    boundaries to avoid substring collisions (e.g. "nfl" inside "NFLX").
    """
    # Priority categories match the QUESTION only. Descriptions are resolution
    # boilerplate ("resolve according to the winner of the ... election",
    # candidate "A vs. B" matchups) that reliably triggers sports/gambling
    # markers on non-sports markets. The fallthrough scoring below still uses
    # the description, where extra signal helps more than it hurts.
    q = question.lower()
    full = f"{question} {description}".lower()

    # 1. Check priority categories first (question only)
    for category, patterns in _PRIORITY_CATEGORIES:
        if any(p.search(q) for p in patterns):
            return category

    # 2. Fall through to general keyword scoring (question + description)
    scores: dict[str, int] = {}
    for category, patterns in _CATEGORY_RES.items():
        score = sum(1 for p in patterns if p.search(full))
        if score > 0:
            scores[category] = score

    # Politics: decide US vs. international by country context. Governance terms
    # (president/election/...) count toward whichever bucket the country markers
    # select, so a foreign election lands in politics_intl instead of defaulting
    # to politics_us. With no country marker, generic governance defaults to US
    # (the bulk of prediction-market election markets), preserving prior behavior.
    us = sum(1 for p in _POLITICS_US_RE if p.search(full))
    intl = sum(1 for p in _POLITICS_INTL_RE if p.search(full))
    gov = sum(1 for p in _POLITICS_GOV_RE if p.search(full))
    if intl and not us:
        scores["politics_intl"] = intl + gov
    elif us:
        scores["politics_us"] = us + gov
    elif gov:
        scores["politics_us"] = gov

    if not scores:
        return "other"

    return max(scores, key=scores.get)
