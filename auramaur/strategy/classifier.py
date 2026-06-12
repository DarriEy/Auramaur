"""Market category classification for exposure tracking."""

from __future__ import annotations

import re

# --- Priority patterns: checked first, before keyword scoring ---
# These categories have distinctive markers that should not fall through
# to generic keyword matching (e.g. "win" matching politics).

# STRONG markers are sport/league/tournament names that are unambiguous even
# inside resolution boilerplate, so they are matched against the DESCRIPTION
# too. Many sports questions carry no marker at all ("Libema Open: Linette vs
# Pohankova") while their description says "the tennis match between ..." —
# question-only matching let those fall through to keyword scoring, where
# boilerplate words ("primary", "American League") filed them as politics_us.
SPORTS_STRONG_KEYWORDS: list[str] = [
    "nba", "nfl", "mlb", "ncaa", "soccer", "football", "super bowl",
    "world cup", "serie a", "bundesliga", "ligue 1", "premier league",
    "eredivisie", "champions league", "europa league", "nhl", "mls",
    # Individual-sport / event markets — these often lack a team-vs marker and
    # used to rely on the removed "winner" keyword, so without explicit markers
    # they fall to "other" (which is NOT blocked) and the bot would trade them.
    "golf", "pga", "masters", "ryder cup", "tennis", "wimbledon", "grand slam",
    "us open", "australian open", "french open", "atp", "wta",
    "grand prix", "nascar", "tour de france", "ufc", "boxing",
]
# WEAK markers are matchup/betting-line syntax and generic words that appear
# in non-sports text too ("match", "game", "spread", a candidate "A vs. B"),
# so they match the QUESTION only.
SPORTS_WEAK_KEYWORDS: list[str] = [
    "championship", "vs.", "o/u", "spread", "game", "match", "fc",
    "league", "premier",
]
# Note: "winner" is intentionally NOT a sports marker — it appears in the
# resolution boilerplate of nearly every election/award market ("resolve
# according to the winner of the ... election"), which is exactly why weak
# markers match only the question.
# Sports markets phrased "<team> win on <date>", e.g. "Will Poland win on
# 2026-03-26?". A plain keyword can't express the year, so it's a pattern.
# "vs" without the period ("Linette vs Pohankova") is a word-boundary pattern
# so it can't match inside words.
SPORTS_PATTERNS: list[str] = [r"win on 20\d\d", r"\bolympics?\b", r"\bf1\b",
                              r"\bvs\b"]

# Checked BEFORE sports so award-show markets ("Eurovision ... Jury Winner",
# "win the Oscar") aren't stolen by the sports "winner"/"win" markers.
ENTERTAINMENT_PRIORITY: list[str] = [
    "eurovision", "song contest", "oscar", "grammy", "academy award",
]

WEATHER_KEYWORDS: list[str] = [
    "temperature", "°c", "°f", "weather", "forecast", "rain", "snow", "wind",
]

ESPORTS_KEYWORDS: list[str] = [
    "lol:", "cs2", "dota", "valorant", "esport", "esports", "game 1 winner",
    "game 2 winner", "game 3 winner",
    # Esports must out-rank sports: CS/Dota matchups read like sports
    # ("X vs Y", "Game 2") and the sports markers would steal them — and
    # sports is a BLOCKED category, so the steal false-blocks esports.
    "counter-strike", "league of legends", "roshan",
    "iem", "esl", "pgl", "blast open", "faceit",
]
ESPORTS_PATTERNS: list[str] = [r"\bmap \d"]

# Description-safe esports markers (game titles can't appear in non-esports
# resolution boilerplate). Checked against the full text, like the strong
# sports markers below — and before them, since esports matchups also carry
# generic sports phrasing.
ESPORTS_STRONG_KEYWORDS: list[str] = [
    "counter-strike", "cs2", "dota", "valorant", "esports",
    "league of legends",
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
# "american" is intentionally absent: it matches "American League" (MLB),
# "American Music Awards", "American Airlines" — it filed the Mariners'
# AL-West market as politics_us. Real US-politics markets carry stronger
# markers below.
# "house" appears as a bare keyword in foreign-parliament markets too ("Dutch
# House of Representatives dissolved?"), so it's matched only in its
# US-specific phrasings (patterns below).
POLITICS_US_KEYWORDS: list[str] = [
    "congress", "senate", "democrat", "republican", "biden", "trump",
    "gop", "white house", "governor", "midterm", "vance", "scotus",
]
POLITICS_US_PATTERNS: list[str] = [
    r"\bhouse (seat|race|majority)\b", r"\bcontrol (of )?the house\b",
    r"\bspeaker of the house\b", r"\bu\.?s\.? house\b",
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
    # European/other countries seen mislabeled politics_us in production
    # ("Dutch House of Representatives dissolved" hit the US "house" marker
    # with no intl marker to compete).
    "netherlands", "dutch", "denmark", "danish", "greenland",
    "norway", "norwegian", "sweden", "swedish", "hungary", "hungarian",
    "austria", "austrian", "kenya", "kenyan", "congo",
]
# "primary" is intentionally absent: description boilerplate ("primary
# listing", "primary market") filed SpaceX-IPO and tennis markets as
# politics_us. Election-primary markets always carry a party/office marker
# (democrat/republican/governor/senate) that classifies them anyway.
POLITICS_GOVERNANCE_KEYWORDS: list[str] = [
    "president", "presidential", "election", "vote", "parliament",
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


_ESPORTS_RE = _compile_many(ESPORTS_KEYWORDS, ESPORTS_PATTERNS)
_ESPORTS_STRONG_RE = _compile_many(ESPORTS_STRONG_KEYWORDS)
_ENTERTAINMENT_PRIORITY_RE = _compile_many(ENTERTAINMENT_PRIORITY)
_SPORTS_STRONG_RE = _compile_many(SPORTS_STRONG_KEYWORDS)
_SPORTS_RE = _compile_many(SPORTS_STRONG_KEYWORDS + SPORTS_WEAK_KEYWORDS,
                           SPORTS_PATTERNS)
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

_POLITICS_US_RE = _compile_many(POLITICS_US_KEYWORDS, POLITICS_US_PATTERNS)
_POLITICS_INTL_RE = _compile_many(POLITICS_INTL_KEYWORDS)
_POLITICS_GOV_RE = _compile_many(POLITICS_GOVERNANCE_KEYWORDS)


# --- Venue taxonomy: Polymarket event tags → internal categories ---
# Gamma markets carry no category field, but their parent EVENTS carry
# curated tags ("Tennis", "Crypto", "France"). These are authoritative when
# present — keyword scoring is the fallback, not the primary. Only
# unambiguous tags are mapped; bare "Politics"/"Elections" (US? intl?) fall
# through to keyword classification. Order matters: first hit wins.
VENUE_TAG_CATEGORIES: list[tuple[str, frozenset[str]]] = [
    ("esports", frozenset({
        "esports", "league of legends", "cs2", "counter-strike", "dota 2",
        "valorant",
    })),
    ("sports", frozenset({
        "sports", "nba", "nfl", "mlb", "nhl", "mls", "soccer", "tennis",
        "golf", "epl", "premier league", "la liga", "serie a", "bundesliga",
        "champions league", "ufc", "mma", "boxing", "f1", "formula 1",
        "nascar", "olympics", "ncaa", "college football",
        "college basketball", "world cup", "fifa", "cricket", "rugby",
        "hockey", "baseball", "basketball", "football", "atp", "wta",
    })),
    ("crypto", frozenset({
        "crypto", "bitcoin", "ethereum", "solana", "memecoins",
        "crypto listings", "defi", "stablecoins", "airdrops",
    })),
    ("politics_us", frozenset({
        "us politics", "u.s. politics", "us-current-affairs", "trump",
        "white house", "congress", "us elections", "midterms",
        "2026 midterms", "scotus", "supreme court",
    })),
    ("politics_intl", frozenset({
        "world", "geopolitics", "global politics", "world elections",
        "ukraine", "russia", "china", "israel", "iran", "middle east",
        "gaza", "france", "uk", "germany", "india", "north korea", "nato",
        "canada", "mexico", "brazil", "japan", "taiwan",
    })),
    ("economics", frozenset({
        "economy", "economics", "fed", "fed rates", "inflation", "macro",
        "interest rates", "jobs report", "gdp", "tariffs", "treasuries",
    })),
    ("tech", frozenset({
        "tech", "ai", "big tech", "openai", "science & tech", "ipos",
    })),
    ("entertainment", frozenset({
        "pop culture", "celebrities", "movies", "music", "entertainment",
        "awards", "oscars", "grammys", "gaming",
    })),
    ("weather", frozenset({
        "weather", "climate and weather", "temperature", "hurricanes",
    })),
    ("science", frozenset({
        "science", "climate", "space", "nasa", "health", "pandemics",
    })),
]


def classify_tags(tag_labels: list[str] | None) -> str:
    """Map venue tag labels to an internal category, '' when inconclusive.

    Returns the first category (in VENUE_TAG_CATEGORIES order) any tag maps
    to. An empty result means the venue taxonomy doesn't decide this market —
    callers fall back to classify_market().
    """
    if not tag_labels:
        return ""
    labels = {(t or "").strip().lower() for t in tag_labels}
    for category, known in VENUE_TAG_CATEGORIES:
        if labels & known:
            return category
    return ""


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

    # 1b. Strong markers (game titles, sport/league/tournament names) also
    # match the description: "Libema Open: Linette vs Pohankova" carries no
    # marker in the question while its description says "the tennis match
    # between ...". Strong names are unambiguous in boilerplate, unlike the
    # weak markers. Esports first — CS/Dota matchups also read like sports.
    if any(p.search(full) for p in _ESPORTS_STRONG_RE):
        return "esports"
    if any(p.search(full) for p in _SPORTS_STRONG_RE):
        return "sports"

    # 2. Fall through to general keyword scoring. Question hits count double:
    # the question is the market's identity, while descriptions are mostly
    # resolution boilerplate ("according to AI transcription services" filed
    # JD-Vance mention markets under tech).
    scores: dict[str, int] = {}
    for category, patterns in _CATEGORY_RES.items():
        score = sum(2 if p.search(q) else 1 for p in patterns if p.search(full))
        if score > 0:
            scores[category] = score

    # Politics: decide US vs. international by country context. Governance terms
    # (president/election/...) count toward whichever bucket the country markers
    # select, so a foreign election lands in politics_intl instead of defaulting
    # to politics_us. With no country marker, generic governance defaults to US
    # (the bulk of prediction-market election markets), preserving prior behavior.
    us = sum(2 if p.search(q) else 1 for p in _POLITICS_US_RE if p.search(full))
    intl = sum(2 if p.search(q) else 1 for p in _POLITICS_INTL_RE if p.search(full))
    gov = sum(2 if p.search(q) else 1 for p in _POLITICS_GOV_RE if p.search(full))
    if intl and not us:
        scores["politics_intl"] = intl + gov
    elif us:
        scores["politics_us"] = us + gov
    elif gov and any(p.search(q) for p in _POLITICS_GOV_RE):
        # Governance terms with no country marker default to politics_us, but
        # only when the QUESTION itself reads political — description
        # boilerplate ("primary listing", "votes are tallied") filed tennis
        # and SpaceX-IPO markets as politics_us.
        scores["politics_us"] = gov

    if not scores:
        return "other"

    return max(scores, key=scores.get)


def ensure_category(question: str, description: str = "",
                    category: str | None = "") -> str:
    """Return *category*, classifying when missing.

    Markets must never be stored category-less: empty categories bypass
    ``blocked_categories`` (the check is ``category in blocked``) and
    pollute graduation cells as '(none)'. Every INSERT-into-markets site
    routes through this instead of writing ``market.category`` raw.
    """
    return category or classify_market(question or "", description or "")
