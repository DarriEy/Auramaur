"""Centralized source-authority weights.

Single source of truth for how much we trust each evidence source. Previously
this map was duplicated (with conflicting values) across the aggregator's
ranker and the evidence compressor; both now import from here so ranking and
compression agree on what "authoritative" means.

Higher = more trustworthy. Default for unknown sources is 1.0.
"""

from __future__ import annotations

from urllib.parse import urlparse

SOURCE_AUTHORITY: dict[str, float] = {
    # Wire services / papers of record
    "reuters": 3.0, "ap": 3.0, "bloomberg": 2.5, "ft": 2.5, "wsj": 2.5,
    "bbc": 2.5, "nyt": 2.5,
    # Quality national/sector outlets
    "guardian": 2.0, "npr": 2.0, "politico": 2.0, "cnbc": 2.0,
    # Structured / primary data
    "fred": 2.0, "usgs": 2.5, "market_data": 2.5, "coingecko": 2.0,
    "manifold": 2.0, "polymarket_context": 2.0,
    # General web / aggregators
    "web": 1.5, "newsapi": 1.5, "espn": 1.5,
    # Lower-signal / social / crypto press
    "rss": 1.0, "coindesk": 1.2, "cointelegraph": 1.0,
    "hackernews": 1.0, "gdelt": 1.0,
    "reddit": 0.8, "google_trends": 0.8, "bluesky": 0.8,
}

_DEFAULT_AUTHORITY = 1.0

# Production URL authority is exact-host based. Candidate/shadow source
# authority belongs to the graduation plane and must not mutate this map.
_AUTHORITY_DOMAINS = {
    "reuters.com": "reuters", "apnews.com": "ap", "bloomberg.com": "bloomberg",
    "ft.com": "ft", "wsj.com": "wsj", "bbc.com": "bbc", "bbc.co.uk": "bbc",
    "nytimes.com": "nyt", "theguardian.com": "guardian", "npr.org": "npr",
    "politico.com": "politico", "cnbc.com": "cnbc", "fred.stlouisfed.org": "fred",
    "usgs.gov": "usgs", "coingecko.com": "coingecko", "coindesk.com": "coindesk",
    "cointelegraph.com": "cointelegraph",
}


def authority(source: str | None, url: str | None = "") -> float:
    """Return the trust weight for a source, checking the URL for known domains.

    Args:
        source: The source name (e.g. "reuters").
        url: Optional article URL; if it contains a known domain we take the
            higher of the source-name and domain weights.

    Returns:
        A positive float weight (default 1.0 for unknown sources).
    """
    weight = SOURCE_AUTHORITY.get((source or "").lower(), _DEFAULT_AUTHORITY)
    host = (urlparse(url or "").hostname or "").lower().rstrip(".")
    for domain, key in _AUTHORITY_DOMAINS.items():
        if host == domain or host.endswith(f".{domain}"):
            weight = max(weight, SOURCE_AUTHORITY[key])
            break
    return weight
