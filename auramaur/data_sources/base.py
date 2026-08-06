"""Base protocol and models for data sources."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator

# aiohttp's ClientResponseError stringifies to include the full request URL,
# query string and all. Sources that pass a credential as a query parameter
# therefore leak it the moment the venue returns 4xx/5xx — which for a metered
# API is the *normal* failure (403 over-quota, 401 after rotation). Truncating
# the message does not help: the key is early in the URL when it is the first
# parameter. Redact by name, then by shape for names we do not know.
_SECRET_PARAM_RE = re.compile(
    r"([?&](?:api[_-]?key|apikey|key|token|access[_-]?token|auth|"
    r"user[_-]?id|userid|secret|password|passwd|pwd|sig|signature)=)"
    r"[^&\s'\"]*",
    re.IGNORECASE,
)
# Backstop for parameters we have not enumerated: any long value is treated as
# a credential regardless of its name.
#
# This class went through two rounds, and the second one is the point. It began
# as [A-Za-z0-9_-]{20,}; that missed '%', because aiohttp percent-encodes the
# reserved characters of a standard-base64 secret ('+' -> %2B, '=' -> %3D) and
# the escape split the value into runs under the threshold. Adding '%' fixed
# that case and left '/' open — yarl does not encode '/' inside a query value —
# which was recorded as needing its own decision, since adding '/' to a
# character class would also redact ordinary path-shaped parameters.
#
# The decision: stop enumerating. Match everything up to a real delimiter, the
# same terminator set _SECRET_PARAM_RE already uses. Enumerating the characters
# a secret may contain is the same losing move as enumerating the ways
# untrusted text can be unsafe — both are written from the author's
# imagination, and the gap is precisely what was not imagined. The cost is
# over-redaction of long benign values; status, host, path and short
# diagnostic parameters still log verbatim, which is what an error line is for.
_OPAQUE_VALUE_RE = re.compile(r"([?&][A-Za-z0-9_.\-]+=)([^&\s'\"]{20,})")


def redact_error(exc: BaseException | str, limit: int = 120) -> str:
    """Stringify an exception for logging with credential-bearing params removed.

    Redaction happens *before* truncation so a slice can never expose the tail
    of a key that the redactor would otherwise have covered.
    """
    text = _SECRET_PARAM_RE.sub(r"\1<redacted>", str(exc))
    text = _OPAQUE_VALUE_RE.sub(r"\1<redacted>", text)
    return text[:limit]


class NewsItem(BaseModel):
    id: str
    source: str
    title: str
    content: str = ""
    url: str = ""
    published_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # exact: publisher supplied; inferred/provider_seen/unknown: progressively
    # weaker time semantics. Unknown dates must not masquerade as breaking news.
    timestamp_quality: str = "exact"
    relevance_score: float = 0.0
    keywords: list[str] = Field(default_factory=list)
    ingestion_run_id: str = ""
    information_mode: str = "production"  # production | shadow | paired

    @field_validator("published_at")
    @classmethod
    def _published_at_is_utc_aware(cls, value: datetime) -> datetime:
        """Normalize provider timestamps once at the ingestion boundary."""
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


@runtime_checkable
class DataSource(Protocol):
    """Protocol that all data sources must implement.

    ``categories`` gates when the source fires:
      * ``None`` (or attr missing): fires for every query — use for broad
        sources like general news, web search, or top-of-funnel RSS.
      * A ``set[str]`` of market categories: fires only when the market's
        category is in the set. Domain sources (e.g. USGS for seismic)
        declare their scope this way so they don't add latency to
        unrelated market analyses.
    """

    source_name: str
    categories: set[str] | None

    async def fetch(self, query: str, limit: int = 20) -> list[NewsItem]:
        """Fetch news items matching the query."""
        ...

    async def close(self) -> None:
        """Clean up resources."""
        ...
