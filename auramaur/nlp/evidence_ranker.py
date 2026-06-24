"""Global, market-aware evidence ranking.

The per-market evidence gather concatenates several queries' results, so a naive
``items[:N]`` truncates by query order, not by signal — the best item from the
second query loses to mediocre items from the first. This module re-ranks the
*merged* candidate set against the market question, combining three axes:

    score = recency_decay(age) * source_authority(source) * (1 + relevance)

so we feed the model the best N items, not the first N.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from auramaur.data_sources.base import NewsItem
from auramaur.nlp.relevance import relevance_scores
from auramaur.nlp.sources import authority

log = structlog.get_logger()

# Recency half-life in hours: a story this old counts for half a fresh one.
_RECENCY_HALFLIFE_HOURS = 36.0


def _recency_decay(item: NewsItem, now: datetime) -> float:
    pub = item.published_at
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)
    age_hours = max((now - pub).total_seconds() / 3600.0, 0.0)
    return 0.5 ** (age_hours / _RECENCY_HALFLIFE_HOURS)


def rank_evidence(
    question: str,
    items: list[NewsItem],
    *,
    top_n: int,
    backend: str = "embeddings",
    model_name: str = "all-MiniLM-L6-v2",
    now: datetime | None = None,
) -> list[NewsItem]:
    """Return the top-``top_n`` items by combined recency/authority/relevance.

    Relevance is scored against the market question over the merged candidate
    set (title + content snippet). Pure function aside from logging.
    """
    if not items:
        return []
    if len(items) <= top_n and top_n > 0:
        # Still reorder by score so the most relevant lead the block.
        pass

    now = now or datetime.now(tz=timezone.utc)

    texts = [f"{it.title or ''}. {(it.content or '')[:400]}" for it in items]
    rel = relevance_scores(question, texts, backend=backend, model_name=model_name)

    scored: list[tuple[float, NewsItem]] = []
    for item, r in zip(items, rel):
        score = _recency_decay(item, now) * authority(item.source, item.url) * (1.0 + r)
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    ranked = [it for _, it in scored]
    selected = ranked[:top_n] if top_n > 0 else ranked

    log.debug(
        "evidence.ranked",
        candidates=len(items),
        selected=len(selected),
        top_score=round(scored[0][0], 4) if scored else 0.0,
    )
    return selected
