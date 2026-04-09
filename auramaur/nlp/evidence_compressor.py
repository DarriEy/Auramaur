"""Evidence compressor — extracts signal from noisy evidence.

Instead of feeding Claude 15 raw articles (most redundant or irrelevant),
compress the evidence into structured dimensions that matter for prediction:

1. FACTUAL STATE — what is objectively true right now?
2. DIRECTIONAL SIGNALS — what points toward YES vs NO?
3. TEMPORAL DYNAMICS — what's changing and how fast?
4. INFORMATION GAPS — what's conspicuously absent?
5. SOURCE CONSENSUS — do sources agree or conflict?

This is conceptually similar to SVD: reduce high-dimensional noisy evidence
into the principal components that carry the most predictive signal.
"""

from __future__ import annotations

import asyncio
import json
import re

import structlog

from auramaur.data_sources.base import NewsItem

log = structlog.get_logger()


def compress_evidence(
    question: str,
    description: str,
    evidence: list[NewsItem],
    max_chars: int = 2000,
) -> str:
    """Compress raw evidence into structured signal dimensions.

    This is a fast, local compression — no LLM call needed.
    Extracts the key information axes from raw evidence.
    """
    if not evidence:
        return "(No evidence available)"

    # Deduplicate by content similarity
    unique = _deduplicate(evidence)

    # Extract structured dimensions
    parts: list[str] = []

    # 1. FACTUAL STATE — extract concrete facts (dates, numbers, names)
    facts = _extract_facts(unique)
    if facts:
        parts.append("FACTS: " + " | ".join(facts[:8]))

    # 2. DIRECTIONAL SIGNALS — what points YES vs NO
    yes_signals, no_signals = _extract_directional(question, unique)
    if yes_signals:
        parts.append("FOR YES: " + "; ".join(yes_signals[:4]))
    if no_signals:
        parts.append("FOR NO: " + "; ".join(no_signals[:4]))

    # 3. TEMPORAL — recency and momentum
    temporal = _extract_temporal(unique)
    if temporal:
        parts.append("TIMELINE: " + temporal)

    # 4. SOURCE CONSENSUS
    consensus = _extract_consensus(unique)
    if consensus:
        parts.append("CONSENSUS: " + consensus)

    # 5. RAW EXCERPTS — top 3 most relevant snippets (for Claude to reason over)
    excerpts = _extract_top_excerpts(question, unique, max_excerpts=3)
    if excerpts:
        parts.append("KEY EXCERPTS:\n" + "\n".join(f"- [{e[0]}] {e[1]}" for e in excerpts))

    compressed = "\n".join(parts)

    # Ensure we stay under budget
    if len(compressed) > max_chars:
        compressed = compressed[:max_chars - 20] + "\n...(truncated)"

    log.debug(
        "evidence.compressed",
        raw_items=len(evidence),
        unique_items=len(unique),
        compressed_chars=len(compressed),
    )

    return compressed


def _deduplicate(evidence: list[NewsItem]) -> list[NewsItem]:
    """Remove near-duplicate evidence items by title similarity."""
    seen_words: list[set[str]] = []
    unique: list[NewsItem] = []

    for item in evidence:
        words = set(item.title.lower().split())
        # Check overlap with existing items
        is_dup = False
        for seen in seen_words:
            if len(words) > 0 and len(words & seen) / max(len(words), 1) > 0.6:
                is_dup = True
                break
        if not is_dup:
            unique.append(item)
            seen_words.append(words)

    return unique


def _extract_facts(evidence: list[NewsItem]) -> list[str]:
    """Extract concrete factual statements from evidence."""
    facts: list[str] = []

    # Patterns that indicate factual content
    fact_patterns = [
        r'\$[\d,.]+[BMK]?\b',           # Dollar amounts
        r'\d+\.?\d*%',                    # Percentages
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', # Dates
        r'\b(?:signed|announced|confirmed|approved|rejected|passed|failed)\b',
        r'\b(?:according to|reported|stated|said)\b',
    ]

    for item in evidence[:10]:
        content = f"{item.title} {item.content or ''}"
        for pattern in fact_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                # Extract the sentence containing the fact
                sentences = re.split(r'[.!?]', content)
                for sent in sentences:
                    if re.search(pattern, sent, re.IGNORECASE) and len(sent.strip()) > 15:
                        fact = sent.strip()[:150]
                        if fact not in facts:
                            facts.append(fact)
                            break
                break

    return facts


def _extract_directional(
    question: str, evidence: list[NewsItem],
) -> tuple[list[str], list[str]]:
    """Classify evidence as supporting YES or NO."""
    yes_signals: list[str] = []
    no_signals: list[str] = []

    # Simple keyword-based directional classification
    positive_words = {
        "will", "likely", "expected", "confirmed", "approved", "agreed",
        "advancing", "progress", "success", "growth", "increase", "rising",
        "support", "momentum", "boost",
    }
    negative_words = {
        "unlikely", "rejected", "failed", "denied", "blocked", "delayed",
        "declined", "falling", "collapse", "opposition", "against",
        "stalled", "uncertain", "doubt",
    }

    q_words = set(question.lower().split())

    for item in evidence[:10]:
        text = f"{item.title} {item.content or ''}".lower()
        text_words = set(text.split())

        # Check relevance to question
        relevance = len(q_words & text_words) / max(len(q_words), 1)
        if relevance < 0.15:
            continue

        pos_count = len(positive_words & text_words)
        neg_count = len(negative_words & text_words)

        snippet = item.title[:100]
        if pos_count > neg_count:
            yes_signals.append(f"({item.source}) {snippet}")
        elif neg_count > pos_count:
            no_signals.append(f"({item.source}) {snippet}")

    return yes_signals, no_signals


def _extract_temporal(evidence: list[NewsItem]) -> str:
    """Extract timing and recency information."""
    if not evidence:
        return ""

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    ages = []
    for item in evidence:
        pub = item.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        hours = (now - pub).total_seconds() / 3600
        ages.append(hours)

    if not ages:
        return ""

    newest = min(ages)
    oldest = max(ages)

    if newest < 1:
        freshness = "Breaking — evidence from last hour"
    elif newest < 6:
        freshness = f"Recent — newest evidence {newest:.0f}h ago"
    elif newest < 24:
        freshness = f"Today's news — newest {newest:.0f}h ago"
    else:
        freshness = f"Aging — newest evidence {newest:.0f}h ago"

    return freshness


def _extract_consensus(evidence: list[NewsItem]) -> str:
    """Assess whether sources agree or conflict."""
    sources = set()
    for item in evidence:
        sources.add(item.source)

    if len(sources) <= 1:
        return f"Single source ({next(iter(sources), 'unknown')})"
    elif len(sources) >= 4:
        return f"Broad coverage ({len(sources)} sources: {', '.join(sorted(sources)[:5])})"
    else:
        return f"{len(sources)} sources: {', '.join(sorted(sources))}"


def _extract_top_excerpts(
    question: str,
    evidence: list[NewsItem],
    max_excerpts: int = 3,
) -> list[tuple[str, str]]:
    """Extract the most relevant text excerpts."""
    q_words = set(question.lower().split()) - {"will", "the", "a", "an", "of", "in", "on", "by", "to", "be", "is"}

    scored: list[tuple[float, str, str]] = []
    for item in evidence:
        content = item.content or item.title
        if not content:
            continue

        # Score by word overlap with question
        content_words = set(content.lower().split())
        overlap = len(q_words & content_words)
        # Bonus for source reliability
        source_bonus = {"reuters": 2, "ap": 2, "bbc": 1.5, "web": 1, "newsapi": 1}.get(item.source.lower(), 0.5)

        score = overlap * source_bonus
        snippet = content[:200].strip()
        if len(content) > 200:
            snippet += "..."

        scored.append((score, item.source, snippet))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [(source, text) for _, source, text in scored[:max_excerpts]]
