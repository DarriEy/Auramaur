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

import re

import structlog

from auramaur.data_sources.base import NewsItem
from auramaur.nlp.relevance import relevance_scores
from auramaur.nlp.sources import authority

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
    yes_signals, no_signals, used_titles = _extract_directional(question, unique)
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

    # 5. RAW EXCERPTS — top relevant snippets, skipping items already surfaced
    #    as directional signals so we don't pay tokens for the same headline twice.
    excerpts = _extract_top_excerpts(question, unique, max_excerpts=3, exclude_titles=used_titles)
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
    """Remove near-duplicate evidence items by title AND lead-content similarity.

    Syndicated copies often reword the headline but share the body, so title
    similarity alone misses them. We compare both and treat an item as a
    duplicate if either the title or the content lead substantially overlaps a
    kept item (Jaccard over word sets).
    """
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    seen_titles: list[set[str]] = []
    seen_bodies: list[set[str]] = []
    unique: list[NewsItem] = []

    for item in evidence:
        t_words = set((item.title or "").lower().split())
        b_words = set((item.content or "")[:300].lower().split())
        is_dup = False
        for st, sb in zip(seen_titles, seen_bodies):
            if _jaccard(t_words, st) > 0.6 or (len(b_words) >= 4 and _jaccard(b_words, sb) > 0.7):
                is_dup = True
                break
        if not is_dup:
            unique.append(item)
            seen_titles.append(t_words)
            seen_bodies.append(b_words)

    return unique


def _extract_facts(evidence: list[NewsItem]) -> list[str]:
    """Extract concrete factual statements from evidence."""
    facts: list[str] = []

    # Patterns that indicate factual content. Attribution boilerplate
    # ("according to", "said") was dropped — it matched non-facts and wasted
    # tokens. Concrete figures, dates, and decisive actions carry the signal.
    fact_patterns = [
        r'\$[\d,.]+[BMK]?\b',           # Dollar amounts
        r'\d+\.?\d*%',                    # Percentages
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', # Dates
        r'\b(?:signed|announced|confirmed|approved|rejected|passed|failed|'
        r'vetoed|resigned|launched|banned|ruled|sentenced|acquired|merged)\b',
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
) -> tuple[list[str], list[str], set[str]]:
    """Classify evidence as supporting YES or NO.

    Returns (yes_signals, no_signals, used_titles). ``used_titles`` lets the
    caller avoid repeating these headlines in the excerpt section. A simple
    negation guard flips a polarity word when it's immediately preceded by a
    negator ("not approved" -> NO, not YES).
    """
    yes_signals: list[str] = []
    no_signals: list[str] = []
    used_titles: set[str] = set()

    positive_words = {
        "likely", "expected", "confirmed", "approved", "agreed",
        "advancing", "progress", "success", "growth", "increase", "rising",
        "support", "momentum", "boost", "wins", "won", "passed",
    }
    negative_words = {
        "unlikely", "rejected", "failed", "denied", "blocked", "delayed",
        "declined", "falling", "collapse", "opposition", "against",
        "stalled", "uncertain", "doubt", "loses", "lost", "vetoed",
    }
    negators = {"not", "no", "never", "without", "fails", "fail", "denies"}

    # Rank candidates by relevance so we describe the most on-topic items first.
    titles = [f"{it.title or ''}. {(it.content or '')[:200]}" for it in evidence[:10]]
    rel = relevance_scores(question, titles, backend="heuristic")

    for item, r in sorted(zip(evidence[:10], rel), key=lambda x: x[1], reverse=True):
        if r < 0.12:
            continue
        tokens = re.findall(r"[a-z]+", f"{item.title} {item.content or ''}".lower())
        pos = neg = 0
        for idx, tok in enumerate(tokens):
            # A negator anywhere in the preceding 3-token window flips polarity
            # ("will not be approved" -> negative).
            window = tokens[max(0, idx - 3):idx]
            negated = any(w in negators for w in window)
            if tok in positive_words:
                neg += 1 if negated else 0
                pos += 0 if negated else 1
            elif tok in negative_words:
                pos += 1 if negated else 0
                neg += 0 if negated else 1

        snippet = (item.title or "")[:100]
        if pos > neg:
            yes_signals.append(f"({item.source}) {snippet}")
            used_titles.add(item.title or "")
        elif neg > pos:
            no_signals.append(f"({item.source}) {snippet}")
            used_titles.add(item.title or "")

    return yes_signals, no_signals, used_titles


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
    exclude_titles: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Extract the most relevant text excerpts, skipping already-surfaced items.

    Relevance uses the rarity-weighted scorer (so common words don't dominate)
    and is multiplied by centralized source authority. Items whose headline was
    already emitted as a directional signal are skipped to avoid duplication.
    """
    exclude_titles = exclude_titles or set()
    candidates = [
        it for it in evidence
        if (it.content or it.title) and (it.title or "") not in exclude_titles
    ]
    if not candidates:
        return []

    texts = [it.content or it.title or "" for it in candidates]
    rel = relevance_scores(question, texts, backend="heuristic")

    scored: list[tuple[float, str, str]] = []
    for item, r in zip(candidates, rel):
        content = item.content or item.title or ""
        score = (0.1 + r) * authority(item.source, item.url)
        snippet = content[:200].strip()
        if len(content) > 200:
            snippet += "..."
        scored.append((score, item.source, snippet))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [(source, text) for _, source, text in scored[:max_excerpts]]
