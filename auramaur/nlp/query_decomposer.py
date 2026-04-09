"""Decompose market questions into effective search queries.

Two modes:
  1. Per-market decomposition (original) — breaks a single question into
     2-3 targeted queries.
  2. Entity-aware batch decomposition — extracts shared entities across
     a batch of markets and produces entity-level queries that serve
     multiple markets simultaneously.  Reduces API calls while improving
     relational evidence coverage.
"""

from __future__ import annotations

import re

import structlog

log = structlog.get_logger()

# Stop words for entity extraction
_STOP = frozenset({
    "will", "does", "the", "a", "an", "of", "in", "on", "by", "to", "be",
    "is", "at", "for", "and", "or", "not", "this", "that", "with", "before",
    "after", "between", "more", "than", "its", "has", "have", "from", "are",
    "was", "were", "been", "being", "which", "their", "would", "could",
    "should", "may", "might", "can", "most", "over", "under", "yes", "no",
})

# Category → specialized query suffixes for better evidence routing
_CATEGORY_QUERY_HINTS: dict[str, list[str]] = {
    "politics_us": ["policy", "congress vote", "poll"],
    "politics_intl": ["government", "election", "minister"],
    "economics": ["economic data", "forecast", "GDP CPI rate"],
    "crypto": ["price", "blockchain", "token"],
    "tech": ["launch", "product", "announcement"],
    "science": ["study", "research", "FDA trial"],
    "legal": ["ruling", "court", "lawsuit"],
    "sports": ["trade", "injury", "standings", "roster"],
    "entertainment": ["release", "box office", "ratings"],
}


def _extract_named_entities(text: str) -> list[str]:
    """Extract likely named entities from text."""
    # Capitalized word sequences
    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    # ALL-CAPS acronyms (3+ chars)
    acronyms = re.findall(r"\b[A-Z]{3,}\b", text)

    result = []
    seen = set()
    for e in entities + acronyms:
        # Split multi-word matches and filter stop words from each part
        parts = [p for p in e.split() if p.lower() not in _STOP and len(p) > 2]
        for part in parts:
            lower = part.lower()
            if lower not in seen:
                seen.add(lower)
                result.append(part)
    return result


def extract_search_queries(
    question: str,
    description: str = "",
    category: str = "",
) -> list[str]:
    """Extract 2-3 targeted search queries from a market question.

    Uses heuristic decomposition — no API calls needed.
    Now category-aware: routes queries toward domain-specific terms.
    """
    queries: list[str] = []

    # 1. Clean the question — remove "Will", "Does", question marks, etc.
    text = question.strip().rstrip("?")
    text = re.sub(
        r"^(Will|Does|Is|Are|Has|Have|Can|Could|Should|Would)\s+",
        "", text, flags=re.IGNORECASE,
    )

    # 2. Add the cleaned question as first query
    queries.append(text)

    # 3. Extract named entities
    entities = _extract_named_entities(question)

    # 4. Extract key terms — numbers, percentages, dates
    numbers = re.findall(r"\d+(?:\.\d+)?%?", question)

    # 5. Build focused queries from entities + action keywords
    action_words = re.findall(
        r"\b(win|lose|pass|fail|approve|reject|resign|elect|announce|"
        r"cut|raise|drop|rise|ban|launch|release|sign|veto|defeat|"
        r"trade|fire|hire|acquire|merge|settle|sanction|deploy)\b",
        text.lower(),
    )

    if entities and action_words:
        queries.append(f"{entities[0]} {action_words[0]}")
    elif entities:
        entity_query = " ".join(entities[:2])
        if numbers:
            entity_query += " " + numbers[0]
        queries.append(entity_query)

    # 6. Category-aware query — add domain-specific suffix
    if entities and category:
        hints = _CATEGORY_QUERY_HINTS.get(category, [])
        if hints:
            queries.append(f"{entities[0]} {hints[0]}")
        else:
            queries.append(f"{entities[0]} latest news")
    elif entities:
        queries.append(f"{entities[0]} latest news")

    # Deduplicate and limit
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        q_lower = q.lower().strip()
        if q_lower and q_lower not in seen:
            seen.add(q_lower)
            unique.append(q)

    return unique[:3]


def extract_entity_queries(
    markets: list[dict],
) -> dict[str, list[str]]:
    """Extract entity-level queries across a batch of markets.

    Instead of generating queries per-market, this identifies SHARED
    entities across the batch and generates queries at the entity level.
    Evidence gathered for an entity serves ALL markets that reference it.

    Args:
        markets: List of dicts with 'id', 'question', 'description', 'category'.

    Returns:
        Dict mapping entity query string → list of market IDs that
        would benefit from this evidence.
    """
    # Build entity → market_ids mapping
    entity_markets: dict[str, list[str]] = {}
    market_categories: dict[str, str] = {}

    for m in markets:
        market_id = m["id"]
        question = m.get("question", "")
        category = m.get("category", "")
        market_categories[market_id] = category

        entities = _extract_named_entities(question)
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower not in entity_markets:
                entity_markets[entity_lower] = []
            entity_markets[entity_lower].append(market_id)

    # Generate queries, prioritizing shared entities (appear in 2+ markets)
    queries: dict[str, list[str]] = {}

    # First: shared entities (relational value)
    shared = {e: mids for e, mids in entity_markets.items() if len(mids) >= 2}
    for entity, market_ids in sorted(shared.items(), key=lambda x: -len(x[1])):
        query = f"{entity} latest news"
        queries[query] = market_ids

        # Add category-aware variant using the most common category
        categories = [market_categories.get(mid, "") for mid in market_ids]
        most_common = max(set(categories), key=categories.count) if categories else ""
        hints = _CATEGORY_QUERY_HINTS.get(most_common, [])
        if hints:
            queries[f"{entity} {hints[0]}"] = market_ids

    # Then: unique entities (still useful but lower priority)
    unique = {e: mids for e, mids in entity_markets.items() if len(mids) == 1}
    for entity, market_ids in unique.items():
        query = f"{entity} latest news"
        queries[query] = market_ids

    if shared:
        log.info(
            "query_decomposer.entity_queries",
            shared_entities=len(shared),
            unique_entities=len(unique),
            total_queries=len(queries),
        )

    return queries
