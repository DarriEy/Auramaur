"""Decompose market questions into effective search queries."""

from __future__ import annotations

import asyncio
import json
import re

import structlog

log = structlog.get_logger()


# Simple entity/keyword extraction without Claude (fast, no API cost)
def extract_search_queries(question: str, description: str = "") -> list[str]:
    """Extract 2-3 targeted search queries from a market question.

    Uses heuristic decomposition — no API calls needed.
    Much better than searching for the full question verbatim.
    """
    queries: list[str] = []

    # 1. Clean the question — remove "Will", "Does", question marks, etc.
    text = question.strip().rstrip("?")
    text = re.sub(r"^(Will|Does|Is|Are|Has|Have|Can|Could|Should|Would)\s+", "", text, flags=re.IGNORECASE)

    # 2. Add the cleaned question as first query
    queries.append(text)

    # 3. Extract named entities (capitalized multi-word sequences)
    entities = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", question)
    entities = [e for e in entities if len(e) > 2 and e not in ("Will", "Does", "The", "Yes", "No")]

    # 4. Extract key terms — numbers, percentages, dates
    numbers = re.findall(r"\d+(?:\.\d+)?%?", question)

    # 5. Build focused queries from entities + action keywords
    action_words = re.findall(r"\b(win|lose|pass|fail|approve|reject|resign|elect|announce|cut|raise|drop|rise|ban|launch|release|sign|veto|defeat)\b", text.lower())

    if entities and action_words:
        # Entity + action query: "Trump win Georgia"
        queries.append(f"{entities[0]} {action_words[0]}")
    elif entities:
        # Just entity: "Federal Reserve March 2025"
        entity_query = " ".join(entities[:2])
        if numbers:
            entity_query += " " + numbers[0]
        queries.append(entity_query)

    # 6. Add a recency-biased query: entity + "latest" or "today"
    if entities:
        queries.append(f"{entities[0]} latest news")

    # Deduplicate and limit
    seen = set()
    unique = []
    for q in queries:
        q_lower = q.lower().strip()
        if q_lower and q_lower not in seen:
            seen.add(q_lower)
            unique.append(q)

    return unique[:3]
