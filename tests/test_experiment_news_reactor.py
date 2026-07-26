"""Parity and safety tests for pure news-reactor market proposals."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

from auramaur.experiments.strategies.news_reactor import (
    NewsMarketCandidate,
    NewsReactorRules,
    extract_proper_nouns,
    extract_search_terms,
    form_news_market_proposal,
)
from auramaur.strategy.news_reactor import NewsReactor


def _candidate(**changes: object) -> NewsMarketCandidate:
    values: dict[str, object] = {
        "market_id": "m1",
        "question": "Will the Federal Reserve cut rates in September?",
        "description": "Decision by the Federal Reserve",
        "category": "economics",
        "active": True,
        "liquidity": 2500.0,
        "volume": 500.0,
    }
    values.update(changes)
    return NewsMarketCandidate(**values)  # type: ignore[arg-type]


def test_pure_proposal_matches_production_headline_decisions() -> None:
    headline = "Federal Reserve signals September interest rate cut"
    reactor = NewsReactor(MagicMock(), {}, {}, MagicMock())
    proposal = form_news_market_proposal(headline, _candidate(), NewsReactorRules())

    assert proposal is not None
    assert proposal.market_id == "m1"
    assert proposal.matched_proper_nouns == ("Federal", "Reserve", "September")
    assert reactor._extract_proper_nouns(headline) == extract_proper_nouns(headline)
    assert reactor._extract_search_terms(headline) == extract_search_terms(headline)


def test_volume_fallback_preserves_active_kalshi_style_candidate() -> None:
    proposal = form_news_market_proposal(
        "Federal Reserve announces decision",
        _candidate(liquidity=0.0, volume=9000.0),
        NewsReactorRules(),
    )
    assert proposal is not None
    assert proposal.activity == 9000.0


def test_proposal_fails_closed_for_ineligible_or_invalid_candidates() -> None:
    headline = "Federal Reserve announces decision"
    rejected = (
        _candidate(active=False),
        _candidate(category="Sports"),
        _candidate(liquidity=100.0, volume=100.0),
        _candidate(liquidity=float("nan")),
        _candidate(question="Unrelated market", description="No relevant entity"),
    )
    assert all(
        form_news_market_proposal(headline, candidate, NewsReactorRules()) is None
        for candidate in rejected
    )
    assert form_news_market_proposal(
        "Cheap stuff that does not suck", _candidate(), NewsReactorRules()
    ) is None


def test_experiment_strategy_has_no_live_runtime_imports() -> None:
    path = Path("auramaur/experiments/strategies/news_reactor.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    forbidden = ("auramaur.exchange", "auramaur.strategy", "auramaur.risk", "auramaur.db")
    assert not any(name.startswith(forbidden) for name in imported)
