"""Tests for the materiality triage and its news-reactor hook (fail-open)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from auramaur.data_sources.base import NewsItem
from auramaur.nlp.news_triage import MaterialityTriage
from auramaur.strategy.news_reactor import NewsReactor
from config.settings import LocalLLMConfig, LocalTriageConfig


def _triage(reply, *, available: bool = True, max_calls: int = 20) -> MaterialityTriage:
    client = MagicMock()
    client.available = available
    client.generate_json = AsyncMock(return_value=reply)
    settings = SimpleNamespace(local_llm=LocalLLMConfig(
        enabled=True,
        triage=LocalTriageConfig(enabled=True, max_calls_per_cycle=max_calls)))
    return MaterialityTriage(client, settings)


async def test_screen_scores_and_caches():
    triage = _triage({"material": True, "score": 0.8})
    assert await triage.screen("i1", "headline", "question?") == 0.8
    assert await triage.screen("i1", "headline", "question?") == 0.8
    assert triage._client.generate_json.await_count == 1  # pair cache hit


async def test_screen_not_material_forces_zero():
    triage = _triage({"material": False, "score": 0.9})
    assert await triage.screen("i1", "headline", "question?") == 0.0


async def test_screen_fail_open_paths():
    assert await _triage(None).screen("i1", "h", "q") is None          # transport
    assert await _triage({"material": True}, available=False).screen(
        "i1", "h", "q") is None                                        # breaker open
    capped = _triage({"material": True, "score": 0.9}, max_calls=0)
    assert await capped.screen("i1", "h", "q") is None                 # over cap
    assert await _triage({"material": True, "score": "junk"}).screen(
        "i1", "h", "q") is None                                        # bad score


def _reactor(triage) -> tuple[NewsReactor, MagicMock, MagicMock]:
    rss = MagicMock()
    rss.fetch = AsyncMock(return_value=[
        NewsItem(id="story-1", source="rss", title="Fed surprise decision")])
    engine = MagicMock()
    market = SimpleNamespace(
        id="m1", question="Will the Fed cut rates?", category="economics",
        liquidity=5000.0)
    reactor = NewsReactor(
        rss_source=rss,
        discoveries={"polymarket": MagicMock()},
        engines={"polymarket": engine},
        db=MagicMock(),
        triage=triage,
    )
    reactor._find_related_markets = AsyncMock(
        return_value={"polymarket": [market]})
    return reactor, engine, market


async def test_reactor_skips_below_threshold():
    reactor, engine, _ = _reactor(_triage({"material": True, "score": 0.1}))
    flagged = await reactor.check_for_news()
    engine.flag_market_from_news.assert_not_called()
    assert flagged == []


async def test_reactor_flags_above_threshold():
    reactor, engine, market = _reactor(_triage({"material": True, "score": 0.9}))
    flagged = await reactor.check_for_news()
    engine.flag_market_from_news.assert_called_once_with(market.id)
    assert len(flagged) == 1


async def test_reactor_flags_when_screen_unavailable():
    reactor, engine, market = _reactor(_triage(None))  # local model down
    await reactor.check_for_news()
    engine.flag_market_from_news.assert_called_once_with(market.id)


async def test_reactor_flags_when_screen_raises():
    triage = _triage({"material": True, "score": 0.9})
    triage.screen = AsyncMock(side_effect=RuntimeError("boom"))
    reactor, engine, market = _reactor(triage)
    await reactor.check_for_news()
    engine.flag_market_from_news.assert_called_once_with(market.id)


async def test_reactor_without_triage_unchanged():
    reactor, engine, market = _reactor(None)
    await reactor.check_for_news()
    engine.flag_market_from_news.assert_called_once_with(market.id)
