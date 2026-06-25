"""Tests for news-reactor integration contracts."""

from inspect import signature
from unittest.mock import AsyncMock, MagicMock

import pytest

from auramaur.data_sources.base import NewsItem
from auramaur.exchange.models import Market
from auramaur.strategy.engine import TradingEngine
from auramaur.strategy.news_reactor import NewsReactor


def test_trading_engine_accepts_news_strategy_source():
    params = signature(TradingEngine.analyze_market).parameters
    assert "strategy_source" in params


def _market(mid, category="", question="Q?"):
    return Market(id=mid, question=question, category=category,
                  clob_token_yes="y", clob_token_no="n")


def test_is_blocked_category_blocks_live_event_outcomes():
    # esports / sports (any case) are blocked; news-tractable categories are not.
    assert NewsReactor._is_blocked_category(_market("a", "esports")) is True
    assert NewsReactor._is_blocked_category(_market("b", "Sports")) is True
    assert NewsReactor._is_blocked_category(_market("c", "politics_intl")) is False
    assert NewsReactor._is_blocked_category(_market("d", "crypto")) is False
    # A political head-to-head must NOT be blocked just for reading "vs".
    assert NewsReactor._is_blocked_category(
        _market("e", "politics_us", "Trump vs Biden 2028?")) is False


@pytest.mark.asyncio
async def test_check_for_news_skips_blocked_category_markets():
    """A flagged esports market is dropped; the tractable match still flags."""
    reactor = NewsReactor(
        rss_source=MagicMock(),
        discoveries={"polymarket": MagicMock()},
        engines={"polymarket": MagicMock()},
        db=MagicMock(),
    )
    # One new story.
    reactor._rss.fetch = AsyncMock(return_value=[
        NewsItem(id="n1", source="rss", title="Breaking developments today"),
    ])
    esports = _market("esp", "esports", "Team A vs Team B: map 1 winner")
    geo = _market("geo", "politics_intl", "Will the ceasefire hold by July?")
    reactor._find_related_markets = AsyncMock(
        return_value={"polymarket": [esports, geo]})
    reactor._engines["polymarket"].flag_market_from_news = MagicMock()

    flagged = await reactor.check_for_news()

    flagged_ids = {f["market_id"] for f in flagged}
    assert "esp" not in flagged_ids        # esports dropped
    assert "geo" in flagged_ids            # tractable match kept
    reactor._engines["polymarket"].flag_market_from_news.assert_called_once_with("geo")
