"""Tests for news-reactor integration contracts."""

from inspect import signature

from auramaur.strategy.engine import TradingEngine


def test_trading_engine_accepts_news_strategy_source():
    params = signature(TradingEngine.analyze_market).parameters
    assert "strategy_source" in params
