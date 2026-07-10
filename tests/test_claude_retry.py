"""Tests for Claude CLI retry logic and graceful degradation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auramaur.nlp.analyzer import AnalysisResult, ClaudeAnalyzer


@pytest.fixture
def settings():
    s = MagicMock()
    s.nlp.model = "claude-sonnet-4-20250514"
    s.nlp.daily_claude_call_budget = 100
    s.nlp.claude_reserve_for_pinned = 0
    s.nlp.skip_second_opinion = False
    s.nlp.cache_ttl_breaking_seconds = 900
    # Pacing disabled: these tests exercise retries, not the envelope.
    s.nlp.budget_peak_start_hour_utc = 12
    s.nlp.budget_peak_end_hour_utc = 22
    s.nlp.budget_offpeak_share = 1.0
    return s


@pytest.fixture
def analyzer(settings):
    return ClaudeAnalyzer(settings=settings)


@pytest.mark.asyncio
async def test_retry_on_timeout(analyzer):
    """Subprocess times out twice, then succeeds on third attempt."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(side_effect=[
        asyncio.TimeoutError("timed out"),
        asyncio.TimeoutError("timed out"),
        (b'{"probability": 0.7, "confidence": "HIGH", "reasoning": "ok"}', b""),
    ])
    mock_proc.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        result = await analyzer._call_claude_cli("test prompt")
        assert "probability" in result
        assert mock_proc.communicate.await_count == 3


@pytest.mark.asyncio
async def test_retry_exhausted_raises(analyzer):
    """All 3 attempts fail — should raise."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError("timed out"))

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await analyzer._call_claude_cli("test prompt")
        assert mock_proc.communicate.await_count == 3


@pytest.mark.asyncio
async def test_no_retry_on_budget(settings):
    """Budget exhausted raises BudgetExhausted immediately, no retries.

    BudgetExhausted subclasses RuntimeError so existing `except RuntimeError`
    handlers keep working, while callers can catch it specifically to treat the
    throttle as expected (logged at debug, not error)."""
    from unittest.mock import patch

    from auramaur.nlp.errors import BudgetExhausted

    assert issubclass(BudgetExhausted, RuntimeError)  # backward-compat contract

    settings.nlp.daily_claude_call_budget = 1
    a = ClaudeAnalyzer(settings=settings)
    # Simulate one call already made today (persistent counter)
    with patch("auramaur.nlp.call_budget.calls_today", return_value=1):
        with pytest.raises(BudgetExhausted, match="budget"):
            await a._call_claude_cli("test prompt")


@pytest.mark.asyncio
async def test_analyze_graceful_degradation(analyzer):
    """CLI fails for primary → returns AnalysisResult with skipped_reason."""
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)

    market = MagicMock()
    market.id = "test-market"
    market.question = "Will X happen?"
    market.description = "Test"
    market.outcome_yes_price = 0.5

    evidence = []

    with patch.object(analyzer, "estimate_probability", side_effect=RuntimeError("CLI down")), \
         patch("auramaur.monitoring.display.show_claude_thinking"), \
         patch("auramaur.nlp.analyzer._evidence_digest", return_value="abc"):
        result = await analyzer.analyze(market, evidence, mock_cache)
        assert result.probability == 0.5
        assert result.confidence == "LOW"
        assert result.skipped_reason is not None
        assert "failed" in result.skipped_reason.lower()


@pytest.mark.asyncio
async def test_second_opinion_failure_proceeds(analyzer):
    """Second opinion fails → returns primary result without second opinion."""
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.put = AsyncMock()

    market = MagicMock()
    market.id = "test-market"
    market.question = "Will X happen?"
    market.description = "Test"
    market.outcome_yes_price = 0.5

    primary_result = AnalysisResult(
        probability=0.7,
        confidence="HIGH",
        reasoning="Good reasons",
    )

    with patch.object(analyzer, "estimate_probability", return_value=primary_result), \
         patch.object(analyzer, "get_second_opinion", side_effect=RuntimeError("CLI down")), \
         patch("auramaur.monitoring.display.show_claude_thinking"), \
         patch("auramaur.nlp.analyzer._evidence_digest", return_value="abc"):
        result = await analyzer.analyze(market, [], mock_cache)
        assert result.probability == 0.7
        assert result.second_opinion_prob is None
        assert result.divergence is None
