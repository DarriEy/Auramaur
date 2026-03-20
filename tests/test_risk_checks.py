"""Tests for all 15 risk checks."""

import pytest
from pathlib import Path
from unittest.mock import patch

from auramaur.risk.checks import (
    CheckResult,
    check_kill_switch,
    check_max_drawdown,
    check_drawdown_heat,
    check_max_stake,
    check_daily_loss,
    check_max_positions,
    check_min_edge,
    check_min_liquidity,
    check_max_spread,
    check_confidence_floor,
    check_implied_prob_bounds,
    check_category_exposure,
    check_correlation,
    check_time_to_resolution,
    check_second_opinion_divergence,
)


# 1. Kill switch
@pytest.mark.asyncio
async def test_kill_switch_inactive():
    with patch.object(Path, "exists", return_value=False):
        result = await check_kill_switch()
        assert result.passed is True


@pytest.mark.asyncio
async def test_kill_switch_active():
    with patch.object(Path, "exists", return_value=True):
        result = await check_kill_switch()
        assert result.passed is False


# 2. Max drawdown
@pytest.mark.asyncio
async def test_max_drawdown_within_limit():
    result = await check_max_drawdown(10.0, 15.0)
    assert result.passed is True


@pytest.mark.asyncio
async def test_max_drawdown_exceeded():
    result = await check_max_drawdown(16.0, 15.0)
    assert result.passed is False


# 3. Drawdown heat
@pytest.mark.asyncio
async def test_drawdown_heat_green():
    result = await check_drawdown_heat(3.0, 15.0)
    assert result.passed is True
    assert result.value == "GREEN"


@pytest.mark.asyncio
async def test_drawdown_heat_yellow():
    result = await check_drawdown_heat(7.0, 15.0)
    assert result.passed is True
    assert result.value == "YELLOW"


@pytest.mark.asyncio
async def test_drawdown_heat_orange():
    result = await check_drawdown_heat(12.0, 15.0)
    assert result.passed is True
    assert result.value == "ORANGE"


@pytest.mark.asyncio
async def test_drawdown_heat_red():
    result = await check_drawdown_heat(14.0, 15.0)
    assert result.passed is False
    assert result.value == "RED"


# 4. Max stake
@pytest.mark.asyncio
async def test_max_stake_within_limit():
    result = await check_max_stake(20.0, 25.0)
    assert result.passed is True


@pytest.mark.asyncio
async def test_max_stake_exceeded():
    result = await check_max_stake(30.0, 25.0)
    assert result.passed is False


# 5. Daily loss limit
@pytest.mark.asyncio
async def test_daily_loss_within_limit():
    result = await check_daily_loss(100.0, 200.0)
    assert result.passed is True


@pytest.mark.asyncio
async def test_daily_loss_exceeded():
    result = await check_daily_loss(250.0, 200.0)
    assert result.passed is False


# 6. Max positions
@pytest.mark.asyncio
async def test_max_positions_within_limit():
    result = await check_max_positions(10, 15)
    assert result.passed is True


@pytest.mark.asyncio
async def test_max_positions_at_limit():
    result = await check_max_positions(15, 15)
    assert result.passed is False


# 7. Min edge
@pytest.mark.asyncio
async def test_min_edge_sufficient():
    result = await check_min_edge(7.0, 5.0)
    assert result.passed is True


@pytest.mark.asyncio
async def test_min_edge_insufficient():
    result = await check_min_edge(3.0, 5.0)
    assert result.passed is False


# 8. Min liquidity
@pytest.mark.asyncio
async def test_min_liquidity_sufficient():
    result = await check_min_liquidity(5000.0, 1000.0)
    assert result.passed is True


@pytest.mark.asyncio
async def test_min_liquidity_insufficient():
    result = await check_min_liquidity(500.0, 1000.0)
    assert result.passed is False


# 9. Max spread
@pytest.mark.asyncio
async def test_max_spread_within_limit():
    result = await check_max_spread(3.0, 5.0)
    assert result.passed is True


@pytest.mark.asyncio
async def test_max_spread_exceeded():
    result = await check_max_spread(7.0, 5.0)
    assert result.passed is False


# 10. Confidence floor
@pytest.mark.asyncio
async def test_confidence_floor_high():
    result = await check_confidence_floor("HIGH", "MEDIUM")
    assert result.passed is True


@pytest.mark.asyncio
async def test_confidence_floor_medium_meets_medium():
    result = await check_confidence_floor("MEDIUM", "MEDIUM")
    assert result.passed is True


@pytest.mark.asyncio
async def test_confidence_floor_low_below_medium():
    result = await check_confidence_floor("LOW", "MEDIUM")
    assert result.passed is False


# 11. Implied prob bounds
@pytest.mark.asyncio
async def test_implied_prob_within_bounds():
    result = await check_implied_prob_bounds(0.50, 0.05, 0.95)
    assert result.passed is True


@pytest.mark.asyncio
async def test_implied_prob_too_low():
    result = await check_implied_prob_bounds(0.03, 0.05, 0.95)
    assert result.passed is False


@pytest.mark.asyncio
async def test_implied_prob_too_high():
    result = await check_implied_prob_bounds(0.97, 0.05, 0.95)
    assert result.passed is False


# 12. Category exposure
@pytest.mark.asyncio
async def test_category_exposure_within_cap():
    result = await check_category_exposure("politics_us", 20.0, 30.0)
    assert result.passed is True


@pytest.mark.asyncio
async def test_category_exposure_exceeded():
    result = await check_category_exposure("politics_us", 35.0, 30.0)
    assert result.passed is False


# 13. Correlation
@pytest.mark.asyncio
async def test_correlation_few():
    result = await check_correlation("market1", ["m2", "m3"])
    assert result.passed is True


@pytest.mark.asyncio
async def test_correlation_too_many():
    result = await check_correlation("market1", ["m2", "m3", "m4", "m5", "m6", "m7"])
    assert result.passed is False


# 14. Time to resolution
@pytest.mark.asyncio
async def test_time_to_resolution_sufficient():
    result = await check_time_to_resolution(48, 24)
    assert result.passed is True


@pytest.mark.asyncio
async def test_time_to_resolution_too_soon():
    result = await check_time_to_resolution(12, 24)
    assert result.passed is False


# 15. Second opinion divergence
@pytest.mark.asyncio
async def test_divergence_within_limit():
    result = await check_second_opinion_divergence(0.10, 0.15)
    assert result.passed is True


@pytest.mark.asyncio
async def test_divergence_exceeded():
    result = await check_second_opinion_divergence(0.20, 0.15)
    assert result.passed is False


@pytest.mark.asyncio
async def test_divergence_none():
    result = await check_second_opinion_divergence(None, 0.15)
    assert result.passed is True  # No second opinion = skip check
