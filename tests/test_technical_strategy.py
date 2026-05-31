"""Tests for the technical analysis strategy."""

import pytest
from auramaur.exchange.models import Market, OrderSide
from auramaur.strategy.technical import TechnicalAnalyzer


@pytest.fixture
def analyzer():
    from config.settings import Settings
    return TechnicalAnalyzer(Settings())


def _make_market(price: float):
    return Market(
        id="test_market",
        exchange="polymarket",
        question="Will X happen?",
        outcome_yes_price=price,
        volume=1000,
        liquidity=1000,
    )


@pytest.mark.asyncio
async def test_mean_reversion_oversold(analyzer):
    # Price is 0.40, Mean is 0.50 -> 20% below mean
    market = _make_market(0.40)
    history = [0.50] * 10
    
    candidates = await analyzer.analyze_markets([market], {market.id: history})
    
    assert len(candidates) == 1
    assert candidates[0].signal.recommended_side == OrderSide.BUY
    assert "Oversold" in candidates[0].signal.evidence_summary
    assert candidates[0].signal.strategy_source == "technical_mean_reversion"


@pytest.mark.asyncio
async def test_mean_reversion_overbought(analyzer):
    # Price is 0.60, Mean is 0.50 -> 20% above mean
    market = _make_market(0.60)
    history = [0.50] * 10
    
    candidates = await analyzer.analyze_markets([market], {market.id: history})
    
    assert len(candidates) == 1
    assert candidates[0].signal.recommended_side == OrderSide.SELL
    assert "Overbought" in candidates[0].signal.evidence_summary


@pytest.mark.asyncio
async def test_momentum_upward(analyzer):
    # Price moved from 0.40 to 0.43 (7.5% move) with consistent trend
    # Mean is (0.4+0.41+0.42+0.425+0.43)/5 = 0.417
    # Dev = (0.43 - 0.417) / 0.417 = 3.1% (below 10% MR threshold)
    market = _make_market(0.43)
    history = [0.40, 0.41, 0.42, 0.425, 0.43]
    
    candidates = await analyzer.analyze_markets([market], {market.id: history})
    
    assert len(candidates) == 1
    assert candidates[0].signal.recommended_side == OrderSide.BUY
    assert "Momentum (Upward)" in candidates[0].signal.evidence_summary
    assert candidates[0].signal.strategy_source == "technical_momentum"


@pytest.mark.asyncio
async def test_momentum_downward(analyzer):
    # Price moved from 0.50 to 0.47 (6% move) with consistent trend
    # Mean is (0.5+0.49+0.485+0.48+0.47)/5 = 0.485
    # Dev = (0.47 - 0.485) / 0.485 = -3% (below 10% MR threshold)
    market = _make_market(0.47)
    history = [0.50, 0.49, 0.485, 0.48, 0.47]
    
    candidates = await analyzer.analyze_markets([market], {market.id: history})
    
    assert len(candidates) == 1
    assert candidates[0].signal.recommended_side == OrderSide.SELL
    assert "Momentum (Downward)" in candidates[0].signal.evidence_summary


@pytest.mark.asyncio
async def test_no_signal_flat_price(analyzer):
    market = _make_market(0.50)
    history = [0.50] * 10
    
    candidates = await analyzer.analyze_markets([market], {market.id: history})
    
    assert len(candidates) == 0


@pytest.mark.asyncio
async def test_insufficient_history(analyzer):
    market = _make_market(0.40)
    history = [0.50, 0.50] # Only 2 points, min is 5
    
    candidates = await analyzer.analyze_markets([market], {market.id: history})
    
    assert len(candidates) == 0
