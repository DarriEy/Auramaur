from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from auramaur.experiments.strategies.momentum_coupling import (
    CouplingCandidate, CouplingInputs, CouplingRejection, CouplingRules,
    assess_momentum_coupling,
)
from auramaur.strategy.momentum_coupling import MomentumCouplingPillar

RULES = CouplingRules(0.5, 0.05, 20.0)

def test_proposal_matches_formula_and_skips_ineligible_candidates():
    result = assess_momentum_coupling(CouplingInputs("BTC", 50_000, 2.0, (
        CouplingCandidate("far", "Bitcoin above $60,000?", 0.5),
        CouplingCandidate("live", "Bitcoin above $50,000?", 0.4),
    )), RULES)
    proposal = result.proposal
    assert proposal is not None
    assert proposal.market_id == "live"
    assert proposal.direction == "BUY_YES"
    assert proposal.fair_probability == pytest.approx(0.5)
    assert proposal.edge_percent == pytest.approx(10.0)
    assert proposal.target_quantity == pytest.approx(50.0)
    assert proposal.confidence == "HIGH"

def test_negative_move_preserves_shift_and_medium_confidence():
    result = assess_momentum_coupling(CouplingInputs("ETH", 4_000, -0.75, (
        CouplingCandidate("m", "Ethereum above $4,000?", 0.5),
    )), RULES)
    proposal = result.proposal
    assert proposal is not None
    assert proposal.direction == "BUY_NO"
    assert proposal.fair_probability == pytest.approx(0.4625)
    assert proposal.confidence == "MEDIUM"

def test_small_move_and_pinned_market_are_rejected():
    small = assess_momentum_coupling(CouplingInputs("BTC", 50_000, 0.1, ()), RULES)
    assert small.rejection is CouplingRejection.MOVE_TOO_SMALL
    pinned = assess_momentum_coupling(CouplingInputs("BTC", 50_000, 1.0, (
        CouplingCandidate("m", "Bitcoin above $50,000?", 0.95),
    )), RULES)
    assert pinned.rejection is CouplingRejection.PINNED_MARKET

@pytest.mark.asyncio
async def test_production_detect_only_has_no_execution_side_effect():
    cfg = SimpleNamespace(move_threshold_pct=0.5, near_money_pct=0.05,
                          max_position_usd=20.0, execute=False)
    market = SimpleNamespace(id="m", question="Will Bitcoin be above $50,000?",
                             outcome_yes_price=0.4)
    pillar = MomentumCouplingPillar(SimpleNamespace(momentum_coupling=cfg))
    pillar._gamma = SimpleNamespace(search_markets=AsyncMock(return_value=[market]))
    pillar._execute = AsyncMock()
    await pillar._emit("BTC", 50_000, 1.0)
    pillar._execute.assert_not_awaited()
