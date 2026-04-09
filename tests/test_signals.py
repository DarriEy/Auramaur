"""Tests for signal/edge detection."""

import pytest
from auramaur.exchange.models import Market, Confidence
from auramaur.nlp.analyzer import AnalysisResult
from auramaur.strategy.signals import detect_edge


def _make_market(yes_price: float = 0.5) -> Market:
    return Market(
        id="test-market",
        condition_id="cond-1",
        question="Will X happen?",
        outcome_yes_price=yes_price,
        outcome_no_price=1 - yes_price,
    )


def _make_analysis(prob: float = 0.7, confidence: str = "HIGH", skipped: str | None = None) -> AnalysisResult:
    return AnalysisResult(
        probability=prob,
        confidence=confidence,
        reasoning="test reasoning",
        key_factors=["factor1"],
        time_sensitivity="LOW",
        skipped_reason=skipped,
    )


def test_positive_edge_buy():
    signal = detect_edge(_make_market(0.5), _make_analysis(0.7))
    assert signal is not None
    assert signal.recommended_side.value == "BUY"
    assert signal.edge > 0


def test_negative_edge_sell():
    signal = detect_edge(_make_market(0.7), _make_analysis(0.5))
    assert signal is not None
    assert signal.recommended_side.value == "SELL"


def test_no_edge():
    signal = detect_edge(_make_market(0.5), _make_analysis(0.5))
    assert signal is None


def test_skipped_analysis():
    signal = detect_edge(_make_market(0.5), _make_analysis(skipped="divergence too high"))
    assert signal is None


def test_edge_accounts_for_fees():
    # 3% raw edge, 0% fee = 3% net edge (reward tier)
    signal = detect_edge(_make_market(0.50), _make_analysis(0.53))
    assert signal is not None
    assert signal.edge == pytest.approx(3.0, abs=0.5)


# ---------------------------------------------------------------------------
# Divergence-weighted blending tests
# ---------------------------------------------------------------------------

def test_blend_no_second_opinion():
    """With no second opinion, should use primary estimate."""
    from auramaur.strategy.signals import _blend_estimates
    result = _blend_estimates(0.7, None, 0.5)
    assert result == 0.7


def test_blend_agreement_averages():
    """When opinions agree closely, should average them."""
    from auramaur.strategy.signals import _blend_estimates
    result = _blend_estimates(0.70, 0.72, 0.50)
    assert abs(result - 0.71) < 0.01


def test_blend_divergence_weights_by_confidence():
    """When opinions diverge, should weight by confidence not market proximity."""
    from auramaur.strategy.signals import _blend_estimates
    # Primary says 0.80 (MEDIUM conf = 0.60 weight), second says 0.60
    # Result should be 0.80 * 0.60 + 0.60 * 0.40 = 0.72
    result = _blend_estimates(0.80, 0.60, 0.50, primary_confidence="MEDIUM")
    assert 0.68 < result < 0.76  # Weighted toward primary


def test_blend_both_above_market():
    """When both opinions are above market, blend should stay above."""
    from auramaur.strategy.signals import _blend_estimates
    result = _blend_estimates(0.75, 0.65, 0.50)
    assert result > 0.50  # Both agree it's underpriced


def test_inverted_semantics_detected():
    """Questions with negation patterns should be flagged as inverted."""
    from auramaur.strategy.signals import _has_inverted_semantics
    assert _has_inverted_semantics("Will Biden NOT run for re-election?")
    assert _has_inverted_semantics("Will Tesla fail to reach $300 by 2027?")
    assert _has_inverted_semantics("Will someone NOT become a trillionaire?")
    assert not _has_inverted_semantics("Will Biden win the 2024 election?")
    assert not _has_inverted_semantics("Will Tesla reach $300 by 2027?")


def test_inverted_semantics_large_edge_skipped():
    """Large edge on inverted-semantic market should return no signal."""
    market = Market(
        id="test-inv", condition_id="c", question="Will it fail to exceed $30M?",
        outcome_yes_price=0.86,
    )
    analysis = _make_analysis(prob=0.40, confidence="HIGH")
    signal = detect_edge(market, analysis)
    assert signal is None  # 46% edge + negation = skip


def test_inverted_semantics_small_edge_passes():
    """Small edges on inverted markets should still produce signals."""
    market = Market(
        id="test-inv2", condition_id="c", question="Will X fail to reach the target?",
        outcome_yes_price=0.60,
    )
    analysis = _make_analysis(prob=0.70, confidence="HIGH")  # 10% edge
    signal = detect_edge(market, analysis)
    assert signal is not None  # Below 25% threshold, trust the model


def test_calibrated_probability_used_in_edge():
    """detect_edge should prefer calibrated_probability when available."""
    market = Market(
        id="test", condition_id="c", question="Test?",
        outcome_yes_price=0.50,
    )
    analysis = AnalysisResult(
        probability=0.80,  # raw
        calibrated_probability=0.65,  # calibrated (should be used)
        confidence="HIGH",
    )
    signal = detect_edge(market, analysis)
    # Edge should be based on 0.65, not 0.80
    assert signal is not None
    assert signal.claude_prob < 0.70  # Uses calibrated, not raw
