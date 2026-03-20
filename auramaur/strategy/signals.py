"""Edge detection — compares Claude probability estimates against market prices."""

from __future__ import annotations

import structlog

from auramaur.exchange.models import Confidence, Market, OrderSide, Signal
from auramaur.nlp.analyzer import AnalysisResult

log = structlog.get_logger()

# Polymarket fee — 0% for reward tier accounts
POLYMARKET_FEE_PCT = 0.0


def _blend_estimates(
    primary: float,
    second_opinion: float | None,
    market_prob: float,
) -> float:
    """Blend primary and second opinion estimates, weighted by agreement.

    When opinions agree: average them (high confidence).
    When opinions diverge: weight toward the more conservative one
    (closer to market price), since disagreement = uncertainty.
    """
    if second_opinion is None:
        return primary

    divergence = abs(primary - second_opinion)

    if divergence < 0.05:
        # Strong agreement — simple average
        return (primary + second_opinion) / 2.0

    # Divergence exists — weight toward the more conservative estimate
    # "Conservative" = closer to market price
    primary_dist = abs(primary - market_prob)
    second_dist = abs(second_opinion - market_prob)

    if primary_dist <= second_dist:
        # Primary is more conservative — weight it more
        conservative, aggressive = primary, second_opinion
    else:
        conservative, aggressive = second_opinion, primary

    # Higher divergence → more weight on conservative estimate
    # At 5% divergence: 60/40 conservative/aggressive
    # At 15% divergence: 80/20 conservative/aggressive
    conserv_weight = min(0.9, 0.5 + divergence * 2)

    return conservative * conserv_weight + aggressive * (1 - conserv_weight)


def detect_edge(market: Market, analysis: AnalysisResult) -> Signal | None:
    """
    Compare Claude's probability estimate against the market price.
    Returns a Signal if there's a tradeable edge, None otherwise.
    """
    if analysis.skipped_reason:
        log.info("signal.skipped", market_id=market.id, reason=analysis.skipped_reason)
        return None

    raw_prob = analysis.calibrated_probability if analysis.calibrated_probability is not None else analysis.probability
    market_prob = market.outcome_yes_price

    # Blend with second opinion using divergence-weighted averaging
    claude_prob = _blend_estimates(raw_prob, analysis.second_opinion_prob, market_prob)

    # Raw edge
    edge = claude_prob - market_prob

    # Determine side
    if edge > 0:
        # Claude thinks YES is underpriced → buy YES
        recommended_side = OrderSide.BUY
    elif edge < 0:
        # Claude thinks YES is overpriced → buy NO (sell YES equivalent)
        recommended_side = OrderSide.SELL
        edge = abs(edge)
    else:
        return None

    # Adjust for fees
    net_edge = edge - (POLYMARKET_FEE_PCT / 100)

    signal = Signal(
        market_id=market.id,
        market_question=market.question,
        claude_prob=claude_prob,
        claude_confidence=Confidence(analysis.confidence),
        market_prob=market_prob,
        edge=net_edge * 100,  # Store as percentage
        second_opinion_prob=analysis.second_opinion_prob,
        divergence=analysis.divergence,
        evidence_summary=analysis.reasoning[:500],
        recommended_side=recommended_side,
    )

    log.debug(
        "signal.detected",
        market_id=market.id,
        claude_prob=f"{claude_prob:.3f}",
        market_prob=f"{market_prob:.3f}",
        edge_pct=f"{signal.edge:.1f}%",
        side=recommended_side.value,
    )

    return signal
