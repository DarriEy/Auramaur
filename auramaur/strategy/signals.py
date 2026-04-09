"""Edge detection — compares Claude probability estimates against market prices."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import structlog

from auramaur.exchange.models import Confidence, Market, OrderSide, Signal
from auramaur.nlp.analyzer import AnalysisResult

log = structlog.get_logger()

# Per-exchange fee rates (as fractions, not percentages).
# Applied to edge calculation so only genuinely profitable trades pass.
EXCHANGE_FEES: dict[str, float] = {
    "polymarket": 0.0,    # 0% for reward tier accounts
    "kalshi": 0.07,       # 7% fee on winnings
    "cryptodotcom": 0.01, # ~1% fee
}

# Legacy alias for backward compatibility
POLYMARKET_FEE_PCT = 0.0


def _blend_estimates(
    primary: float,
    second_opinion: float | None,
    market_prob: float,
    primary_confidence: str = "MEDIUM",
) -> float:
    """Blend primary and second opinion using confidence-weighted averaging.

    Key principles:
    1. Higher-confidence estimates get more weight
    2. When opinions AGREE, average them (high conviction)
    3. When opinions DIVERGE, weight by confidence — NOT toward market price
       (biasing toward market erases the edge we're trying to capture)
    4. Large divergence with low confidence = shrink toward 50% (uncertainty)
    """
    if second_opinion is None:
        return primary

    divergence = abs(primary - second_opinion)

    # Confidence weight for primary
    conf_weights = {"HIGH": 0.75, "MEDIUM_HIGH": 0.675, "MEDIUM": 0.60, "MEDIUM_LOW": 0.525, "LOW": 0.45}
    primary_weight = conf_weights.get(primary_confidence, 0.60)

    if divergence < 0.05:
        # Strong agreement — confidence-weighted average
        return primary * primary_weight + second_opinion * (1 - primary_weight)

    if divergence > 0.25:
        # Massive divergence = genuine uncertainty
        # Shrink toward uncertainty (not toward market — that erases edge)
        midpoint = (primary + second_opinion) / 2
        uncertainty_pull = 0.5  # Pure uncertainty
        shrink = min(0.3, divergence - 0.25)  # How much to shrink
        return midpoint * (1 - shrink) + uncertainty_pull * shrink

    # Moderate divergence — weight by confidence
    return primary * primary_weight + second_opinion * (1 - primary_weight)


def _compute_signal_quality(
    edge: float,
    confidence: str,
    divergence: float | None,
    evidence_count: int,
    hours_to_resolution: float | None,
) -> float:
    """Compute a composite signal quality score (0-1).

    Combines:
    - Edge magnitude (larger = better)
    - Confidence tier
    - Low divergence (agreement = conviction)
    - Evidence quantity (more = better informed)
    - Time value (sooner resolution = capital freed faster)
    """
    # Edge component (0-1): 5% edge = 0.25, 20% edge = 1.0
    edge_score = min(1.0, abs(edge) / 20.0)

    # Confidence component
    conf_score = {"HIGH": 1.0, "MEDIUM_HIGH": 0.85, "MEDIUM": 0.7, "MEDIUM_LOW": 0.55, "LOW": 0.4}.get(confidence, 0.5)

    # Divergence component (0-1): 0% div = 1.0, 25% div = 0.0
    div_score = 1.0
    if divergence is not None:
        div_score = max(0.0, 1.0 - divergence * 4)

    # Evidence component (0-1): 0 items = 0.2, 5+ items = 0.8, 10+ = 1.0
    ev_score = min(1.0, 0.2 + evidence_count * 0.08)

    # Time value component (0-1): resolves tomorrow = 1.0, 6 months = 0.3
    time_score = 0.5  # default for unknown
    if hours_to_resolution is not None and hours_to_resolution > 0:
        # Log scale: 24h = 1.0, 720h (30d) = 0.6, 4320h (180d) = 0.3
        time_score = max(0.2, 1.0 - math.log(hours_to_resolution / 24 + 1) / 6)

    # Weighted composite
    quality = (
        edge_score * 0.30
        + conf_score * 0.25
        + div_score * 0.20
        + ev_score * 0.15
        + time_score * 0.10
    )

    return round(quality, 3)


def _has_inverted_semantics(question: str) -> bool:
    """Detect if a market question has inverted/negated semantics.

    Markets phrased as "Will X NOT happen?" can confuse the probability
    estimate — Claude may answer the affirmative form while the market
    trades the negation.
    """
    import re
    q = question.lower()
    # Match negation words as standalone tokens (not substrings)
    _NEGATION_REGEXES = [
        r"\bnot\b", r"\bnever\b", r"\bfail(?:s|ed)? to\b",
        r"\bwon'?t\b", r"\bunable to\b", r"\bfalls? short\b",
        r"\bno longer\b", r"\bavoid\b",
    ]
    return any(re.search(pat, q) for pat in _NEGATION_REGEXES)


def detect_edge(market: Market, analysis: AnalysisResult) -> Signal | None:
    """Compare Claude's probability estimate against the market price.

    Returns a Signal if there's a tradeable edge, None otherwise.
    """
    if analysis.skipped_reason:
        log.info("signal.skipped", market_id=market.id, reason=analysis.skipped_reason)
        return None

    raw_prob = analysis.calibrated_probability if analysis.calibrated_probability is not None else analysis.probability
    market_prob = market.outcome_yes_price

    # Blend with second opinion using confidence-weighted averaging
    claude_prob = _blend_estimates(
        raw_prob, analysis.second_opinion_prob, market_prob,
        primary_confidence=analysis.confidence,
    )

    # Detect inverted semantics — if the question is negated, Claude may
    # have estimated the affirmative while the market trades the negation.
    # When edge is suspiciously large (>25%) and semantics look inverted,
    # skip the signal entirely rather than risk a miscalibrated trade.
    inverted = _has_inverted_semantics(market.question)

    # Raw edge
    edge = claude_prob - market_prob

    # Determine side
    if edge > 0:
        recommended_side = OrderSide.BUY
    elif edge < 0:
        recommended_side = OrderSide.SELL
        edge = abs(edge)
    else:
        return None

    if inverted and edge > 0.25:
        log.warning(
            "signal.inverted_semantics",
            market_id=market.id,
            question=market.question[:100],
            claude_prob=f"{claude_prob:.3f}",
            market_prob=f"{market_prob:.3f}",
            edge_pct=f"{edge * 100:.1f}%",
            hint="Question appears negated and edge is suspiciously large — skipping",
        )
        return None

    # Adjust for exchange-specific fees
    fee_rate = EXCHANGE_FEES.get(market.exchange, 0.0)
    net_edge = edge - fee_rate

    # Compute hours to resolution for signal quality
    hours_to_res = None
    if market.end_date is not None:
        end = market.end_date if market.end_date.tzinfo else market.end_date.replace(tzinfo=timezone.utc)
        hours_to_res = max(0, (end - datetime.now(timezone.utc)).total_seconds() / 3600)

    # Evidence count from the reasoning length as proxy
    evidence_count = len(analysis.key_factors) if analysis.key_factors else 0

    quality = _compute_signal_quality(
        net_edge * 100,
        analysis.confidence,
        analysis.divergence,
        evidence_count,
        hours_to_res,
    )

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
        recommended_size=quality,  # Repurpose as signal quality score
    )

    log.debug(
        "signal.detected",
        market_id=market.id,
        claude_prob=f"{claude_prob:.3f}",
        market_prob=f"{market_prob:.3f}",
        edge_pct=f"{signal.edge:.1f}%",
        side=recommended_side.value,
        quality=quality,
    )

    # Flag suspiciously large edges — likely a calibration problem, not alpha
    if signal.edge > 30:
        log.warning(
            "signal.suspicious_edge",
            market_id=market.id,
            edge_pct=f"{signal.edge:.1f}%",
            claude_prob=f"{claude_prob:.3f}",
            market_prob=f"{market_prob:.3f}",
            hint="Edge >30% is rare — check if market question semantics are inverted or if Claude is miscalibrated",
        )

    return signal
