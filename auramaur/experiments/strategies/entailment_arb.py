"""Pure formation of atomic entailment-arbitrage pair proposals."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EntailmentMarket:
    """Point-in-time market facts used by the structural bound."""

    market_id: str
    question: str
    yes_probability: float


@dataclass(frozen=True)
class EntailmentLeg:
    market_id: str
    question: str
    side: str
    market_probability: float
    fair_probability: float
    edge_percent: float


@dataclass(frozen=True)
class EntailmentPairProposal:
    """An indivisible two-leg proposal; neither leg is a standalone target."""

    leg_a: EntailmentLeg
    leg_b: EntailmentLeg
    gap: float
    rationale: str
    confidence: float
    negative_implication: bool


def form_entailment_pair(
    implier: EntailmentMarket,
    implied: EntailmentMarket,
    *,
    rationale: str,
    confidence: float,
    minimum_gap: float,
    negative_implication: bool = False,
) -> EntailmentPairProposal | None:
    """Apply the implication bound and form both legs, or fail closed."""
    values = (
        implier.yes_probability,
        implied.yes_probability,
        confidence,
        minimum_gap,
    )
    if (
        not implier.market_id
        or not implied.market_id
        or implier.market_id == implied.market_id
        or not rationale.strip()
        or any(not math.isfinite(value) for value in values)
        or not 0.0 < implier.yes_probability < 1.0
        or not 0.0 < implied.yes_probability < 1.0
        or not 0.0 <= confidence <= 1.0
        or minimum_gap < 0.0
    ):
        return None

    if negative_implication:
        gap = implier.yes_probability + implied.yes_probability - 1.0
        side_a = side_b = "SELL"
        fair_a = 1.0 - implied.yes_probability
        fair_b = 1.0 - implier.yes_probability
    else:
        gap = implier.yes_probability - implied.yes_probability
        side_a, side_b = "SELL", "BUY"
        fair_a = implied.yes_probability
        fair_b = implier.yes_probability

    if not math.isfinite(gap) or gap < minimum_gap:
        return None

    edge_percent = gap * 100.0
    return EntailmentPairProposal(
        leg_a=EntailmentLeg(
            implier.market_id,
            implier.question,
            side_a,
            implier.yes_probability,
            fair_a,
            edge_percent,
        ),
        leg_b=EntailmentLeg(
            implied.market_id,
            implied.question,
            side_b,
            implied.yes_probability,
            fair_b,
            edge_percent,
        ),
        gap=gap,
        rationale=rationale,
        confidence=confidence,
        negative_implication=negative_implication,
    )
