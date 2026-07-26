"""Pure proposal formation for the core LLM trading family."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum

from auramaur.experiments.models import MarketSnapshot, PortfolioSnapshot, TargetPosition


class CoreTradingRejection(str, Enum):
    INVALID_PROBABILITY = "invalid_probability"
    NO_EDGE = "no_edge"
    INVERTED_SEMANTICS = "inverted_semantics"
    FEES_CONSUME_EDGE = "fees_consume_edge"


@dataclass(frozen=True)
class CoreTradingRules:
    blend_second_opinion: bool
    reject_suspicious_negation: bool
    compute_quality: bool
    max_notional: float
    require_positive_net_edge: bool = True


@dataclass(frozen=True)
class CoreTradingInputs:
    market_id: str
    question: str
    market_probability: float
    model_probability: float
    confidence: str
    second_opinion_probability: float | None
    divergence: float | None
    reasoning: str
    evidence_count: int
    hours_to_resolution: float | None
    fee_rate: float


@dataclass(frozen=True)
class CoreTradingProposal:
    market_id: str
    instrument_id: str
    side: str
    market_probability: float
    model_probability: float
    edge_percent: float
    reference_price: float
    target_quantity: float
    max_notional: float
    quality: float | None
    evidence_summary: str


@dataclass(frozen=True)
class CoreTradingAssessment:
    proposal: CoreTradingProposal | None
    rejection: CoreTradingRejection | None = None


def blend_estimates(
    primary: float,
    second_opinion: float | None,
    market_probability: float,
    primary_confidence: str = "MEDIUM",
) -> float:
    del market_probability  # retained in the stable production helper signature
    if second_opinion is None:
        return primary
    divergence = abs(primary - second_opinion)
    primary_weight = {
        "HIGH": 0.75,
        "MEDIUM_HIGH": 0.675,
        "MEDIUM": 0.60,
        "MEDIUM_LOW": 0.525,
        "LOW": 0.45,
    }.get(primary_confidence, 0.60)
    if divergence < 0.05:
        return primary * primary_weight + second_opinion * (1 - primary_weight)
    if divergence > 0.25:
        midpoint = (primary + second_opinion) / 2
        shrink = min(0.3, divergence - 0.25)
        return midpoint * (1 - shrink) + 0.5 * shrink
    return primary * primary_weight + second_opinion * (1 - primary_weight)


def has_inverted_semantics(question: str) -> bool:
    patterns = (
        r"\bnot\b", r"\bnever\b", r"\bfail(?:s|ed)? to\b",
        r"\bwon'?t\b", r"\bunable to\b", r"\bfalls? short\b",
        r"\bno longer\b", r"\bavoid\b",
    )
    return any(re.search(pattern, question.lower()) for pattern in patterns)


def compute_signal_quality(
    edge: float,
    confidence: str,
    divergence: float | None,
    evidence_count: int,
    hours_to_resolution: float | None,
) -> float:
    edge_score = min(1.0, abs(edge) / 20.0)
    conf_score = {
        "HIGH": 1.0, "MEDIUM_HIGH": 0.85, "MEDIUM": 0.7,
        "MEDIUM_LOW": 0.55, "LOW": 0.4,
    }.get(confidence, 0.5)
    div_score = 1.0 if divergence is None else max(0.0, 1.0 - divergence * 4)
    ev_score = min(1.0, 0.2 + evidence_count * 0.08)
    time_score = 0.5
    if hours_to_resolution is not None and hours_to_resolution > 0:
        time_score = max(
            0.2, 1.0 - math.log(hours_to_resolution / 24 + 1) / 6
        )
    return round(
        edge_score * 0.30
        + conf_score * 0.25
        + div_score * 0.20
        + ev_score * 0.15
        + time_score * 0.10,
        3,
    )


def assess_core_trading(
    inputs: CoreTradingInputs,
    rules: CoreTradingRules,
) -> CoreTradingAssessment:
    probabilities = (
        inputs.market_probability,
        inputs.model_probability,
        inputs.second_opinion_probability,
    )
    if any(
        value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0)
        for value in probabilities
    ) or not math.isfinite(inputs.fee_rate) or inputs.fee_rate < 0:
        return CoreTradingAssessment(None, CoreTradingRejection.INVALID_PROBABILITY)

    model_probability = inputs.model_probability
    if rules.blend_second_opinion:
        model_probability = blend_estimates(
            model_probability,
            inputs.second_opinion_probability,
            inputs.market_probability,
            inputs.confidence,
        )
    raw_edge = model_probability - inputs.market_probability
    if raw_edge == 0 or (
        not rules.blend_second_opinion and abs(raw_edge) < 0.001
    ):
        return CoreTradingAssessment(None, CoreTradingRejection.NO_EDGE)
    side = "BUY" if raw_edge > 0 else "SELL"
    absolute_edge = abs(raw_edge)
    if (
        rules.reject_suspicious_negation
        and absolute_edge > 0.25
        and has_inverted_semantics(inputs.question)
    ):
        return CoreTradingAssessment(None, CoreTradingRejection.INVERTED_SEMANTICS)
    net_edge = (
        absolute_edge
        - inputs.fee_rate
        * inputs.market_probability
        * (1.0 - inputs.market_probability)
    )
    if rules.require_positive_net_edge and net_edge <= 0:
        return CoreTradingAssessment(None, CoreTradingRejection.FEES_CONSUME_EDGE)
    quality = None
    if rules.compute_quality:
        quality = compute_signal_quality(
            net_edge * 100,
            inputs.confidence,
            inputs.divergence,
            inputs.evidence_count,
            inputs.hours_to_resolution,
        )
    reference_price = (
        inputs.market_probability if side == "BUY"
        else 1.0 - inputs.market_probability
    )
    return CoreTradingAssessment(CoreTradingProposal(
        market_id=inputs.market_id,
        instrument_id=f"{inputs.market_id}:{'YES' if side == 'BUY' else 'NO'}",
        side=side,
        market_probability=inputs.market_probability,
        model_probability=model_probability,
        edge_percent=net_edge * 100,
        reference_price=reference_price,
        target_quantity=rules.max_notional / reference_price,
        max_notional=rules.max_notional,
        quality=quality,
        evidence_summary=inputs.reasoning[:500],
    ))


@dataclass(frozen=True)
class CoreTradingExperiment:
    rules: CoreTradingRules

    async def evaluate(
        self, snapshot: MarketSnapshot, portfolio: PortfolioSnapshot
    ) -> list[TargetPosition]:
        del portfolio
        assessment = assess_core_trading(
            CoreTradingInputs(**snapshot.feature("core_trading_inputs")), self.rules
        )
        if assessment.proposal is None:
            return []
        proposal = assessment.proposal
        return [TargetPosition(
            instrument_id=proposal.instrument_id,
            target_quantity=proposal.target_quantity,
            reference_price=proposal.reference_price,
            rationale=proposal.evidence_summary or "core trading probability edge",
            max_notional=proposal.max_notional,
        )]
