"""Pure economic-indicator candidate decision used by lab and production."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from auramaur.experiments.models import MarketSnapshot, PortfolioSnapshot, TargetPosition


class EconIndicatorRejection(str, Enum):
    INSUFFICIENT_EDGE = "insufficient_edge"
    IMPLAUSIBLE_DIVERGENCE = "implausible_divergence"
    ALREADY_ENTERED_OR_HELD = "already_entered_or_held"


@dataclass(frozen=True)
class EconIndicatorInputs:
    market_id: str
    market_probability: float
    model_probability: float
    required_edge: float
    max_divergence: float
    already_entered_or_held: bool


@dataclass(frozen=True)
class EconIndicatorProposal:
    market_id: str
    buy_yes: bool
    instrument_id: str
    market_probability: float
    model_probability: float
    signal_probability: float
    edge_percent: float


@dataclass(frozen=True)
class EconIndicatorAssessment:
    proposal: EconIndicatorProposal | None
    rejection: EconIndicatorRejection | None = None


def assess_econ_indicator(inputs: EconIndicatorInputs) -> EconIndicatorAssessment:
    """Turn a priced ladder member into a direction, without side effects."""
    edge = inputs.model_probability - inputs.market_probability
    if abs(edge) < inputs.required_edge:
        return EconIndicatorAssessment(
            None, EconIndicatorRejection.INSUFFICIENT_EDGE
        )
    if abs(edge) > inputs.max_divergence:
        return EconIndicatorAssessment(
            None, EconIndicatorRejection.IMPLAUSIBLE_DIVERGENCE
        )
    if inputs.already_entered_or_held:
        return EconIndicatorAssessment(
            None, EconIndicatorRejection.ALREADY_ENTERED_OR_HELD
        )
    buy_yes = edge > 0
    return EconIndicatorAssessment(EconIndicatorProposal(
        market_id=inputs.market_id,
        buy_yes=buy_yes,
        instrument_id=f"{inputs.market_id}:{'YES' if buy_yes else 'NO'}",
        market_probability=inputs.market_probability,
        model_probability=inputs.model_probability,
        signal_probability=max(0.01, min(0.99, inputs.model_probability)),
        edge_percent=abs(edge) * 100.0,
    ))


@dataclass(frozen=True)
class EconIndicatorExperiment:
    """Portable target generator for a pre-priced economic ladder member."""

    stake_usd: float

    async def evaluate(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioSnapshot,
    ) -> list[TargetPosition]:
        details = snapshot.feature("econ_indicator_inputs")
        assessment = assess_econ_indicator(EconIndicatorInputs(
            market_id=snapshot.market_id,
            market_probability=float(details["market_probability"]),
            model_probability=float(details["model_probability"]),
            required_edge=float(details["required_edge"]),
            max_divergence=float(details["max_divergence"]),
            already_entered_or_held=(
                bool(details["already_entered_or_held"])
                or bool(portfolio.quantities().get(snapshot.market_id + ":YES", 0.0))
                or bool(portfolio.quantities().get(snapshot.market_id + ":NO", 0.0))
            ),
        ))
        if assessment.proposal is None:
            return []
        proposal = assessment.proposal
        price = (
            proposal.market_probability
            if proposal.buy_yes
            else 1.0 - proposal.market_probability
        )
        return [TargetPosition(
            instrument_id=proposal.instrument_id,
            target_quantity=self.stake_usd / price,
            reference_price=price,
            rationale="economic indicator distribution versus market",
            max_notional=self.stake_usd,
        )]
