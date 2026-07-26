"""Pure informed-flow candidate assessment shared by every runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from auramaur.experiments.models import MarketSnapshot, PortfolioSnapshot, TargetPosition


class InformedFlowRejection(str, Enum):
    INACTIVE = "inactive"
    WRONG_VENUE = "wrong_venue"
    MISSING_TICKER = "missing_ticker"
    OUTSIDE_PRICE_BAND = "outside_price_band"
    INSUFFICIENT_LIQUIDITY = "insufficient_liquidity"
    BLOCKED_CATEGORY = "blocked_category"
    MISSING_RESOLUTION_TIME = "missing_resolution_time"
    OUTSIDE_RESOLUTION_WINDOW = "outside_resolution_window"
    ALREADY_ENTERED_OR_HELD = "already_entered_or_held"
    NO_INFORMED_FLOW = "no_informed_flow"


@dataclass(frozen=True)
class InformedFlowRules:
    band_lo: float
    band_hi: float
    min_liquidity: float
    min_hours_to_resolution: float
    max_days_to_resolution: float
    uplift: float
    stake_usd: float


@dataclass(frozen=True)
class InformedFlowInputs:
    market_id: str
    venue: str
    ticker: str | None
    active: bool
    market_probability: float
    liquidity: float
    blocked_category: bool
    hours_to_resolution: float | None
    already_entered_or_held: bool
    has_signal: bool
    informed_side: str | None
    abnormal_count: int
    baseline_size: float
    signal_volume: float
    sample: int


@dataclass(frozen=True)
class InformedFlowProposal:
    market_id: str
    buy_yes: bool
    instrument_id: str
    market_probability: float
    model_probability: float
    edge_percent: float
    reference_price: float
    target_quantity: float
    max_notional: float
    evidence_summary: str


@dataclass(frozen=True)
class InformedFlowAssessment:
    proposal: InformedFlowProposal | None
    rejection: InformedFlowRejection | None = None


def informed_flow_eligibility(
    inputs: InformedFlowInputs, rules: InformedFlowRules
) -> InformedFlowRejection | None:
    """Apply cheap deterministic gates before trade-tape acquisition."""
    if not inputs.active:
        return InformedFlowRejection.INACTIVE
    if inputs.venue != "kalshi":
        return InformedFlowRejection.WRONG_VENUE
    if not inputs.ticker:
        return InformedFlowRejection.MISSING_TICKER
    if not rules.band_lo <= inputs.market_probability <= rules.band_hi:
        return InformedFlowRejection.OUTSIDE_PRICE_BAND
    if inputs.liquidity < rules.min_liquidity:
        return InformedFlowRejection.INSUFFICIENT_LIQUIDITY
    if inputs.blocked_category:
        return InformedFlowRejection.BLOCKED_CATEGORY
    if inputs.hours_to_resolution is None:
        return InformedFlowRejection.MISSING_RESOLUTION_TIME
    if (
        inputs.hours_to_resolution < rules.min_hours_to_resolution
        or inputs.hours_to_resolution > rules.max_days_to_resolution * 24.0
    ):
        return InformedFlowRejection.OUTSIDE_RESOLUTION_WINDOW
    return None


def assess_informed_flow(
    inputs: InformedFlowInputs, rules: InformedFlowRules
) -> InformedFlowAssessment:
    rejection = informed_flow_eligibility(inputs, rules)
    if rejection is not None:
        return InformedFlowAssessment(None, rejection)
    if inputs.already_entered_or_held:
        return InformedFlowAssessment(None, InformedFlowRejection.ALREADY_ENTERED_OR_HELD)
    if not inputs.has_signal or inputs.informed_side not in {"yes", "no"}:
        return InformedFlowAssessment(None, InformedFlowRejection.NO_INFORMED_FLOW)

    buy_yes = inputs.informed_side == "yes"
    probability = (
        min(0.99, inputs.market_probability + rules.uplift)
        if buy_yes else max(0.01, inputs.market_probability - rules.uplift)
    )
    reference_price = inputs.market_probability if buy_yes else 1 - inputs.market_probability
    return InformedFlowAssessment(InformedFlowProposal(
        market_id=inputs.market_id,
        buy_yes=buy_yes,
        instrument_id=f"{inputs.market_id}:{'YES' if buy_yes else 'NO'}",
        market_probability=inputs.market_probability,
        model_probability=probability,
        edge_percent=rules.uplift * 100,
        reference_price=reference_price,
        target_quantity=rules.stake_usd / reference_price,
        max_notional=rules.stake_usd,
        evidence_summary=(
            f"Informed-flow follow: {inputs.abnormal_count} abnormal trades "
            f"({inputs.signal_volume:.0f} contracts) on the "
            f"{inputs.informed_side.upper()} side vs a "
            f"{inputs.baseline_size:.0f}-size baseline (n={inputs.sample})."
        ),
    ))


@dataclass(frozen=True)
class InformedFlowExperiment:
    rules: InformedFlowRules

    async def evaluate(
        self, snapshot: MarketSnapshot, portfolio: PortfolioSnapshot
    ) -> list[TargetPosition]:
        del portfolio
        assessment = assess_informed_flow(
            InformedFlowInputs(**snapshot.feature("informed_flow_inputs")), self.rules
        )
        if assessment.proposal is None:
            return []
        proposal = assessment.proposal
        return [TargetPosition(
            instrument_id=proposal.instrument_id,
            target_quantity=proposal.target_quantity,
            reference_price=proposal.reference_price,
            rationale=proposal.evidence_summary,
            max_notional=proposal.max_notional,
        )]
