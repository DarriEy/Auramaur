"""Pure long-horizon favorite-underpricing candidate assessment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from auramaur.experiments.models import MarketSnapshot, PortfolioSnapshot, TargetPosition


def calibrated_fair(price: float, slope: float) -> float:
    p = min(max(price, 1e-6), 1.0 - 1e-6)
    logit = math.log(p / (1.0 - p))
    return 1.0 / (1.0 + math.exp(-slope * logit))


class LongHorizonRejection(str, Enum):
    OUT_OF_BAND = "out_of_band"
    INACTIVE = "inactive"
    WRONG_VENUE = "wrong_venue"
    INSUFFICIENT_LIQUIDITY = "insufficient_liquidity"
    BLOCKED_CATEGORY = "blocked_category"
    MISSING_END = "missing_end"
    TOO_SOON = "too_soon"
    TOO_FAR = "too_far"
    ALREADY_ENTERED_OR_HELD = "already_entered_or_held"
    INSUFFICIENT_EDGE = "insufficient_edge"


@dataclass(frozen=True)
class LongHorizonRules:
    venue: str
    band_lo: float
    band_hi: float
    slope: float
    min_edge: float
    stake_usd: float
    min_liquidity: float
    min_days_to_resolution: float
    max_days_to_resolution: float
    source_tag: str


@dataclass(frozen=True)
class LongHorizonInputs:
    market_id: str
    yes_price: float
    active: bool
    venue: str
    liquidity: float
    category_blocked: bool
    end_at: datetime | None
    as_of: datetime
    already_entered_or_held: bool


@dataclass(frozen=True)
class LongHorizonProposal:
    market_id: str
    buy_yes: bool
    instrument_id: str
    favored_price: float
    market_probability: float
    fair_probability: float
    edge_percent: float
    entry_price: float
    target_quantity: float
    max_notional: float
    source_tag: str


@dataclass(frozen=True)
class LongHorizonAssessment:
    proposal: LongHorizonProposal | None
    rejection: LongHorizonRejection | None = None


def assess_long_horizon(
    inputs: LongHorizonInputs, rules: LongHorizonRules
) -> LongHorizonAssessment:
    p_yes = inputs.yes_price
    if not 0.0 < p_yes < 1.0:
        return LongHorizonAssessment(None, LongHorizonRejection.OUT_OF_BAND)
    p_fav = max(p_yes, 1.0 - p_yes)
    if not rules.band_lo <= p_fav <= rules.band_hi:
        return LongHorizonAssessment(None, LongHorizonRejection.OUT_OF_BAND)
    if not inputs.active:
        return LongHorizonAssessment(None, LongHorizonRejection.INACTIVE)
    if (inputs.venue or "polymarket") != rules.venue:
        return LongHorizonAssessment(None, LongHorizonRejection.WRONG_VENUE)
    if inputs.liquidity < rules.min_liquidity:
        return LongHorizonAssessment(None, LongHorizonRejection.INSUFFICIENT_LIQUIDITY)
    if inputs.category_blocked:
        return LongHorizonAssessment(None, LongHorizonRejection.BLOCKED_CATEGORY)
    if inputs.end_at is None:
        return LongHorizonAssessment(None, LongHorizonRejection.MISSING_END)
    end_at = inputs.end_at
    if end_at.tzinfo is None or end_at.utcoffset() is None:
        end_at = end_at.replace(tzinfo=timezone.utc)
    days_left = (end_at - inputs.as_of).total_seconds() / 86400.0
    if days_left < rules.min_days_to_resolution:
        return LongHorizonAssessment(None, LongHorizonRejection.TOO_SOON)
    if days_left > rules.max_days_to_resolution:
        return LongHorizonAssessment(None, LongHorizonRejection.TOO_FAR)
    if inputs.already_entered_or_held:
        return LongHorizonAssessment(None, LongHorizonRejection.ALREADY_ENTERED_OR_HELD)
    buy_yes = p_yes >= 0.5
    fair_fav = calibrated_fair(p_fav, rules.slope)
    edge = fair_fav - p_fav
    if edge < rules.min_edge:
        return LongHorizonAssessment(None, LongHorizonRejection.INSUFFICIENT_EDGE)
    fair_yes = min(0.99, fair_fav) if buy_yes else max(0.01, 1.0 - fair_fav)
    entry = p_yes if buy_yes else 1.0 - p_yes
    return LongHorizonAssessment(LongHorizonProposal(
        market_id=inputs.market_id,
        buy_yes=buy_yes,
        instrument_id=f"{inputs.market_id}:{'YES' if buy_yes else 'NO'}",
        favored_price=p_fav,
        market_probability=p_yes,
        fair_probability=fair_yes,
        edge_percent=edge * 100.0,
        entry_price=entry,
        target_quantity=rules.stake_usd / entry,
        max_notional=rules.stake_usd,
        source_tag=rules.source_tag,
    ))


@dataclass(frozen=True)
class LongHorizonExperiment:
    rules: LongHorizonRules

    async def evaluate(
        self, snapshot: MarketSnapshot, portfolio: PortfolioSnapshot
    ) -> list[TargetPosition]:
        del portfolio
        details = snapshot.feature("long_horizon_inputs")
        assessment = assess_long_horizon(LongHorizonInputs(
            market_id=snapshot.market_id,
            yes_price=float(details["yes_price"]),
            active=bool(details["active"]),
            venue=snapshot.venue,
            liquidity=float(details["liquidity"]),
            category_blocked=bool(details["category_blocked"]),
            end_at=datetime.fromisoformat(details["end_at"]) if details["end_at"] else None,
            as_of=snapshot.observed_at,
            already_entered_or_held=bool(details["already_entered_or_held"]),
        ), self.rules)
        if assessment.proposal is None:
            return []
        proposal = assessment.proposal
        return [TargetPosition(
            instrument_id=proposal.instrument_id,
            target_quantity=proposal.target_quantity,
            reference_price=proposal.entry_price,
            rationale="long-horizon structural underconfidence",
            max_notional=proposal.max_notional,
        )]
