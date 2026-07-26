"""Pure favorite-longshot band decision used by lab and production."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from auramaur.experiments.models import MarketSnapshot, PortfolioSnapshot, TargetPosition

SOURCE_TAG = "bias_harvest"
SOURCE_TAG_WIDE = "bias_harvest_wide"
LONGSHOT_LO = 0.70
LONGSHOT_HI = 0.90


@dataclass(frozen=True)
class BiasBandDecision:
    favored_price: float
    buy_yes: bool
    action_description: str
    source_tag: str


class BiasRejection(str, Enum):
    OUT_OF_BAND = "out_of_band"
    INACTIVE = "inactive"
    WRONG_VENUE = "wrong_venue"
    DISPUTED = "disputed"
    INSUFFICIENT_LIQUIDITY = "insufficient_liquidity"
    BLOCKED_CATEGORY = "blocked_category"
    MISSING_END = "missing_end"
    TOO_SOON = "too_soon"
    TOO_FAR = "too_far"
    ALREADY_ENTERED_OR_HELD = "already_entered_or_held"
    PAPER_FILL_HAIRCUT = "paper_fill_haircut"
    BOOK_UNAVAILABLE = "book_unavailable"
    SPREAD_TOO_THIN = "spread_too_thin"
    INVALID_MAKER_PRICE = "invalid_maker_price"


@dataclass(frozen=True)
class BiasHarvestRules:
    band_lo: float
    band_hi: float
    edge_uplift: float
    stake_usd: float
    min_liquidity: float
    min_hours_to_resolution: float
    max_days_to_resolution: float
    skip_disputed: bool
    maker_entry: bool
    maker_min_spread: float
    paper: bool


@dataclass(frozen=True)
class BiasHarvestInputs:
    market_id: str
    yes_price: float
    no_price: float
    active: bool
    venue: str
    dispute_risk: str
    liquidity: float
    category_blocked: bool
    end_at: datetime | None
    as_of: datetime
    already_entered_or_held: bool
    paper_admitted: bool
    selected_token_available: bool = True
    best_bid: float | None = None
    best_ask: float | None = None


@dataclass(frozen=True)
class BiasHarvestProposal:
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
    action_description: str
    source_tag: str


@dataclass(frozen=True)
class BiasHarvestAssessment:
    proposal: BiasHarvestProposal | None
    rejection: BiasRejection | None = None


def select_bias_band(
    yes_price: float,
    *,
    band_lo: float,
    band_hi: float,
) -> BiasBandDecision | None:
    """Select the existing bi-directional favorite/longshot regime."""
    if not 0.0 < yes_price < 1.0:
        return None
    favored_price = max(yes_price, 1.0 - yes_price)
    if not band_lo <= favored_price < band_hi:
        return None
    favorite_is_yes = yes_price >= 0.5
    if LONGSHOT_LO <= favored_price < LONGSHOT_HI:
        buy_yes = not favorite_is_yes
        action = "longshot side (favorite overpriced)"
    else:
        buy_yes = favorite_is_yes
        action = "favored side (favorite underpriced)"
    source = SOURCE_TAG if favored_price >= LONGSHOT_HI else SOURCE_TAG_WIDE
    return BiasBandDecision(favored_price, buy_yes, action, source)


def assess_bias_harvest(
    inputs: BiasHarvestInputs,
    rules: BiasHarvestRules,
) -> BiasHarvestAssessment:
    """Pure candidate-to-proposal implementation used by every runtime."""
    band = select_bias_band(
        inputs.yes_price, band_lo=rules.band_lo, band_hi=rules.band_hi
    )
    if band is None:
        return BiasHarvestAssessment(None, BiasRejection.OUT_OF_BAND)
    if not inputs.active:
        return BiasHarvestAssessment(None, BiasRejection.INACTIVE)
    if (inputs.venue or "polymarket") != "polymarket":
        return BiasHarvestAssessment(None, BiasRejection.WRONG_VENUE)
    if rules.skip_disputed and inputs.dispute_risk == "DO_NOT_ACT":
        return BiasHarvestAssessment(None, BiasRejection.DISPUTED)
    if inputs.liquidity < rules.min_liquidity:
        return BiasHarvestAssessment(None, BiasRejection.INSUFFICIENT_LIQUIDITY)
    if inputs.category_blocked:
        return BiasHarvestAssessment(None, BiasRejection.BLOCKED_CATEGORY)
    if inputs.end_at is None:
        return BiasHarvestAssessment(None, BiasRejection.MISSING_END)
    end_at = inputs.end_at
    if end_at.tzinfo is None or end_at.utcoffset() is None:
        end_at = end_at.replace(tzinfo=timezone.utc)
    hours_left = (end_at - inputs.as_of).total_seconds() / 3600.0
    if hours_left < rules.min_hours_to_resolution:
        return BiasHarvestAssessment(None, BiasRejection.TOO_SOON)
    if hours_left > rules.max_days_to_resolution * 24.0:
        return BiasHarvestAssessment(None, BiasRejection.TOO_FAR)
    if inputs.already_entered_or_held:
        return BiasHarvestAssessment(None, BiasRejection.ALREADY_ENTERED_OR_HELD)
    if rules.maker_entry and rules.paper and not inputs.paper_admitted:
        return BiasHarvestAssessment(None, BiasRejection.PAPER_FILL_HAIRCUT)

    selected_price = inputs.yes_price if band.buy_yes else inputs.no_price
    if rules.maker_entry:
        if (
            not inputs.selected_token_available
            or inputs.best_bid is None
            or inputs.best_ask is None
        ):
            return BiasHarvestAssessment(None, BiasRejection.BOOK_UNAVAILABLE)
        if inputs.best_ask - inputs.best_bid < rules.maker_min_spread:
            return BiasHarvestAssessment(None, BiasRejection.SPREAD_TOO_THIN)
        selected_price = round(inputs.best_bid, 2)
        if not 0.01 <= selected_price <= 0.99:
            return BiasHarvestAssessment(None, BiasRejection.INVALID_MAKER_PRICE)
    fair = (
        min(0.99, inputs.yes_price + rules.edge_uplift)
        if band.buy_yes
        else max(0.01, inputs.yes_price - rules.edge_uplift)
    )
    return BiasHarvestAssessment(BiasHarvestProposal(
        market_id=inputs.market_id,
        buy_yes=band.buy_yes,
        instrument_id=f"{inputs.market_id}:{'YES' if band.buy_yes else 'NO'}",
        favored_price=band.favored_price,
        market_probability=inputs.yes_price,
        fair_probability=fair,
        edge_percent=rules.edge_uplift * 100.0,
        entry_price=selected_price,
        target_quantity=rules.stake_usd / selected_price,
        max_notional=rules.stake_usd,
        action_description=band.action_description,
        source_tag=band.source_tag,
    ))


@dataclass(frozen=True)
class BiasHarvestExperiment:
    """Portable target generator for the migrated band decision only."""

    rules: BiasHarvestRules

    async def evaluate(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioSnapshot,
    ) -> list[TargetPosition]:
        del portfolio
        details = snapshot.feature("bias_harvest_inputs")
        assessment = assess_bias_harvest(BiasHarvestInputs(
            market_id=snapshot.market_id,
            yes_price=float(details["yes_price"]),
            no_price=float(details["no_price"]),
            active=bool(details["active"]),
            venue=snapshot.venue,
            dispute_risk=str(details["dispute_risk"]),
            liquidity=float(details["liquidity"]),
            category_blocked=bool(details["category_blocked"]),
            end_at=datetime.fromisoformat(details["end_at"]) if details["end_at"] else None,
            as_of=snapshot.observed_at,
            already_entered_or_held=bool(details["already_entered_or_held"]),
            paper_admitted=bool(details["paper_admitted"]),
            selected_token_available=bool(details["selected_token_available"]),
            best_bid=details.get("best_bid"),
            best_ask=details.get("best_ask"),
        ), self.rules)
        if assessment.proposal is None:
            return []
        proposal = assessment.proposal
        return [TargetPosition(
            instrument_id=proposal.instrument_id,
            target_quantity=proposal.target_quantity,
            reference_price=proposal.entry_price,
            rationale=proposal.action_description,
            max_notional=proposal.max_notional,
        )]
