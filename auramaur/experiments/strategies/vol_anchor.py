"""Pure deterministic candidate-to-proposal logic for vol-anchor markets."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from auramaur.experiments.models import MarketSnapshot, PortfolioSnapshot, TargetPosition

_ASSETS = {
    "bitcoin": "bitcoin",
    "btc": "bitcoin",
    "ethereum": "ethereum",
    "eth": "ethereum",
    "solana": "solana",
    "sol": "solana",
    "xrp": "ripple",
    "dogecoin": "dogecoin",
    "doge": "dogecoin",
}
_MONTHS = {
    month.lower(): index + 1
    for index, month in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
    )
}
_STRIKE = r"\$([0-9][0-9,]*(?:\.[0-9]+)?)"
_DATE = r"([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?"
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("touch_up", re.compile(rf"\b(?:reach|hit)\s+{_STRIKE}.*\bby\s+{_DATE}\s*\??\s*$", re.I)),
    ("touch_down", re.compile(rf"\b(?:dip|fall|drop)\s+to\s+{_STRIKE}.*\bby\s+{_DATE}\s*\??\s*$", re.I)),
    ("above", re.compile(rf"\bbe\s+above\s+{_STRIKE}\s+on\s+{_DATE}\s*\??\s*$", re.I)),
    ("below", re.compile(rf"\bbe\s+below\s+{_STRIKE}\s+on\s+{_DATE}\s*\??\s*$", re.I)),
)


def _phi(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def touch_prob(spot: float, barrier: float, sigma: float, t_years: float) -> float:
    if spot <= 0 or barrier <= 0 or sigma <= 0 or t_years <= 0:
        return 0.0
    distance = math.log(barrier / spot)
    drift = -0.5 * sigma * sigma
    if distance == 0:
        return 1.0
    if distance < 0:
        distance, drift = -distance, -drift
    scaled_time = sigma * math.sqrt(t_years)
    probability = _phi((drift * t_years - distance) / scaled_time) + math.exp(
        2.0 * drift * distance / (sigma * sigma)
    ) * _phi((-drift * t_years - distance) / scaled_time)
    return min(1.0, max(0.0, probability))


def terminal_above_prob(
    spot: float, strike: float, sigma: float, t_years: float
) -> float:
    if spot <= 0 or strike <= 0 or sigma <= 0 or t_years <= 0:
        return 0.0
    scaled_time = sigma * math.sqrt(t_years)
    distance = (
        math.log(spot / strike) - 0.5 * sigma * sigma * t_years
    ) / scaled_time
    return min(1.0, max(0.0, _phi(distance)))


def blended_sigma(
    realized: float, anchor: float, t_years: float, tau_years: float
) -> float:
    weight = math.exp(-t_years / max(tau_years, 1e-6))
    return anchor + (realized - anchor) * weight


def parse_threshold(
    question: str, *, default_year: int | None = None
) -> tuple[str, float, datetime] | None:
    for kind, pattern in _PATTERNS:
        match = pattern.search((question or "").strip())
        if match is None:
            continue
        try:
            strike = float(match.group(1).replace(",", ""))
            month = _MONTHS.get(match.group(2).lower())
            if month is None:
                continue
            year = int(match.group(4)) if match.group(4) else (
                default_year or datetime.now(timezone.utc).year
            )
            deadline = datetime(year, month, int(match.group(3)), tzinfo=timezone.utc)
        except ValueError:
            continue
        return kind, strike, deadline
    return None


def detect_asset(question: str) -> str | None:
    lowered = (question or "").lower()
    for keyword, asset in _ASSETS.items():
        if re.search(rf"\b{keyword}\b", lowered):
            return asset
    return None


class VolAnchorRejection(str, Enum):
    INACTIVE = "inactive"
    WRONG_VENUE = "wrong_venue"
    INSUFFICIENT_LIQUIDITY = "insufficient_liquidity"
    INVALID_MARKET_PRICE = "invalid_market_price"
    BLOCKED_CATEGORY = "blocked_category"
    UNSUPPORTED_MARKET = "unsupported_market"
    MISSING_ASSET_STATE = "missing_asset_state"
    EXPIRED = "expired"
    MISSING_ANCHOR = "missing_anchor"
    INSUFFICIENT_EDGE = "insufficient_edge"
    ALREADY_HELD = "already_held"


@dataclass(frozen=True)
class VolAnchorRules:
    min_liquidity: float
    min_edge_pts: float
    stake_usd: float
    tau_years: float
    long_run_vol: dict[str, float]


@dataclass(frozen=True)
class VolAnchorInputs:
    market_id: str
    question: str
    yes_price: float
    no_price: float
    active: bool
    venue: str
    liquidity: float
    category_blocked: bool
    as_of: datetime
    spot: float | None
    realized_vol: float | None
    already_held: bool
    term_sigma: float | None = None


@dataclass(frozen=True)
class VolAnchorProposal:
    market_id: str
    asset: str
    kind: str
    strike: float
    deadline: datetime
    spot: float
    realized_vol: float
    sigma: float
    sigma_source: str
    fair_probability: float
    market_probability: float
    edge_percent: float
    buy_yes: bool
    instrument_id: str
    target_quantity: float
    max_notional: float


@dataclass(frozen=True)
class VolAnchorAssessment:
    proposal: VolAnchorProposal | None
    rejection: VolAnchorRejection | None = None


def assess_vol_anchor(
    inputs: VolAnchorInputs, rules: VolAnchorRules
) -> VolAnchorAssessment:
    """Return the same proposal for research, replay, shadow, and production."""
    if not inputs.active:
        return VolAnchorAssessment(None, VolAnchorRejection.INACTIVE)
    if (inputs.venue or "polymarket") != "polymarket":
        return VolAnchorAssessment(None, VolAnchorRejection.WRONG_VENUE)
    if inputs.liquidity < rules.min_liquidity:
        return VolAnchorAssessment(None, VolAnchorRejection.INSUFFICIENT_LIQUIDITY)
    if not 0.02 <= inputs.yes_price <= 0.98:
        return VolAnchorAssessment(None, VolAnchorRejection.INVALID_MARKET_PRICE)
    if inputs.category_blocked:
        return VolAnchorAssessment(None, VolAnchorRejection.BLOCKED_CATEGORY)
    asset = detect_asset(inputs.question)
    parsed = parse_threshold(inputs.question, default_year=inputs.as_of.year)
    if asset is None or parsed is None:
        return VolAnchorAssessment(None, VolAnchorRejection.UNSUPPORTED_MARKET)
    if inputs.spot is None or inputs.realized_vol is None:
        return VolAnchorAssessment(None, VolAnchorRejection.MISSING_ASSET_STATE)
    kind, strike, deadline = parsed
    t_years = (deadline - inputs.as_of).total_seconds() / (365.0 * 86400.0)
    if t_years <= 0:
        return VolAnchorAssessment(None, VolAnchorRejection.EXPIRED)
    anchor = rules.long_run_vol.get(asset, 0.0)
    if anchor <= 0:
        return VolAnchorAssessment(None, VolAnchorRejection.MISSING_ANCHOR)
    sigma = inputs.term_sigma
    sigma_source = "deribit_iv"
    if sigma is None:
        sigma = blended_sigma(inputs.realized_vol, anchor, t_years, rules.tau_years)
        sigma_source = "blend"
    if kind in ("touch_up", "touch_down"):
        fair = touch_prob(inputs.spot, strike, sigma, t_years)
    elif kind == "above":
        fair = terminal_above_prob(inputs.spot, strike, sigma, t_years)
    else:
        fair = 1.0 - terminal_above_prob(inputs.spot, strike, sigma, t_years)
    edge = abs(fair - inputs.yes_price) * 100.0
    if edge < rules.min_edge_pts:
        return VolAnchorAssessment(None, VolAnchorRejection.INSUFFICIENT_EDGE)
    if inputs.already_held:
        return VolAnchorAssessment(None, VolAnchorRejection.ALREADY_HELD)
    buy_yes = fair > inputs.yes_price
    entry_price = inputs.yes_price if buy_yes else inputs.no_price
    return VolAnchorAssessment(VolAnchorProposal(
        market_id=inputs.market_id,
        asset=asset,
        kind=kind,
        strike=strike,
        deadline=deadline,
        spot=inputs.spot,
        realized_vol=inputs.realized_vol,
        sigma=sigma,
        sigma_source=sigma_source,
        fair_probability=fair,
        market_probability=inputs.yes_price,
        edge_percent=edge,
        buy_yes=buy_yes,
        instrument_id=f"{inputs.market_id}:{'YES' if buy_yes else 'NO'}",
        target_quantity=rules.stake_usd / entry_price,
        max_notional=rules.stake_usd,
    ))


@dataclass(frozen=True)
class VolAnchorExperiment:
    rules: VolAnchorRules

    async def evaluate(
        self, snapshot: MarketSnapshot, portfolio: PortfolioSnapshot
    ) -> list[TargetPosition]:
        details = snapshot.feature("vol_anchor_inputs")
        assessment = assess_vol_anchor(
            VolAnchorInputs(
                market_id=snapshot.market_id,
                question=str(details["question"]),
                yes_price=float(details["yes_price"]),
                no_price=float(details["no_price"]),
                active=bool(details["active"]),
                venue=snapshot.venue,
                liquidity=float(details["liquidity"]),
                category_blocked=bool(details["category_blocked"]),
                as_of=snapshot.observed_at,
                spot=details.get("spot"),
                realized_vol=details.get("realized_vol"),
                already_held=bool(details["already_held"]),
                term_sigma=details.get("term_sigma"),
            ),
            self.rules,
        )
        if assessment.proposal is None:
            return []
        proposal = assessment.proposal
        entry_price = snapshot.price(proposal.instrument_id)
        current = portfolio.quantities().get(proposal.instrument_id, 0.0)
        return [TargetPosition(
            instrument_id=proposal.instrument_id,
            target_quantity=max(current, proposal.target_quantity),
            reference_price=entry_price,
            rationale=(
                f"vol-anchor {proposal.kind}: {proposal.sigma_source} sigma "
                f"{proposal.sigma:.3f}, fair {proposal.fair_probability:.3f}"
            ),
            max_notional=max(
                proposal.max_notional,
                max(current, proposal.target_quantity) * entry_price,
            ),
        )]
