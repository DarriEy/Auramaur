"""Pure proposal assessment for the interim-manager strategy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from math import isfinite


class InterimManagerRejection(str, Enum):
    EXPIRED = "expired"
    THESIS_TOO_SHORT = "thesis_too_short"
    DELEGATED = "delegated"
    SUNSET_REACHED = "sunset_reached"
    POSITION_CAP = "position_cap"
    ENTRY_LIMIT = "entry_limit"
    INSUFFICIENT_EDGE = "insufficient_edge"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class InterimManagerRules:
    proposal_ttl_hours: float
    max_open_positions: int
    min_robust_edge: float
    default_uncertainty_buffer: float
    slippage_buffer: float
    liquidity_penalty: float
    thin_liquidity_usd: float
    correlation_penalty_per_position: float
    stake_usd: float
    paper: bool
    min_thesis_chars: int = 40


@dataclass(frozen=True)
class InterimManagerInputs:
    proposal_id: int
    market_id: str
    thesis: str
    side: str
    fair_probability: float
    requested_stake_usd: float
    market_yes_price: float
    market_liquidity: float
    category: str
    created_at: datetime | None
    sunset_at: datetime | None
    as_of: datetime
    delegated_to: str | None
    open_positions: int
    open_positions_in_category: int
    fee_rate: float
    confidence_lo: float | None = None
    confidence_hi: float | None = None
    max_entry_price: float | None = None


@dataclass(frozen=True)
class InterimManagerProposal:
    proposal_id: int
    market_id: str
    buy_yes: bool
    fair_probability: float
    market_probability: float
    entry_price: float
    robust_edge: float
    robust_edge_detail: str
    requested_stake_usd: float
    force_paper: bool
    thesis: str


@dataclass(frozen=True)
class InterimManagerAssessment:
    proposal: InterimManagerProposal | None
    rejection: InterimManagerRejection | None = None
    reason: str = ""
    retryable: bool = False
    robust_edge: float | None = None
    decision_price: float | None = None


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def assess_interim_manager(
    inputs: InterimManagerInputs,
    rules: InterimManagerRules,
) -> InterimManagerAssessment:
    """Apply the charter's deterministic gates and construct a proposal."""
    now = _aware(inputs.as_of)
    created = _aware(inputs.created_at)
    sunset = _aware(inputs.sunset_at)
    numeric = (
        inputs.fair_probability,
        inputs.requested_stake_usd,
        inputs.market_yes_price,
        inputs.market_liquidity,
        inputs.fee_rate,
    )
    if now is None or not all(isfinite(value) for value in numeric):
        return InterimManagerAssessment(
            None, InterimManagerRejection.INVALID_INPUT, "invalid numeric input"
        )
    if not 0.0 < inputs.market_yes_price < 1.0:
        return InterimManagerAssessment(
            None, InterimManagerRejection.INVALID_INPUT, "invalid market price"
        )
    if created is not None and (now - created).total_seconds() > (
        rules.proposal_ttl_hours * 3600.0
    ):
        return InterimManagerAssessment(
            None, InterimManagerRejection.EXPIRED, "proposal ttl elapsed"
        )
    thesis = inputs.thesis.strip()
    if len(thesis) < rules.min_thesis_chars:
        return InterimManagerAssessment(
            None,
            InterimManagerRejection.THESIS_TOO_SHORT,
            "charter: no nameable mispricing mechanism",
        )
    if inputs.delegated_to is not None:
        return InterimManagerAssessment(
            None,
            InterimManagerRejection.DELEGATED,
            f"delegated to {inputs.delegated_to}",
        )
    if sunset is not None and now >= sunset:
        return InterimManagerAssessment(
            None, InterimManagerRejection.SUNSET_REACHED, "thesis sunset reached"
        )
    if inputs.open_positions >= rules.max_open_positions:
        return InterimManagerAssessment(
            None,
            InterimManagerRejection.POSITION_CAP,
            "position cap reached",
            retryable=True,
        )

    buy_yes = inputs.side.upper() == "BUY"
    entry_price = (
        inputs.market_yes_price if buy_yes else 1.0 - inputs.market_yes_price
    )
    if (
        inputs.max_entry_price is not None
        and entry_price > inputs.max_entry_price + 1e-9
    ):
        return InterimManagerAssessment(
            None,
            InterimManagerRejection.ENTRY_LIMIT,
            f"entry {entry_price:.3f} above limit {inputs.max_entry_price:.3f}",
        )

    gross = (
        inputs.fair_probability - inputs.market_yes_price
        if buy_yes
        else inputs.market_yes_price - inputs.fair_probability
    )
    if (
        inputs.confidence_lo is not None
        and inputs.confidence_hi is not None
        and inputs.confidence_hi >= inputs.confidence_lo
    ):
        uncertainty = (inputs.confidence_hi - inputs.confidence_lo) / 2.0
    else:
        uncertainty = rules.default_uncertainty_buffer
    fee = (
        inputs.fee_rate
        * inputs.market_yes_price
        * (1.0 - inputs.market_yes_price)
    )
    liquidity = (
        rules.liquidity_penalty
        if inputs.market_liquidity < rules.thin_liquidity_usd
        else 0.0
    )
    correlation = (
        rules.correlation_penalty_per_position * inputs.open_positions_in_category
    )
    robust = (
        gross
        - uncertainty
        - fee
        - rules.slippage_buffer
        - liquidity
        - correlation
    )
    detail = (
        f"gross {gross:+.3f} − unc {uncertainty:.3f} − fee {fee:.3f} "
        f"− slip {rules.slippage_buffer:.3f} − liq {liquidity:.3f} "
        f"− corr {correlation:.3f}"
    )
    if robust < rules.min_robust_edge:
        return InterimManagerAssessment(
            None,
            InterimManagerRejection.INSUFFICIENT_EDGE,
            f"robust edge {robust:+.3f} < {rules.min_robust_edge:.3f} ({detail})",
            robust_edge=robust,
            decision_price=entry_price,
        )
    return InterimManagerAssessment(
        InterimManagerProposal(
            proposal_id=inputs.proposal_id,
            market_id=inputs.market_id,
            buy_yes=buy_yes,
            fair_probability=max(0.01, min(0.99, inputs.fair_probability)),
            market_probability=inputs.market_yes_price,
            entry_price=entry_price,
            robust_edge=robust,
            robust_edge_detail=detail,
            requested_stake_usd=min(rules.stake_usd, inputs.requested_stake_usd),
            force_paper=rules.paper,
            thesis=thesis,
        )
    )
