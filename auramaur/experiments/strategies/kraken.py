"""Pure proposal contract for Kraken long-only spot experiments.

The contract deliberately describes a spot order candidate, not a binary-market
``TargetPosition``.  Market-data acquisition, balances, persistence, risk
scaling, paper/live gates, and order submission remain production concerns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class KrakenAction(str, Enum):
    BUY = "buy"
    SELL = "sell"


class KrakenRejection(str, Enum):
    INVALID_INPUT = "invalid_input"
    ORPHAN_REENTRY = "orphan_reentry"
    COOLDOWN = "cooldown"
    SIGNAL_GATE = "signal_gate"
    NO_EXIT = "no_exit"
    BUDGET_FULL = "budget_full"
    INVALID_VOLUME = "invalid_volume"
    BELOW_ORDER_MINIMUM = "below_order_minimum"
    MINIMUM_TOO_EXPENSIVE = "minimum_too_expensive"
    UNFUNDED = "unfunded"


@dataclass(frozen=True)
class KrakenSpotRules:
    max_order_usd: float
    lot_decimals: int
    order_minimum: float = 0.0
    funding_fee_buffer: float = 0.005
    minimum_notional_multiplier: float = 1.5


@dataclass(frozen=True)
class KrakenSpotInputs:
    pair: str
    price: float
    holding: bool
    orphaned: bool
    in_cooldown: bool = False
    signal_accepted: bool = False
    exit_reason: str | None = None
    held_quantity: float = 0.0
    sized_entry_volume: float = 0.0
    allocated_usd: float = 0.0
    budget_usd: float = 0.0
    free_quote: float | None = None
    minimum_notional_usd: float | None = None
    paper: bool = True


@dataclass(frozen=True)
class KrakenSpotOrderProposal:
    pair: str
    action: KrakenAction
    volume: float
    reference_price: float
    estimated_quote_notional: float
    rationale: str


@dataclass(frozen=True)
class KrakenSpotAssessment:
    proposal: KrakenSpotOrderProposal | None
    rejection: KrakenRejection | None = None


def _floor_lot(quantity: float, decimals: int) -> float:
    scale = 10 ** decimals
    return math.floor(quantity * scale) / scale


def assess_kraken_spot(
    inputs: KrakenSpotInputs, rules: KrakenSpotRules
) -> KrakenSpotAssessment:
    """Return one deterministic order candidate, failing closed on bad data."""
    numbers = (
        inputs.price, inputs.held_quantity, inputs.sized_entry_volume,
        inputs.allocated_usd, inputs.budget_usd, rules.max_order_usd,
        rules.order_minimum, rules.funding_fee_buffer,
    )
    if (
        not inputs.pair or inputs.price <= 0 or rules.max_order_usd <= 0
        or rules.lot_decimals < 0 or any(not math.isfinite(value) for value in numbers)
        or any(value < 0 for value in numbers[1:])
    ):
        return KrakenSpotAssessment(None, KrakenRejection.INVALID_INPUT)

    if inputs.holding:
        if not inputs.exit_reason:
            return KrakenSpotAssessment(None, KrakenRejection.NO_EXIT)
        volume = _floor_lot(inputs.held_quantity, rules.lot_decimals)
        if volume <= 0:
            return KrakenSpotAssessment(None, KrakenRejection.INVALID_VOLUME)
        if rules.order_minimum and volume < rules.order_minimum:
            return KrakenSpotAssessment(None, KrakenRejection.BELOW_ORDER_MINIMUM)
        return KrakenSpotAssessment(KrakenSpotOrderProposal(
            pair=inputs.pair, action=KrakenAction.SELL, volume=volume,
            reference_price=inputs.price,
            estimated_quote_notional=volume * inputs.price,
            rationale=inputs.exit_reason,
        ))

    if inputs.orphaned:
        return KrakenSpotAssessment(None, KrakenRejection.ORPHAN_REENTRY)
    if inputs.in_cooldown:
        return KrakenSpotAssessment(None, KrakenRejection.COOLDOWN)
    if not inputs.signal_accepted:
        return KrakenSpotAssessment(None, KrakenRejection.SIGNAL_GATE)
    if inputs.allocated_usd + rules.max_order_usd > inputs.budget_usd:
        return KrakenSpotAssessment(None, KrakenRejection.BUDGET_FULL)

    volume = _floor_lot(inputs.sized_entry_volume, rules.lot_decimals)
    if rules.order_minimum and volume < rules.order_minimum:
        minimum_notional = (inputs.minimum_notional_usd
                            if inputs.minimum_notional_usd is not None
                            else rules.order_minimum * inputs.price)
        if not math.isfinite(minimum_notional) or minimum_notional < 0:
            return KrakenSpotAssessment(None, KrakenRejection.INVALID_INPUT)
        if minimum_notional > rules.max_order_usd * rules.minimum_notional_multiplier:
            return KrakenSpotAssessment(None, KrakenRejection.MINIMUM_TOO_EXPENSIVE)
        volume = rules.order_minimum
    if volume <= 0:
        return KrakenSpotAssessment(None, KrakenRejection.INVALID_VOLUME)
    notional = volume * inputs.price
    if not inputs.paper:
        if inputs.free_quote is None or not math.isfinite(inputs.free_quote):
            return KrakenSpotAssessment(None, KrakenRejection.INVALID_INPUT)
        if inputs.free_quote < notional * (1.0 + rules.funding_fee_buffer):
            return KrakenSpotAssessment(None, KrakenRejection.UNFUNDED)
    return KrakenSpotAssessment(KrakenSpotOrderProposal(
        pair=inputs.pair, action=KrakenAction.BUY, volume=volume,
        reference_price=inputs.price, estimated_quote_notional=notional,
        rationale="directional_signal",
    ))
