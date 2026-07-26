"""Pure contracts and quote formation for two-sided market-making experiments."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


@dataclass(frozen=True)
class InventoryBounds:
    """Inventory state and hard capacity carried with a quote proposal."""

    net_yes: float
    max_abs_yes: float
    remaining_capacity: float


@dataclass(frozen=True)
class CoupledQuoteProposal:
    """An indivisible YES/NO maker quote; never a directional target."""

    market_id: str
    token_yes_id: str
    token_no_id: str
    bid_price: float
    ask_price: float
    size: float
    spread_bps: int
    inventory: InventoryBounds
    post_only: bool = True

    @property
    def no_leg_price(self) -> float:
        return round(1.0 - self.ask_price, 2)

    @property
    def expected_profit(self) -> float:
        return (self.ask_price - self.bid_price) * self.size


@dataclass(frozen=True)
class QuoteFormation:
    proposal: CoupledQuoteProposal | None
    rejection_reason: str | None = None


class QuoteReconciliation(str, Enum):
    PLACE = "place"
    KEEP = "keep"
    CANCEL_REPLACE = "cancel_replace"


class QuoteShape(Protocol):
    bid_price: float
    ask_price: float
    size: float


def reconcile_quote(
    proposed: QuoteShape,
    *,
    active_bid_price: float | None,
    active_ask_price: float | None,
    active_size: float | None,
    both_legs_pending: bool,
) -> QuoteReconciliation:
    """Choose lifecycle intent while leaving cancellation/execution to production."""
    if active_bid_price is None or active_ask_price is None or active_size is None:
        return QuoteReconciliation.PLACE
    unchanged = (
        abs(active_bid_price - proposed.bid_price) < 1e-9
        and abs(active_ask_price - proposed.ask_price) < 1e-9
        and active_size == proposed.size
    )
    if unchanged and both_legs_pending:
        return QuoteReconciliation.KEEP
    return QuoteReconciliation.CANCEL_REPLACE


def form_coupled_quote(
    *,
    market_id: str,
    token_yes_id: str,
    token_no_id: str,
    best_bid: float | None,
    best_ask: float | None,
    inventory: float,
    quote_size: float,
    max_inventory: float,
    min_spread_bps: int,
    max_spread_bps: int,
) -> QuoteFormation:
    """Form a coupled post-only quote, failing closed on invalid evidence."""
    if (
        not market_id
        or not token_yes_id
        or not token_no_id
        or best_bid is None
        or best_ask is None
        or not all(math.isfinite(v) for v in (best_bid, best_ask, inventory, quote_size, max_inventory))
        or not (0 < best_bid < best_ask < 1)
        or quote_size <= 0
        or max_inventory <= 0
        or min_spread_bps < 0
        or max_spread_bps < min_spread_bps
    ):
        return QuoteFormation(None, "no_bbo" if best_bid is None or best_ask is None else "invalid_input")

    book_spread_bps = int((best_ask - best_bid) * 10000)
    if book_spread_bps > max_spread_bps:
        return QuoteFormation(None, "dead_book")

    raw_bid, raw_ask = best_bid + 0.01, best_ask - 0.01
    if int((raw_ask - raw_bid) * 10000) < min_spread_bps:
        bid, ask = round(best_bid, 2), round(best_ask, 2)
    else:
        bid, ask = round(raw_bid, 2), round(raw_ask, 2)

    if abs(inventory) > 0.01:
        skew = round(inventory / max_inventory * 0.03, 2)
        bid, ask = round(bid - skew, 2), round(ask - skew, 2)
    bid, ask = max(0.01, min(0.99, bid)), max(0.01, min(0.99, ask))
    if bid >= best_ask:
        bid = round(best_ask - 0.01, 2)
    if ask <= best_bid:
        ask = round(best_bid + 0.01, 2)
    if bid >= ask:
        return QuoteFormation(None, "bid_gte_ask")

    spread_bps = int((ask - bid) * 10000)
    if spread_bps < min_spread_bps:
        return QuoteFormation(None, "spread_too_narrow")
    no_price = round(1.0 - ask, 2)
    if not 0.01 <= no_price <= 0.99:
        return QuoteFormation(None, "invalid_no_price")

    remaining = max_inventory - abs(inventory)
    size = min(quote_size, remaining)
    if size < 1.0:
        return QuoteFormation(None, "inventory_capacity")
    min_size = max(1.0 / bid, 1.0 / no_price)
    if size * bid < 1.0 or size * no_price < 1.0:
        bumped = math.ceil(min_size * 100) / 100
        if bumped > remaining:
            return QuoteFormation(None, "min_notional")
        size = bumped

    proposal = CoupledQuoteProposal(
        market_id=market_id,
        token_yes_id=token_yes_id,
        token_no_id=token_no_id,
        bid_price=bid,
        ask_price=ask,
        size=round(size, 2),
        spread_bps=spread_bps,
        inventory=InventoryBounds(inventory, max_inventory, remaining),
    )
    return QuoteFormation(proposal)
