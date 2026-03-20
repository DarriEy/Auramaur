"""Capital allocator — ranks and sizes candidate trades optimally."""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from auramaur.exchange.models import Confidence, Market, Signal
from auramaur.risk.manager import RiskDecision

log = structlog.get_logger()

# Weights used to discount expected value by confidence tier
_CONFIDENCE_WEIGHTS: dict[Confidence, float] = {
    Confidence.HIGH: 1.0,
    Confidence.MEDIUM: 0.75,
    Confidence.LOW: 0.5,
}


@dataclass
class CandidateTrade:
    """A trade that has passed risk checks and is eligible for allocation."""

    market: Market
    signal: Signal
    risk_decision: RiskDecision
    kelly_size: float  # position_size from risk decision (dollars)
    expected_value: float  # edge * confidence_weight
    allocated_size: float = 0.0  # final size after capital allocation


class CapitalAllocator:
    """Allocates capital across a set of approved candidate trades.

    Candidates are ranked by expected value (best opportunities first) and
    greedily allocated until capital or position limits are exhausted.
    """

    def __init__(self, settings):
        self._settings = settings

    def allocate(
        self,
        candidates: list[CandidateTrade],
        available_capital: float,
        current_positions: list,
    ) -> list[CandidateTrade]:
        """Allocate capital optimally across approved candidates.

        1. Sort by ``expected_value`` descending (best opportunities first).
        2. For each candidate:
           - Skip if we already hold a position in this market.
           - Skip if open position count would exceed ``max_open_positions``.
           - Reduce size if it exceeds remaining capital.
           - Track per-category exposure and skip if the cap would be breached.
           - Subtract allocated capital from remaining.
        3. Return the list of candidates that received an allocation (with
           ``allocated_size`` set to their final dollar amounts).
        """
        max_positions = self._settings.risk.max_open_positions
        category_cap_pct = self._settings.risk.category_exposure_cap_pct / 100.0

        # Markets we already hold
        held_market_ids: set[str] = {
            getattr(pos, "market_id", None) for pos in current_positions
        }
        held_market_ids.discard(None)

        # Sort best-first
        ranked = sorted(candidates, key=lambda c: c.expected_value, reverse=True)

        remaining_capital = available_capital
        open_count = len(current_positions)
        category_allocated: dict[str, float] = {}
        allocated: list[CandidateTrade] = []

        for candidate in ranked:
            market_id = candidate.market.id
            category = candidate.market.category

            # Skip if already holding this market
            if market_id in held_market_ids:
                log.debug(
                    "allocator.skip_held",
                    market_id=market_id,
                )
                continue

            # Skip if we've hit the position limit
            if open_count >= max_positions:
                log.info(
                    "allocator.position_limit",
                    open_count=open_count,
                    max_positions=max_positions,
                )
                break

            # Check category exposure cap
            cat_total = category_allocated.get(category, 0.0)
            category_budget = category_cap_pct * available_capital
            if cat_total >= category_budget:
                log.debug(
                    "allocator.category_cap",
                    category=category,
                    cat_total=round(cat_total, 2),
                    category_budget=round(category_budget, 2),
                )
                continue

            # Determine actual size — may be reduced by remaining capital or
            # category budget headroom
            desired = candidate.kelly_size
            cat_headroom = category_budget - cat_total
            size = min(desired, remaining_capital, cat_headroom)

            if size <= 0:
                log.debug(
                    "allocator.no_capital",
                    market_id=market_id,
                    remaining=round(remaining_capital, 2),
                )
                break

            candidate.allocated_size = round(size, 2)
            remaining_capital -= candidate.allocated_size
            category_allocated[category] = cat_total + candidate.allocated_size
            open_count += 1
            allocated.append(candidate)

            log.info(
                "allocator.allocated",
                market_id=market_id,
                category=category,
                size=candidate.allocated_size,
                ev=round(candidate.expected_value, 4),
                remaining_capital=round(remaining_capital, 2),
            )

        log.info(
            "allocator.summary",
            candidates_total=len(candidates),
            allocated_count=len(allocated),
            capital_used=round(available_capital - remaining_capital, 2),
            capital_remaining=round(remaining_capital, 2),
        )

        return allocated

    @staticmethod
    def compute_expected_value(signal: Signal, kelly_size: float) -> float:
        """Compute expected value for ranking.

        EV = (edge_pct / 100) * confidence_weight * kelly_size

        Where confidence_weight is HIGH=1.0, MEDIUM=0.75, LOW=0.5.
        """
        confidence_weight = _CONFIDENCE_WEIGHTS.get(signal.claude_confidence, 0.5)
        edge_frac = abs(signal.edge) / 100.0
        return edge_frac * confidence_weight * kelly_size
