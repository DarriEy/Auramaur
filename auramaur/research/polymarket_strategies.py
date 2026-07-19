"""Paper-only, mechanism-first Polymarket strategy evaluators.

These functions produce research observations, never orders.  Execution must
remain behind the normal strategy protocol, risk manager, and graduation gate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


@dataclass(frozen=True)
class OutcomeQuote:
    market_id: str
    outcome: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    taker_fee_per_share: float = 0.0


@dataclass(frozen=True)
class ResearchOpportunity:
    strategy_source: str
    market_id: str
    mechanism: str
    expected_edge: float
    capacity_usd: float
    score: float
    details: dict


def complete_set_arbitrage(
    event_id: str,
    outcomes: Iterable[OutcomeQuote],
    *,
    safety_buffer: float = 0.01,
) -> list[ResearchOpportunity]:
    """Evaluate executable buy-all and sell-all bounds for a full partition."""
    quotes = list(outcomes)
    if len(quotes) < 2:
        return []
    out: list[ResearchOpportunity] = []
    buy_cost = sum(q.ask + q.taker_fee_per_share for q in quotes)
    buy_capacity = min(q.ask_size for q in quotes)
    buy_edge = 1.0 - buy_cost - safety_buffer
    if buy_edge > 0 and buy_capacity > 0:
        out.append(ResearchOpportunity(
            "complete_set_arb", event_id, "buy_all_below_one", buy_edge,
            buy_capacity * buy_cost, min(1.0, buy_edge / 0.05),
            {"outcomes": len(quotes), "executable_cost": buy_cost},
        ))
    sell_proceeds = sum(q.bid - q.taker_fee_per_share for q in quotes)
    sell_capacity = min(q.bid_size for q in quotes)
    sell_edge = sell_proceeds - 1.0 - safety_buffer
    if sell_edge > 0 and sell_capacity > 0:
        out.append(ResearchOpportunity(
            "complete_set_arb", event_id, "sell_all_above_one", sell_edge,
            sell_capacity * sell_proceeds, min(1.0, sell_edge / 0.05),
            {"outcomes": len(quotes), "executable_proceeds": sell_proceeds},
        ))
    return out


class Relation(str, Enum):
    A_IMPLIES_B = "a_implies_b"
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"
    PARTITION = "partition"


def logical_constraint_edge(
    relationship_id: str,
    relation: Relation,
    probability_a: float,
    probability_b: float,
    *,
    verification_confidence: float,
    cost_buffer: float = 0.02,
) -> ResearchOpportunity | None:
    """Price deterministic probability bounds after semantic verification."""
    if verification_confidence < 0.95:
        return None
    if relation is Relation.A_IMPLIES_B:
        violation = probability_a - probability_b
    elif relation is Relation.MUTUALLY_EXCLUSIVE:
        violation = probability_a + probability_b - 1.0
    else:
        violation = abs(probability_a + probability_b - 1.0)
    edge = violation - cost_buffer
    if edge <= 0:
        return None
    return ResearchOpportunity(
        "constraint_arb", relationship_id, relation.value, edge, 0.0,
        min(1.0, edge / 0.10) * verification_confidence,
        {"p_a": probability_a, "p_b": probability_b,
         "verification_confidence": verification_confidence},
    )


def source_latency_edge(
    market_id: str,
    *,
    source_age_seconds: float,
    observed_market_move: float,
    expected_move: float,
    source_is_authoritative: bool,
    max_source_age_seconds: float = 120.0,
    cost_buffer: float = 0.02,
) -> ResearchOpportunity | None:
    """Admit only fresh, authoritative source changes not yet reflected in price."""
    if not source_is_authoritative or source_age_seconds < 0:
        return None
    if source_age_seconds > max_source_age_seconds:
        return None
    residual = abs(expected_move) - abs(observed_market_move) - cost_buffer
    if residual <= 0 or expected_move * observed_market_move < 0:
        return None
    freshness = 1.0 - source_age_seconds / max_source_age_seconds
    return ResearchOpportunity(
        "source_latency", market_id, "official_source_changed_first", residual,
        0.0, max(0.0, min(1.0, freshness * residual / 0.10)),
        {"source_age_seconds": source_age_seconds,
         "observed_market_move": observed_market_move,
         "expected_move": expected_move},
    )


def normal_bin_probability(
    mean: float, standard_deviation: float, lower: float | None, upper: float | None,
) -> float:
    """Scheduled-release nowcast probability for an interval under a normal model."""
    if standard_deviation <= 0:
        raise ValueError("standard_deviation must be positive")
    def cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(
            (x - mean) / (standard_deviation * math.sqrt(2))))
    lo = 0.0 if lower is None else cdf(lower)
    hi = 1.0 if upper is None else cdf(upper)
    return max(0.0, min(1.0, hi - lo))


def maker_quote_economics(
    market_id: str,
    *,
    spread_capture: float,
    fill_probability: float,
    expected_rebate: float,
    adverse_selection: float,
    inventory_cost: float,
    cancel_failure_cost: float,
) -> ResearchOpportunity | None:
    """Expected maker value including rebates and stale-quote tail costs."""
    gross = fill_probability * (spread_capture + expected_rebate)
    edge = gross - adverse_selection - inventory_cost - cancel_failure_cost
    if edge <= 0:
        return None
    return ResearchOpportunity(
        "maker_rebate", market_id, "rebate_adjusted_passive_quote", edge, 0.0,
        min(1.0, edge / 0.02),
        {"fill_probability": fill_probability, "gross": gross,
         "adverse_selection": adverse_selection,
         "inventory_cost": inventory_cost,
         "cancel_failure_cost": cancel_failure_cost},
    )


def flow_confirmation_multiplier(
    *, base_signal_edge: float, signed_flow_z: float, same_direction: bool,
) -> float:
    """Use flow only as a bounded filter; it can never originate a trade."""
    if base_signal_edge <= 0:
        return 0.0
    strength = min(1.0, abs(signed_flow_z) / 4.0)
    return 1.0 + 0.25 * strength if same_direction else 1.0 - 0.75 * strength


class ResearchRecorder:
    """Persist immutable paper evaluations for later walk-forward scoring."""

    def __init__(self, db) -> None:
        self.db = db

    async def record(self, opportunity: ResearchOpportunity) -> None:
        await self.db.execute(
            """INSERT INTO strategy_evaluations
               (strategy_source, market_id, mechanism, score, expected_edge,
                payload_json, is_paper) VALUES (?, ?, ?, ?, ?, ?, 1)""",
            (opportunity.strategy_source, opportunity.market_id,
             opportunity.mechanism, opportunity.score, opportunity.expected_edge,
             json.dumps({**opportunity.details,
                         "capacity_usd": opportunity.capacity_usd}, sort_keys=True)),
        )
        await self.db.commit()


class DecisionTracker:
    """Capture decision-time executable prices and subsequent CLV marks."""

    def __init__(self, db) -> None:
        self.db = db

    async def capture(
        self, *, market_id: str, strategy_source: str, signal_id: int | None,
        side: str, fair_probability: float, reference_price: float,
        executable_price: float | None, best_bid: float | None,
        best_ask: float | None, requested_size: float, fee_estimate: float,
        filled: bool = False,
    ) -> None:
        await self.db.execute(
            """INSERT OR IGNORE INTO decision_snapshots
               (market_id, strategy_source, signal_id, side, fair_probability,
                reference_price, executable_price, best_bid, best_ask,
                requested_size, fee_estimate, filled)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (market_id, strategy_source, signal_id, side, fair_probability,
             reference_price, executable_price, best_bid, best_ask,
             requested_size, fee_estimate, int(filled)),
        )
        # Deliberately durable at the decision boundary. This path is currently
        # low-volume; if capture moves into a high-frequency scanner, batch it
        # behind a non-blocking writer rather than adding latency here.
        await self.db.commit()

    async def mark(self, decision_id: int, horizon_seconds: int, *,
                   bid: float | None, ask: float | None) -> None:
        mid = None if bid is None or ask is None else (bid + ask) / 2
        await self.db.execute(
            """INSERT OR REPLACE INTO decision_marks
               (decision_id, horizon_seconds, bid, ask, mid)
               VALUES (?, ?, ?, ?, ?)""",
            (decision_id, horizon_seconds, bid, ask, mid),
        )
        await self.db.commit()
