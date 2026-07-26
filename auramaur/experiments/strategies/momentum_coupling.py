"""Pure spot-to-prediction momentum-coupling decisions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from auramaur.experiments.models import MarketSnapshot, PortfolioSnapshot, TargetPosition


class CouplingRejection(str, Enum):
    MOVE_TOO_SMALL = "move_too_small"
    STRIKE_MISSING = "strike_missing"
    NOT_NEAR_MONEY = "not_near_money"
    PINNED_MARKET = "pinned_market"


@dataclass(frozen=True)
class CouplingRules:
    move_threshold_pct: float
    near_money_pct: float
    max_position_usd: float


@dataclass(frozen=True)
class CouplingCandidate:
    market_id: str
    question: str
    yes_price: float


@dataclass(frozen=True)
class CouplingInputs:
    asset: str
    spot: float
    move_pct: float
    candidates: tuple[CouplingCandidate, ...]


@dataclass(frozen=True)
class CouplingProposal:
    market_id: str
    question: str
    asset: str
    spot: float
    move_pct: float
    strike: float
    market_probability: float
    fair_probability: float
    edge_percent: float
    direction: str
    target_quantity: float
    max_notional: float
    confidence: str


@dataclass(frozen=True)
class CouplingAssessment:
    proposal: CouplingProposal | None
    rejection: CouplingRejection | None = None


def extract_strike(question: str) -> float | None:
    match = re.search(r"\$?([\d,]{4,})", question)
    if not match:
        return None
    strike = float(match.group(1).replace(",", ""))
    return strike if strike > 0 else None


def assess_momentum_coupling(
    inputs: CouplingInputs,
    rules: CouplingRules,
) -> CouplingAssessment:
    """Choose the first eligible market and form its complete proposal."""
    if abs(inputs.move_pct) < rules.move_threshold_pct:
        return CouplingAssessment(None, CouplingRejection.MOVE_TOO_SMALL)
    last_rejection = CouplingRejection.STRIKE_MISSING
    for candidate in inputs.candidates:
        strike = extract_strike(candidate.question)
        if strike is None:
            last_rejection = CouplingRejection.STRIKE_MISSING
            continue
        if abs(strike - inputs.spot) / inputs.spot > rules.near_money_pct:
            last_rejection = CouplingRejection.NOT_NEAR_MONEY
            continue
        if not 0.10 < candidate.yes_price < 0.90:
            last_rejection = CouplingRejection.PINNED_MARKET
            continue
        direction = "BUY_YES" if inputs.move_pct > 0 else "BUY_NO"
        signed_shift = (1 if inputs.move_pct > 0 else -1) * min(
            abs(inputs.move_pct) / 100.0 * 5, 0.15
        )
        fair = max(0.02, min(0.98, candidate.yes_price + signed_shift))
        execution_price = (
            candidate.yes_price if direction == "BUY_YES" else 1.0 - candidate.yes_price
        )
        return CouplingAssessment(CouplingProposal(
            market_id=candidate.market_id,
            question=candidate.question,
            asset=inputs.asset,
            spot=inputs.spot,
            move_pct=inputs.move_pct,
            strike=strike,
            market_probability=candidate.yes_price,
            fair_probability=fair,
            edge_percent=abs(fair - candidate.yes_price) * 100.0,
            direction=direction,
            target_quantity=rules.max_position_usd / execution_price,
            max_notional=rules.max_position_usd,
            confidence="HIGH" if abs(inputs.move_pct) >= 1.0 else "MEDIUM",
        ))
    return CouplingAssessment(None, last_rejection)


@dataclass(frozen=True)
class MomentumCouplingExperiment:
    rules: CouplingRules

    async def evaluate(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioSnapshot,
    ) -> list[TargetPosition]:
        del portfolio
        details = snapshot.feature("momentum_coupling_inputs")
        result = assess_momentum_coupling(CouplingInputs(
            asset=str(details["asset"]),
            spot=float(details["spot"]),
            move_pct=float(details["move_pct"]),
            candidates=tuple(CouplingCandidate(
                market_id=str(item["market_id"]),
                question=str(item["question"]),
                yes_price=float(item["yes_price"]),
            ) for item in details["candidates"]),
        ), self.rules)
        if result.proposal is None:
            return []
        proposal = result.proposal
        token = "YES" if proposal.direction == "BUY_YES" else "NO"
        price = proposal.market_probability if token == "YES" else 1 - proposal.market_probability
        return [TargetPosition(
            instrument_id=f"{proposal.market_id}:{token}",
            target_quantity=proposal.target_quantity,
            reference_price=price,
            rationale=f"{proposal.asset} momentum {proposal.move_pct:+.3f}%",
            max_notional=proposal.max_notional,
        )]
