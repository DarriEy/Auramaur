"""Pure proposal formation for the model-driven agent trader."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum

from auramaur.experiments.models import MarketSnapshot, PortfolioSnapshot, TargetPosition


class AgentTraderRejection(str, Enum):
    INVALID_INPUT = "invalid_input"
    INSUFFICIENT_EDGE = "insufficient_edge"


@dataclass(frozen=True)
class AgentTraderRules:
    min_edge_pts: float
    stake_usd: float
    source_tag: str


@dataclass(frozen=True)
class AgentTraderInputs:
    market_id: str
    question: str
    market_yes: float
    prob_yes: float
    thesis: str


@dataclass(frozen=True)
class AgentTraderProposal:
    market_id: str
    question: str
    buy_yes: bool
    instrument_id: str
    probability_yes: float
    market_probability: float
    edge_percent: float
    entry_price: float
    target_quantity: float
    max_notional: float
    thesis: str
    source_tag: str


@dataclass(frozen=True)
class AgentTraderAssessment:
    proposal: AgentTraderProposal | None
    rejection: AgentTraderRejection | None = None


def parse_decisions(raw: str, max_entries: int) -> list[dict]:
    """Parse model JSON defensively; malformed opinions fail closed."""
    if max_entries <= 0:
        return []
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        payload = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        return []
    result: list[dict] = []
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        market_id = str(decision.get("market_id", "")).strip()
        thesis = str(decision.get("thesis", "")).strip()
        try:
            probability = float(decision.get("prob_yes"))
        except (TypeError, ValueError):
            continue
        if not market_id or not thesis or not math.isfinite(probability):
            continue
        if not 0.0 < probability < 1.0:
            continue
        result.append({
            "market_id": market_id,
            "prob_yes": probability,
            "thesis": thesis,
        })
        if len(result) >= max_entries:
            break
    return result


def assess_agent_trader(
    inputs: AgentTraderInputs, rules: AgentTraderRules
) -> AgentTraderAssessment:
    values = (inputs.market_yes, inputs.prob_yes, rules.min_edge_pts, rules.stake_usd)
    if (
        not inputs.market_id
        or not inputs.question
        or not inputs.thesis.strip()
        or not rules.source_tag
        or any(not math.isfinite(value) for value in values)
        or not 0.0 < inputs.market_yes < 1.0
        or not 0.0 < inputs.prob_yes < 1.0
        or rules.min_edge_pts < 0.0
        or rules.stake_usd <= 0.0
    ):
        return AgentTraderAssessment(None, AgentTraderRejection.INVALID_INPUT)
    edge_pts = abs(inputs.prob_yes - inputs.market_yes) * 100.0
    if edge_pts < rules.min_edge_pts:
        return AgentTraderAssessment(None, AgentTraderRejection.INSUFFICIENT_EDGE)
    buy_yes = inputs.prob_yes > inputs.market_yes
    entry_price = inputs.market_yes if buy_yes else 1.0 - inputs.market_yes
    return AgentTraderAssessment(AgentTraderProposal(
        market_id=inputs.market_id,
        question=inputs.question,
        buy_yes=buy_yes,
        instrument_id=f"{inputs.market_id}:{'YES' if buy_yes else 'NO'}",
        probability_yes=inputs.prob_yes,
        market_probability=inputs.market_yes,
        edge_percent=edge_pts,
        entry_price=entry_price,
        target_quantity=rules.stake_usd / entry_price,
        max_notional=rules.stake_usd,
        thesis=inputs.thesis[:500],
        source_tag=rules.source_tag,
    ))


@dataclass(frozen=True)
class AgentTraderExperiment:
    rules: AgentTraderRules

    async def evaluate(
        self, snapshot: MarketSnapshot, portfolio: PortfolioSnapshot
    ) -> list[TargetPosition]:
        del portfolio
        details = snapshot.feature("agent_trader_opinion")
        assessment = assess_agent_trader(AgentTraderInputs(
            market_id=snapshot.market_id,
            question=str(details["question"]),
            market_yes=snapshot.price("YES"),
            prob_yes=float(details["prob_yes"]),
            thesis=str(details["thesis"]),
        ), self.rules)
        if assessment.proposal is None:
            return []
        proposal = assessment.proposal
        return [TargetPosition(
            instrument_id=proposal.instrument_id,
            target_quantity=proposal.target_quantity,
            reference_price=proposal.entry_price,
            rationale=proposal.thesis,
            max_notional=proposal.max_notional,
        )]
