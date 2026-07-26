"""Pure platform-consensus decisions shared by research and production."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from auramaur.experiments.models import MarketSnapshot, PortfolioSnapshot, TargetPosition


_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "will", "be", "been",
    "to", "of", "in", "for", "on", "at", "by", "or", "and", "it", "its",
    "this", "that", "with", "from", "as", "do", "does", "did", "not", "no",
    "yes", "has", "have", "had", "can", "could", "would", "should", "may",
    "if", "but", "so", "than", "then", "what", "which", "who", "whom",
    "how", "when", "where", "why", "before", "after", "during", "between",
    "next",
})


def _word_overlap_score(text_a: str, text_b: str) -> float:
    """Match questions without importing the production strategy graph."""
    words_a = set(re.findall(r"[a-z0-9]+", text_a.lower())) - _STOP_WORDS
    words_b = set(re.findall(r"[a-z0-9]+", text_b.lower())) - _STOP_WORDS
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / min(len(words_a), len(words_b))


@dataclass(frozen=True)
class ConsensusRules:
    min_liquidity: float
    min_hours_to_resolution: float
    max_days_to_resolution: float
    match_threshold: float
    min_edge: float
    min_manifold_bettors: int
    min_manifold_liquidity: float
    min_metaculus_forecasters: int
    stake_usd: float


@dataclass(frozen=True)
class ConsensusForecast:
    title: str
    content: str
    source: str


@dataclass(frozen=True)
class ConsensusInputs:
    market_id: str
    question: str
    market_probability: float
    active: bool
    liquidity_or_volume: float
    category_blocked: bool
    end_at: datetime | None
    as_of: datetime
    fee: float
    forecasts: tuple[ConsensusForecast, ...]


@dataclass(frozen=True)
class ConsensusProposal:
    market_id: str
    instrument_id: str
    consensus_source: str
    consensus_probability: float
    market_probability: float
    overlap: float
    buy_yes: bool
    confidence: str
    edge_percent: float
    reference_price: float
    target_quantity: float
    max_notional: float
    evidence_summary: str


def clean_title(title: str) -> str:
    return re.sub(r"^\[(?:Manifold|Metaculus):\s*\d+%\]\s*", "", title).strip()


def parse_probability(title: str) -> float | None:
    match = re.match(r"^\[(?:Manifold|Metaculus):\s*(\d+)%\]", title)
    if match is None:
        return None
    value = float(match.group(1)) / 100.0
    return value if 0.0 <= value <= 1.0 else None


def forecast_quality_ok(forecast: ConsensusForecast, rules: ConsensusRules) -> bool:
    content = forecast.content or ""
    if forecast.source == "Manifold":
        bettors = re.search(r"Unique bettors:\s*([\d,]+)", content)
        liquidity = re.search(r"Liquidity:\s*\$([\d,]+(?:\.\d+)?)", content)
        return bool(
            bettors
            and liquidity
            and int(bettors.group(1).replace(",", "")) >= rules.min_manifold_bettors
            and float(liquidity.group(1).replace(",", "")) >= rules.min_manifold_liquidity
        )
    if forecast.source != "Metaculus":
        return False
    forecasters = re.search(r"Forecasters:\s*([\d,]+)", content)
    return bool(
        forecasters
        and int(forecasters.group(1).replace(",", ""))
        >= rules.min_metaculus_forecasters
    )


def assess_platform_consensus(
    inputs: ConsensusInputs, rules: ConsensusRules
) -> ConsensusProposal | None:
    """Return a proposal without fetching, persistence, risk, or execution."""
    if not inputs.active or inputs.liquidity_or_volume < rules.min_liquidity:
        return None
    if inputs.category_blocked or inputs.end_at is None:
        return None
    end_at = inputs.end_at
    if end_at.tzinfo is None or end_at.utcoffset() is None:
        end_at = end_at.replace(tzinfo=timezone.utc)
    hours_left = (end_at - inputs.as_of).total_seconds() / 3600.0
    if not rules.min_hours_to_resolution <= hours_left <= rules.max_days_to_resolution * 24:
        return None
    best: ConsensusForecast | None = None
    best_overlap = 0.0
    for forecast in inputs.forecasts:
        if not forecast_quality_ok(forecast, rules):
            continue
        overlap = _word_overlap_score(inputs.question, clean_title(forecast.title))
        if overlap > best_overlap:
            best, best_overlap = forecast, overlap
    if best is None or best_overlap < rules.match_threshold:
        return None
    probability = parse_probability(best.title)
    if probability is None:
        return None
    edge = probability - inputs.market_probability
    if abs(edge) < rules.min_edge + inputs.fee:
        return None
    buy_yes = edge > 0
    title = clean_title(best.title)
    evidence = (
        f"Platform consensus follower: found matching market on {best.source} "
        f"('{title[:80]}') with consensus probability {probability:.0%}. "
        f"Market price is {inputs.market_probability:.0%}. Edge: {abs(edge):.1%}. "
        f"Word overlap: {best_overlap:.2f}."
    )
    reference = inputs.market_probability if buy_yes else 1 - inputs.market_probability
    return ConsensusProposal(
        market_id=inputs.market_id,
        instrument_id=f"{inputs.market_id}:{'YES' if buy_yes else 'NO'}",
        consensus_source=best.source,
        consensus_probability=probability,
        market_probability=inputs.market_probability,
        overlap=best_overlap,
        buy_yes=buy_yes,
        confidence="high" if best_overlap >= 0.8 else "medium",
        edge_percent=abs(edge) * 100,
        reference_price=reference,
        target_quantity=rules.stake_usd / reference,
        max_notional=rules.stake_usd,
        evidence_summary=evidence,
    )


@dataclass(frozen=True)
class PlatformConsensusExperiment:
    rules: ConsensusRules

    async def evaluate(
        self, snapshot: MarketSnapshot, portfolio: PortfolioSnapshot
    ) -> list[TargetPosition]:
        del portfolio
        data = snapshot.feature("platform_consensus_inputs")
        proposal = assess_platform_consensus(ConsensusInputs(
            market_id=snapshot.market_id,
            question=str(data["question"]),
            market_probability=float(data["market_probability"]),
            active=bool(data["active"]),
            liquidity_or_volume=float(data["liquidity_or_volume"]),
            category_blocked=bool(data["category_blocked"]),
            end_at=datetime.fromisoformat(data["end_at"]) if data["end_at"] else None,
            as_of=snapshot.observed_at,
            fee=float(data["fee"]),
            forecasts=tuple(ConsensusForecast(**item) for item in data["forecasts"]),
        ), self.rules)
        if proposal is None:
            return []
        return [TargetPosition(
            instrument_id=proposal.instrument_id,
            target_quantity=proposal.target_quantity,
            reference_price=proposal.reference_price,
            rationale=proposal.evidence_summary,
            max_notional=proposal.max_notional,
        )]
