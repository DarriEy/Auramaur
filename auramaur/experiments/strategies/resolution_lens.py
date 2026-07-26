"""Pure proposal formation for resolution-criteria mispricing experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

from auramaur.experiments.models import MarketSnapshot, PortfolioSnapshot, TargetPosition


@dataclass(frozen=True)
class ResolutionLensCandidate:
    market_id: str
    market_probability: float
    fair_probability: float
    gap_score: float
    mechanism: str


@dataclass(frozen=True)
class ResolutionLensRules:
    min_entry_price: float
    high_conf_gap_score: float
    stake_usd: float


@dataclass(frozen=True)
class ResolutionLensProposal:
    market_id: str
    fair_probability: float
    market_probability: float
    edge_percent: float
    buy_yes: bool
    high_confidence: bool
    evidence_summary: str
    mispricing_reason: str
    instrument_id: str
    reference_price: float
    target_quantity: float
    max_notional: float


def form_resolution_lens_proposal(
    candidate: ResolutionLensCandidate,
    rules: ResolutionLensRules,
) -> ResolutionLensProposal | None:
    """Turn an already-grounded lens candidate into bounded desired exposure."""
    values = (
        candidate.market_probability,
        candidate.fair_probability,
        candidate.gap_score,
        rules.min_entry_price,
        rules.high_conf_gap_score,
        rules.stake_usd,
    )
    if (
        not candidate.market_id
        or not candidate.mechanism.strip()
        or any(not math.isfinite(value) for value in values)
        or not 0.0 <= candidate.market_probability <= 1.0
        or not 0.0 <= candidate.fair_probability <= 1.0
        or candidate.gap_score < 0.0
        or not 0.0 <= rules.min_entry_price <= 1.0
        or rules.high_conf_gap_score < 0.0
        or rules.stake_usd <= 0.0
    ):
        return None

    edge = candidate.fair_probability - candidate.market_probability
    if edge == 0.0:
        return None
    buy_yes = edge > 0.0
    if buy_yes and candidate.market_probability < rules.min_entry_price:
        return None
    reference_price = (
        candidate.market_probability if buy_yes else 1.0 - candidate.market_probability
    )
    if not 0.0 < reference_price < 1.0:
        return None
    side = "YES" if buy_yes else "NO"
    mechanism = candidate.mechanism.strip()
    return ResolutionLensProposal(
        market_id=candidate.market_id,
        fair_probability=candidate.fair_probability,
        market_probability=candidate.market_probability,
        edge_percent=abs(edge) * 100.0,
        buy_yes=buy_yes,
        high_confidence=candidate.gap_score >= rules.high_conf_gap_score,
        evidence_summary=f"Resolution lens (gap {candidate.gap_score:.2f}): {mechanism}",
        mispricing_reason=f"behavioral: {mechanism}",
        instrument_id=f"{candidate.market_id}:{side}",
        reference_price=reference_price,
        target_quantity=rules.stake_usd / reference_price,
        max_notional=rules.stake_usd,
    )


@dataclass(frozen=True)
class ResolutionLensExperiment:
    rules: ResolutionLensRules

    async def evaluate(
        self, snapshot: MarketSnapshot, portfolio: PortfolioSnapshot,
    ) -> list[TargetPosition]:
        del portfolio
        value = snapshot.feature("resolution_lens_candidate")
        proposal = form_resolution_lens_proposal(
            ResolutionLensCandidate(
                market_id=snapshot.market_id,
                market_probability=float(value["market_probability"]),
                fair_probability=float(value["fair_probability"]),
                gap_score=float(value["gap_score"]),
                mechanism=str(value["mechanism"]),
            ),
            self.rules,
        )
        if proposal is None:
            return []
        return [TargetPosition(
            instrument_id=proposal.instrument_id,
            target_quantity=proposal.target_quantity,
            reference_price=proposal.reference_price,
            rationale=proposal.evidence_summary,
            max_notional=proposal.max_notional,
        )]
