"""Pure candidate selection and target generation for deadline ladders."""

from __future__ import annotations

from dataclasses import dataclass

from auramaur.experiments.models import MarketSnapshot, PortfolioSnapshot, TargetPosition


@dataclass(frozen=True)
class TermStructureCandidate:
    market_id: str
    market_probability: float
    model_probability: float
    liquidity: float
    claimed: bool = False


@dataclass(frozen=True)
class TermStructureRules:
    min_edge_percent: float
    min_liquidity: float
    max_entries: int
    stake_usd: float


@dataclass(frozen=True)
class TermStructureProposal:
    market_id: str
    buy_yes: bool
    instrument_id: str
    market_probability: float
    fair_probability: float
    edge_percent: float
    reference_price: float
    target_quantity: float
    max_notional: float


def select_term_structure_proposals(
    candidates: list[TermStructureCandidate],
    rules: TermStructureRules,
) -> list[TermStructureProposal]:
    """Select the largest executable curve gaps using production semantics."""
    eligible: list[tuple[float, TermStructureCandidate]] = []
    for candidate in candidates:
        gap = abs(candidate.model_probability - candidate.market_probability) * 100.0
        if (
            candidate.claimed
            or candidate.liquidity < rules.min_liquidity
            or gap < rules.min_edge_percent
        ):
            continue
        eligible.append((gap, candidate))
    eligible.sort(key=lambda item: item[0], reverse=True)
    proposals: list[TermStructureProposal] = []
    for gap, candidate in eligible[: rules.max_entries]:
        buy_yes = candidate.model_probability > candidate.market_probability
        reference_price = (
            candidate.market_probability if buy_yes else 1.0 - candidate.market_probability
        )
        if not 0.0 < reference_price < 1.0:
            continue
        proposals.append(TermStructureProposal(
            market_id=candidate.market_id,
            buy_yes=buy_yes,
            instrument_id=f"{candidate.market_id}:{'YES' if buy_yes else 'NO'}",
            market_probability=candidate.market_probability,
            fair_probability=candidate.model_probability,
            edge_percent=gap,
            reference_price=reference_price,
            target_quantity=rules.stake_usd / reference_price,
            max_notional=rules.stake_usd,
        ))
    return proposals


@dataclass(frozen=True)
class TermStructureExperiment:
    rules: TermStructureRules

    async def evaluate(
        self, snapshot: MarketSnapshot, portfolio: PortfolioSnapshot,
    ) -> list[TargetPosition]:
        del portfolio
        details = snapshot.feature("term_structure_candidates")
        proposals = select_term_structure_proposals(
            [TermStructureCandidate(
                market_id=str(item["market_id"]),
                market_probability=float(item["market_probability"]),
                model_probability=float(item["model_probability"]),
                liquidity=float(item["liquidity"]),
                claimed=bool(item.get("claimed", False)),
            ) for item in details],
            self.rules,
        )
        return [TargetPosition(
            instrument_id=proposal.instrument_id,
            target_quantity=proposal.target_quantity,
            reference_price=proposal.reference_price,
            rationale="deadline-ladder curve mispricing",
            max_notional=proposal.max_notional,
        ) for proposal in proposals]
