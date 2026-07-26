"""Pure published-outcome settlement-lag proposal."""

from __future__ import annotations

from dataclasses import dataclass

from auramaur.experiments.models import MarketSnapshot, PortfolioSnapshot, TargetPosition


def _bin_half_width(threshold: float) -> float:
    text = f"{threshold:.10f}".rstrip("0")
    decimals = len(text.split(".")[1]) if "." in text else 0
    return 0.5 * (10.0 ** -decimals)


def is_satisfied(value: float, operator: str, threshold: float) -> bool:
    if operator == ">=":
        return value >= threshold
    if operator == ">":
        return value > threshold
    if operator == "<=":
        return value <= threshold
    if operator == "<":
        return value < threshold
    if operator == "==":
        return abs(value - threshold) < _bin_half_width(threshold) + 1e-12
    return False


@dataclass(frozen=True)
class SettlementProposal:
    market_id: str
    buy_yes: bool
    fair_probability: float
    market_probability: float
    edge_percent: float
    published_value: float
    evidence_summary: str
    target: TargetPosition


def settlement_proposal(
    *, market_id: str, yes_price: float, no_price: float, published_value: float,
    indicator: str, reference_period: str, operator: str, threshold: float,
    min_edge: float, stake_usd: float,
) -> SettlementProposal | None:
    yes_locked = is_satisfied(published_value, operator, threshold)
    fair = 1.0 if yes_locked else 0.0
    edge = fair - yes_price
    if abs(edge) < min_edge:
        return None
    buy_yes = edge > 0
    price = yes_price if buy_yes else no_price
    if price <= 0:
        return None
    evidence = (
        f"settlement_arb: {indicator} {reference_period} = {published_value:.2f} "
        f"{operator} {threshold} -> {'YES' if yes_locked else 'NO'} locked"
    )
    return SettlementProposal(
        market_id=market_id, buy_yes=buy_yes, fair_probability=fair,
        market_probability=yes_price, edge_percent=abs(edge) * 100.0,
        published_value=published_value, evidence_summary=evidence,
        target=TargetPosition(
            instrument_id=f"{market_id}:{'YES' if buy_yes else 'NO'}",
            target_quantity=stake_usd / price,
            reference_price=price,
            rationale="official print already determines the criterion",
            max_notional=stake_usd,
        ),
    )


@dataclass(frozen=True)
class SettlementArbExperiment:
    min_edge: float
    stake_usd: float

    async def evaluate(self, snapshot: MarketSnapshot,
                       portfolio: PortfolioSnapshot) -> list[TargetPosition]:
        del portfolio
        data = snapshot.feature("settlement_predicate")
        proposal = settlement_proposal(
            market_id=snapshot.market_id,
            yes_price=snapshot.price(f"{snapshot.market_id}:YES"),
            no_price=snapshot.price(f"{snapshot.market_id}:NO"),
            published_value=float(data["published_value"]),
            indicator=str(data["indicator"]), reference_period=str(data["reference_period"]),
            operator=str(data["operator"]), threshold=float(data["threshold"]),
            min_edge=self.min_edge, stake_usd=self.stake_usd,
        )
        return [proposal.target] if proposal else []
