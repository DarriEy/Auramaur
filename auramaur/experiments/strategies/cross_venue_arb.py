"""Pure, atomic paired-package proposals for cross-venue arbitrage."""
from __future__ import annotations
from dataclasses import dataclass
from auramaur.experiments.models import MarketSnapshot, PortfolioSnapshot, TargetPosition

@dataclass(frozen=True)
class PairedLeg:
    market_id: str
    venue: str
    side: str
    market_probability: float
    target: TargetPosition

@dataclass(frozen=True)
class CrossVenuePairProposal:
    """One indivisible research proposal containing both hedge legs."""
    package_id: str
    orientation: str
    edge: float
    confidence: float
    legs: tuple[PairedLeg, PairedLeg]

    @property
    def targets(self) -> tuple[TargetPosition, TargetPosition]:
        return self.legs[0].target, self.legs[1].target

def paired_arb_proposal(*, market_a_id: str, venue_a: str, yes_price_a: float,
                        market_b_id: str, venue_b: str, yes_price_b: float,
                        orientation: str, confidence: float, min_confidence: float,
                        required_gap: float, stake_usd: float) -> CrossVenuePairProposal | None:
    """Form an atomic two-leg proposal, or fail closed."""
    if (orientation not in {"same", "inverted"} or confidence < min_confidence
            or stake_usd <= 0 or required_gap < 0
            or not (0 < yes_price_a < 1) or not (0 < yes_price_b < 1)
            or not market_a_id or not market_b_id or market_a_id == market_b_id
            or not venue_a or not venue_b or venue_a == venue_b):
        return None
    if orientation == "same":
        edge = abs(yes_price_a - yes_price_b)
        sides = ("BUY", "SELL") if yes_price_a < yes_price_b else ("SELL", "BUY")
    else:
        total = yes_price_a + yes_price_b
        edge = abs(1.0 - total)
        sides = ("BUY", "BUY") if total < 1.0 else ("SELL", "SELL")
    if edge < required_gap:
        return None
    legs = []
    rationale = f"cross-venue {orientation} package; gap {edge:.3f}"
    for market_id, venue, yes_price, side in zip(
            (market_a_id, market_b_id), (venue_a, venue_b),
            (yes_price_a, yes_price_b), sides, strict=True):
        contract = "YES" if side == "BUY" else "NO"
        contract_price = yes_price if side == "BUY" else 1.0 - yes_price
        legs.append(PairedLeg(market_id=market_id, venue=venue, side=side,
            market_probability=yes_price, target=TargetPosition(
                instrument_id=f"{venue}:{market_id}:{contract}",
                target_quantity=stake_usd / contract_price,
                reference_price=contract_price, rationale=rationale,
                max_notional=stake_usd)))
    return CrossVenuePairProposal(
        package_id=f"{venue_a}:{market_a_id}|{venue_b}:{market_b_id}",
        orientation=orientation, edge=edge, confidence=confidence,
        legs=(legs[0], legs[1]))

@dataclass(frozen=True)
class CrossVenueArbExperiment:
    min_confidence: float
    required_gap: float
    stake_usd: float

    async def evaluate(self, snapshot: MarketSnapshot,
                       portfolio: PortfolioSnapshot) -> list[TargetPosition]:
        del portfolio
        try:
            pair = snapshot.feature("cross_venue_pair")
            proposal = paired_arb_proposal(
                market_a_id=str(pair["market_a_id"]), venue_a=str(pair["venue_a"]),
                yes_price_a=float(pair["yes_price_a"]), market_b_id=str(pair["market_b_id"]),
                venue_b=str(pair["venue_b"]), yes_price_b=float(pair["yes_price_b"]),
                orientation=str(pair["orientation"]), confidence=float(pair["confidence"]),
                min_confidence=self.min_confidence, required_gap=self.required_gap,
                stake_usd=self.stake_usd)
        except (KeyError, TypeError, ValueError):
            return []
        return list(proposal.targets) if proposal is not None else []
