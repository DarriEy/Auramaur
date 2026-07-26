"""Pure weather-temperature candidate assessment for every runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from auramaur.experiments.models import MarketSnapshot, PortfolioSnapshot, TargetPosition


class WeatherTempRejection(str, Enum):
    NO_MODEL_PRICE = "no_model_price"
    INSUFFICIENT_EDGE = "insufficient_edge"
    IMPLAUSIBLE_DIVERGENCE = "implausible_divergence"
    ALREADY_ENTERED_OR_HELD = "already_entered_or_held"


@dataclass(frozen=True)
class WeatherTempRules:
    required_edge: float
    max_divergence: float
    stake_usd: float


@dataclass(frozen=True)
class WeatherTempInputs:
    market_id: str
    market_probability: float
    model_probability: float | None
    member_count: int
    city: str
    temperature_kind: str
    already_entered_or_held: bool


@dataclass(frozen=True)
class WeatherTempProposal:
    market_id: str
    buy_yes: bool
    instrument_id: str
    market_probability: float
    model_probability: float
    edge_percent: float
    reference_price: float
    target_quantity: float
    max_notional: float
    evidence_summary: str


@dataclass(frozen=True)
class WeatherTempAssessment:
    proposal: WeatherTempProposal | None
    rejection: WeatherTempRejection | None = None


def assess_weather_temp(
    inputs: WeatherTempInputs,
    rules: WeatherTempRules,
) -> WeatherTempAssessment:
    model_probability = inputs.model_probability
    if model_probability is None:
        return WeatherTempAssessment(None, WeatherTempRejection.NO_MODEL_PRICE)
    edge = model_probability - inputs.market_probability
    if abs(edge) < rules.required_edge:
        return WeatherTempAssessment(None, WeatherTempRejection.INSUFFICIENT_EDGE)
    if abs(edge) > rules.max_divergence:
        return WeatherTempAssessment(None, WeatherTempRejection.IMPLAUSIBLE_DIVERGENCE)
    if inputs.already_entered_or_held:
        return WeatherTempAssessment(None, WeatherTempRejection.ALREADY_ENTERED_OR_HELD)

    buy_yes = edge > 0
    reference_price = (
        inputs.market_probability if buy_yes else 1.0 - inputs.market_probability
    )
    return WeatherTempAssessment(WeatherTempProposal(
        market_id=inputs.market_id,
        buy_yes=buy_yes,
        instrument_id=f"{inputs.market_id}:{'YES' if buy_yes else 'NO'}",
        market_probability=inputs.market_probability,
        model_probability=model_probability,
        edge_percent=abs(edge) * 100.0,
        reference_price=reference_price,
        target_quantity=rules.stake_usd / reference_price,
        max_notional=rules.stake_usd,
        evidence_summary=(
            f"GEFS ensemble P(bin)={model_probability:.2f} "
            f"({inputs.member_count} members) vs market "
            f"{inputs.market_probability:.2f} for {inputs.city} "
            f"{inputs.temperature_kind} temp."
        ),
    ))


@dataclass(frozen=True)
class WeatherTempExperiment:
    rules: WeatherTempRules

    async def evaluate(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioSnapshot,
    ) -> list[TargetPosition]:
        del portfolio
        details = snapshot.feature("weather_temp_inputs")
        raw_model = details["model_probability"]
        assessment = assess_weather_temp(
            WeatherTempInputs(
                market_id=snapshot.market_id,
                market_probability=float(details["market_probability"]),
                model_probability=None if raw_model is None else float(raw_model),
                member_count=int(details["member_count"]),
                city=str(details["city"]),
                temperature_kind=str(details["temperature_kind"]),
                already_entered_or_held=bool(details["already_entered_or_held"]),
            ),
            self.rules,
        )
        if assessment.proposal is None:
            return []
        proposal = assessment.proposal
        return [TargetPosition(
            instrument_id=proposal.instrument_id,
            target_quantity=proposal.target_quantity,
            reference_price=proposal.reference_price,
            rationale=proposal.evidence_summary,
            max_notional=proposal.max_notional,
        )]
