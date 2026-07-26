"""Parity and isolation tests for the portable weather-temperature proposal."""

from __future__ import annotations

import ast
import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from auramaur.experiments.models import (
    FeatureValue,
    MarketSnapshot,
    PortfolioSnapshot,
    PricePoint,
)
from auramaur.experiments.strategies.weather_temp import (
    WeatherTempInputs,
    WeatherTempRejection,
    WeatherTempRules,
    WeatherTempExperiment,
    assess_weather_temp,
)


def _inputs(**changes) -> WeatherTempInputs:
    values = {
        "market_id": "weather-1",
        "market_probability": 0.27,
        "model_probability": 0.03,
        "member_count": 10,
        "city": "Tokyo",
        "temperature_kind": "high",
        "already_entered_or_held": False,
    }
    values.update(changes)
    return WeatherTempInputs(**values)


def test_pure_assessment_matches_production_sell_signal():
    result = assess_weather_temp(
        _inputs(),
        WeatherTempRules(required_edge=0.10, max_divergence=0.40, stake_usd=10),
    )
    assert result.rejection is None
    assert result.proposal is not None
    assert result.proposal.buy_yes is False
    assert result.proposal.edge_percent == pytest.approx(24)
    assert result.proposal.reference_price == pytest.approx(0.73)
    assert result.proposal.max_notional == 10


@pytest.mark.parametrize(
    ("changes", "rejection"),
    [
        ({"model_probability": None}, WeatherTempRejection.NO_MODEL_PRICE),
        ({"model_probability": 0.30}, WeatherTempRejection.INSUFFICIENT_EDGE),
        ({"model_probability": 0.90}, WeatherTempRejection.IMPLAUSIBLE_DIVERGENCE),
        ({"already_entered_or_held": True}, WeatherTempRejection.ALREADY_ENTERED_OR_HELD),
    ],
)
def test_rejection_parity(changes, rejection):
    result = assess_weather_temp(
        _inputs(**changes),
        WeatherTempRules(required_edge=0.10, max_divergence=0.40, stake_usd=10),
    )
    assert result.proposal is None
    assert result.rejection == rejection


def test_portable_adapter_emits_same_target_without_live_dependencies():
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    payload = {
        "market_probability": 0.27,
        "model_probability": 0.03,
        "member_count": 10,
        "city": "Tokyo",
        "temperature_kind": "high",
        "already_entered_or_held": False,
    }
    snapshot = MarketSnapshot(
        observed_at=now,
        sequence=1,
        market_id="weather-1",
        venue="polymarket",
        prices=(PricePoint(instrument_id="weather-1:NO", price=0.73),),
        features=(FeatureValue(
            name="weather_temp_inputs",
            payload=payload,
            source="test",
            observed_at=now,
            available_at=now,
        ),),
        data_version="test-v1",
    )
    portfolio = PortfolioSnapshot(observed_at=now, cash=100, equity=100)
    targets = asyncio.run(WeatherTempExperiment(
        WeatherTempRules(required_edge=0.10, max_divergence=0.40, stake_usd=10)
    ).evaluate(snapshot, portfolio))
    assert len(targets) == 1
    assert targets[0].instrument_id == "weather-1:NO"
    assert targets[0].reference_price == pytest.approx(0.73)
    assert targets[0].target_quantity == pytest.approx(10 / 0.73)

    source = Path("auramaur/experiments/strategies/weather_temp.py").read_text()
    imports = {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    }
    forbidden = ("auramaur.broker", "auramaur.db", "auramaur.exchange")
    assert not any(name.startswith(forbidden) for name in imports)
