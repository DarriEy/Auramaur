"""Specialized Kraken spot experiment contract and production parity."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from auramaur.experiments.strategies.kraken import (
    KrakenAction,
    KrakenRejection,
    KrakenSpotInputs,
    KrakenSpotRules,
    assess_kraken_spot,
)


RULES = KrakenSpotRules(max_order_usd=30.0, lot_decimals=4, order_minimum=0.001)


def test_entry_proposal_is_immutable_spot_order_candidate():
    result = assess_kraken_spot(KrakenSpotInputs(
        pair="XBTUSDC", price=60_000.0, holding=False, orphaned=False,
        signal_accepted=True, sized_entry_volume=0.00049,
        allocated_usd=0.0, budget_usd=600.0, paper=True,
        minimum_notional_usd=40.0,
    ), RULES)

    assert result.proposal is not None
    assert result.proposal.action is KrakenAction.BUY
    assert result.proposal.volume == 0.001
    with pytest.raises(Exception):
        result.proposal.volume = 2.0  # type: ignore[misc]


def test_live_entry_fails_closed_without_quote_funding():
    result = assess_kraken_spot(KrakenSpotInputs(
        pair="ETHUSDC", price=3_000.0, holding=False, orphaned=False,
        signal_accepted=True, sized_entry_volume=0.01,
        allocated_usd=0.0, budget_usd=100.0, free_quote=29.0, paper=False,
    ), KrakenSpotRules(max_order_usd=30.0, lot_decimals=6))
    assert result.proposal is None
    assert result.rejection is KrakenRejection.UNFUNDED


@pytest.mark.parametrize("change, rejection", [
    ({"orphaned": True}, KrakenRejection.ORPHAN_REENTRY),
    ({"in_cooldown": True}, KrakenRejection.COOLDOWN),
    ({"signal_accepted": False}, KrakenRejection.SIGNAL_GATE),
    ({"allocated_usd": 90.0}, KrakenRejection.BUDGET_FULL),
])
def test_entry_gates_fail_closed(change, rejection):
    values = dict(
        pair="XBTUSDC", price=100.0, holding=False, orphaned=False,
        signal_accepted=True, sized_entry_volume=0.3,
        allocated_usd=0.0, budget_usd=100.0, paper=True,
    )
    values.update(change)
    result = assess_kraken_spot(KrakenSpotInputs(**values), RULES)
    assert result.proposal is None
    assert result.rejection is rejection


def test_exit_floors_actual_held_quantity_and_preserves_reason():
    result = assess_kraken_spot(KrakenSpotInputs(
        pair="XBTUSDC", price=100.0, holding=True, orphaned=False,
        exit_reason="trailing_stop", held_quantity=0.123456,
    ), RULES)
    assert result.proposal is not None
    assert result.proposal.action is KrakenAction.SELL
    assert result.proposal.volume == 0.1234
    assert result.proposal.rationale == "trailing_stop"


def test_pure_module_has_no_live_execution_imports():
    path = Path("auramaur/experiments/strategies/kraken.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any(name.startswith((
        "auramaur.exchange", "auramaur.broker", "auramaur.gateway",
        "auramaur.treasury", "auramaur.db",
    )) for name in imported)
