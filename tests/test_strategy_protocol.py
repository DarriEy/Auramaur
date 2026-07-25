"""Conformance guards generated from the authoritative strategy registry."""

from __future__ import annotations

import ast
import pathlib

import pytest

from auramaur.bot import AuramaurBot
from auramaur.strategy.protocols import (
    DeploymentMode,
    ExecutionMode,
    NO_DIRECT_PLACE_MODES,
    StrategyCapability,
)
from auramaur.strategy.registry import (
    FULL_TRADE_PIPELINE,
    NON_STRATEGY_TASKS,
    STRATEGY_BY_TASK,
    STRATEGY_SPECS,
)

_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Placement primitives an application module must never call directly when its
# declared mode routes through the gateway. Kept in sync with the exposure
# registry's perimeter so a new adapter verb is covered in one place.
_PLACEMENT_METHODS = frozenset({
    "place_order", "place_spot_order", "place_share_order", "_place_cash_order",
    "place", "placeOrder",
})


def test_registry_identity_and_task_entrypoints_are_unique():
    assert len({spec.key for spec in STRATEGY_SPECS}) == len(STRATEGY_SPECS)
    assert len({spec.task for spec in STRATEGY_SPECS}) == len(STRATEGY_SPECS)
    for spec in STRATEGY_SPECS:
        assert hasattr(AuramaurBot, spec.task), spec.key
        assert spec.venues
        assert FULL_TRADE_PIPELINE <= spec.capabilities


@pytest.mark.parametrize("spec", STRATEGY_SPECS, ids=lambda spec: spec.key)
def test_registered_pillar_declares_matching_execution_contract(spec):
    cls = spec.resolve_class()
    assert cls.execution_mode is spec.execution_mode
    assert callable(getattr(cls, spec.runner, None)), (spec.key, spec.runner)
    # Families with venue/model-scoped runtime names may expose name as a
    # property, but every implementation must explicitly expose one.
    assert hasattr(cls, "name")


@pytest.mark.parametrize("spec", STRATEGY_SPECS, ids=lambda spec: spec.key)
def test_gateway_pillars_do_not_place_orders_directly(spec):
    if spec.execution_mode not in NO_DIRECT_PLACE_MODES:
        pytest.skip(f"{spec.key} has declared {spec.execution_mode.value} execution")
    # Match EVERY placement primitive, not just the literal `.place_order(`.
    # Kraken places via `place_spot_order`, IBKR equities via
    # `place_share_order`/`_place_cash_order`, and the multiasset bridge via
    # `.place(` — none of which the old substring saw, so a pillar could
    # bypass the gateway and still pass this guard.
    src = (_ROOT / (spec.module.replace(".", "/") + ".py")).read_text()
    tree = ast.parse(src)
    offenders = sorted({
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in _PLACEMENT_METHODS
    })
    assert not offenders, (
        f"{spec.key} ({spec.execution_mode.value}) must use its declared "
        f"gateway/simulation path, but calls {offenders} directly"
    )


def test_paper_only_is_explicit_and_cannot_be_mistaken_for_graduatable():
    paper_only = {s.key for s in STRATEGY_SPECS if s.deployment is DeploymentMode.PAPER_ONLY}
    assert paper_only == {"ibkr_etf_paper"}
    for spec in STRATEGY_SPECS:
        if spec.execution_mode is ExecutionMode.PAPER_SIMULATED:
            assert spec.deployment in {DeploymentMode.PAPER_ONLY, DeploymentMode.GRADUATABLE}


def test_arb_mixin_does_not_place_orders_directly():
    src = (_ROOT / "auramaur/bot_arb.py").read_text()
    assert ".place_order(" not in src


def test_exit_path_only_ibkr_places_directly():
    src = (_ROOT / "auramaur/bot_exits.py").read_text()
    tree = ast.parse(src)
    ibkr = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "_execute_ibkr_exit"
    )
    bad = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "place_order"
        and not (ibkr.lineno <= node.lineno <= ibkr.end_lineno)
    ]
    assert not bad


def test_capability_enum_covers_trade_lifecycle():
    assert FULL_TRADE_PIPELINE == {
        StrategyCapability.DISCOVERY,
        StrategyCapability.MARKET_DATA,
        StrategyCapability.SIGNAL,
        StrategyCapability.RISK,
        StrategyCapability.EXECUTION,
        StrategyCapability.RECORDING,
        StrategyCapability.RECONCILIATION,
        StrategyCapability.SETTLEMENT,
    }


def test_every_bot_task_is_classified_as_strategy_or_non_strategy():
    # Discover the modules rather than listing them: hardcoding three files
    # meant bot_order_monitor.py's tasks — one of which cancels LIVE resting
    # orders — were never required to be classified at all.
    discovered: set[str] = set()
    for path in sorted((_ROOT / "auramaur").glob("bot*.py")):
        relative = path.relative_to(_ROOT).as_posix()
        tree = ast.parse((_ROOT / relative).read_text(encoding="utf-8"))
        discovered.update(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("_task_")
        )
    assert not (set(STRATEGY_BY_TASK) & NON_STRATEGY_TASKS)
    assert discovered == set(STRATEGY_BY_TASK) | set(NON_STRATEGY_TASKS)
