"""Mechanical completeness checks for every application exposure mutation."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

from auramaur.strategy.exposure_registry import (
    EXPOSURE_BY_KEY,
    EXPOSURE_PATHS,
    REGISTERED_CALLSITES,
    SENSITIVE_METHODS,
)

_ROOT = Path(__file__).resolve().parent.parent


def _sensitive_callsites() -> Counter[tuple[str, str, str]]:
    found: Counter[tuple[str, str, str]] = Counter()
    for path in sorted((_ROOT / "auramaur").rglob("*.py")):
        relative = path.relative_to(_ROOT).as_posix()
        # Venue adapters are the registered execution boundary. Their internal
        # SDK/HTTP calls are covered by adapter tests; the independently gated
        # IBKR live bridge remains application-level and is included.
        if "exchange" in path.parts and path.name != "ibkr_multiasset_execution.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in SENSITIVE_METHODS
            ):
                continue
            owner: ast.AST | None = node
            while owner is not None and not isinstance(
                owner, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                owner = parents.get(owner)
            function = owner.name if owner is not None else "<module>"
            found[(relative, function, node.func.attr)] += 1
    return found


def test_every_sensitive_callsite_is_registered_exactly_once():
    owners = [(module, function, method) for module, function, method, _ in REGISTERED_CALLSITES]
    assert len(owners) == len(set(owners)), "a sensitive callsite has multiple path owners"
    expected = Counter()
    for (module, function, method, path_key), count in REGISTERED_CALLSITES.items():
        assert path_key in EXPOSURE_BY_KEY
        expected[(module, function, method)] += count
    assert _sensitive_callsites() == expected


def test_every_exposure_path_has_a_complete_lifecycle_contract():
    assert len(EXPOSURE_BY_KEY) == len(EXPOSURE_PATHS)
    used_paths = {key for (*_, key) in REGISTERED_CALLSITES}

    assert set(EXPOSURE_BY_KEY) == used_paths

    for path in EXPOSURE_PATHS:
        assert path.data_services
        assert path.modes
        for field in (
            path.decision_source,
            path.risk_authority,
            path.execution_boundary,
            path.booking,
            path.monitoring,
            path.exit_path,
            path.reconciliation,
            path.settlement,
            path.attribution,
        ):
            assert field and field.strip(), (path.key, field)


def test_agent_tool_is_registered_paper_only_and_server_forces_live_off():
    path = EXPOSURE_BY_KEY["agent_paper"]
    assert path.modes == {"paper"}
    server = (_ROOT / "auramaur/agentmcp/server.py").read_text(encoding="utf-8")
    book = (_ROOT / "auramaur/agentmcp/book.py").read_text(encoding="utf-8")
    assert 'os.environ["AURAMAUR_LIVE"] = "false"' in server
    assert "is_paper=True" in book


def test_web_remains_outside_the_exposure_perimeter():
    for path in (_ROOT / "auramaur/web").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in SENSITIVE_METHODS
            for node in ast.walk(tree)
        ), path


def test_every_live_adapter_submission_boundary_is_known():
    """Audit the final SDK/HTTP calls hidden behind registered venue adapters."""
    expected = Counter(
        {
            ("auramaur/exchange/client.py", "_submit_clob_order", "create_and_post_order"): 1,
            ("auramaur/exchange/kalshi.py", "_post", "call_api:POST"): 1,
            ("auramaur/exchange/cryptodotcom.py", "place_order", "_private_post:create-order"): 1,
            ("auramaur/exchange/kraken.py", "place_spot_order", "_private:AddOrder"): 1,
            ("auramaur/exchange/ibkr.py", "place_order", "placeOrder"): 1,
            ("auramaur/exchange/ibkr_equity.py", "place_share_order", "placeOrder"): 1,
            ("auramaur/exchange/ibkr_equity.py", "_place_cash_order", "placeOrder"): 1,
            ("auramaur/exchange/ibkr_equity.py", "ensure_usd_float", "placeOrder"): 1,
            ("auramaur/exchange/ibkr_multiasset_execution.py", "place", "placeOrder"): 1,
            ("auramaur/broker/onchain.py", "redeem", "send_raw_transaction"): 1,
        }
    )
    found: Counter[tuple[str, str, str]] = Counter()
    roots = (_ROOT / "auramaur/exchange", _ROOT / "auramaur/broker/onchain.py")
    files = [p for root in roots for p in ([root] if root.is_file() else root.rglob("*.py"))]
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            marker = ""
            attr = node.func.attr
            first = node.args[0] if node.args else None
            if attr in {"create_and_post_order", "placeOrder", "send_raw_transaction"}:
                marker = attr
            elif (
                attr == "_private" and isinstance(first, ast.Constant) and first.value == "AddOrder"
            ):
                marker = "_private:AddOrder"
            elif (
                attr == "_private_post"
                and isinstance(first, ast.Constant)
                and first.value == "private/create-order"
            ):
                marker = "_private_post:create-order"
            elif attr == "call_api" and isinstance(first, ast.Constant) and first.value == "POST":
                marker = "call_api:POST"
            if not marker:
                continue
            owner: ast.AST | None = node
            while owner is not None and not isinstance(
                owner, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                owner = parents.get(owner)
            found[(path.relative_to(_ROOT).as_posix(), owner.name, marker)] += 1
    assert found == expected
