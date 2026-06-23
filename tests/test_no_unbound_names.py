"""Guard against the mixin-extraction failure mode: a method moved into a new
module that uses a module-level symbol (often a *lowercase* helper like
blocked_category_hit / taker_fee_rate / console) which wasn't imported there.

Such a bug imports fine and passes tests whose paths don't hit the symbol, then
NameErrors at runtime on a live code path — exactly what shipped in the engine
cycle split. This audits the extracted modules for free names with no module-,
class-, function-, parameter-, assignment-, or except-binding.
"""

from __future__ import annotations

import ast
import builtins
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_EXTRACTED = [
    "auramaur/strategy/engine_cycle.py",
    "auramaur/bot_arb.py",
    "auramaur/bot_exits.py",
    "auramaur/bot_strategy_tasks.py",
    "auramaur/bot_order_monitor.py",
]


def _unbound_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    bound = set(dir(builtins)) | {"self", "cls", "__class__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return sorted(u for u in used if u not in bound)


@pytest.mark.parametrize("rel", _EXTRACTED)
def test_extracted_module_has_no_unbound_names(rel):
    missing = _unbound_names(_ROOT / rel)
    assert not missing, f"{rel} references unbound names (missing imports?): {missing}"
