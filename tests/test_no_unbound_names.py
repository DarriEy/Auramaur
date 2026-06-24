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
    "auramaur/bot_arb_execute.py",
    "auramaur/bot_arb_scan.py",
    "auramaur/bot_exits.py",
    "auramaur/bot_strategy_tasks.py",
    "auramaur/bot_order_monitor.py",
    "auramaur/components.py",
    "auramaur/composition.py",
]


def _scope_bindings(fn) -> set[str]:
    """Names bound directly in a function/lambda's OWN scope: params + its body's
    assignments / local imports / nested-def names / except + comprehension
    targets — descending through control-flow blocks but NOT into nested function
    bodies (those are their own scope, though their *name* binds here).
    """
    bound: set[str] = set()
    args = fn.args
    for a in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        bound.add(a.arg)
    if args.vararg:
        bound.add(args.vararg.arg)
    if args.kwarg:
        bound.add(args.kwarg.arg)

    def walk(node):
        # Process `node` itself, then descend (except into nested scopes).
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)  # binds here; its body is a separate scope
            return
        if isinstance(node, ast.Lambda):
            return  # separate scope
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        if isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        for child in ast.iter_child_nodes(node):
            walk(child)

    body = fn.body if isinstance(fn.body, list) else [fn.body]
    for stmt in body:
        walk(stmt)
    return bound


def _module_bindings(tree: ast.Module) -> set[str]:
    """Module-scope names: top-level (and top-level if-block, e.g. TYPE_CHECKING)
    imports + top-level def/class names + top-level assignments. Function-local
    imports are deliberately excluded — they bind only inside their function.
    """
    bound: set[str] = set(dir(builtins)) | {"self", "cls", "__class__"}
    stack = list(tree.body)
    while stack:
        stmt = stack.pop()
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for a in stmt.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(stmt, ast.If):
            stack.extend(stmt.body)
            stack.extend(stmt.orelse)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    bound.add(t.id)
    return bound


def _unbound_names(path: Path) -> list[str]:
    """Proper lexical-scope audit: a Load name is unbound only if it's missing
    from module scope AND every enclosing function scope (so closures and
    same-function local imports are fine; a symbol imported in method A but used
    free in method B is flagged)."""
    tree = ast.parse(path.read_text())
    problems: set[str] = set()

    def visit(node, scope: set[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                visit(child, scope | _scope_bindings(child))
            else:
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) \
                        and child.id not in scope:
                    problems.add(child.id)
                visit(child, scope)

    visit(tree, _module_bindings(tree))
    return sorted(problems)


@pytest.mark.parametrize("rel", _EXTRACTED)
def test_extracted_module_has_no_unbound_names(rel):
    missing = _unbound_names(_ROOT / rel)
    assert not missing, f"{rel} references unbound names (missing imports?): {missing}"
