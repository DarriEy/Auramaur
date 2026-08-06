"""The sweep's baseline keys must not depend on the host's path separator.

``rglob`` yields ``auramaur\\backtest\\engine.py`` on Windows and
``auramaur/backtest/engine.py`` on POSIX, while ``.sweep-baseline.json``
records one spelling. A key built from a raw ``str(path)`` therefore matches
the baseline on exactly one platform: before 2026-08-06 the documented
invocation reported every baselined finding as new and exited 1 on Windows
while the Linux CI job stayed green — the defect was invisible precisely
where it was checked, which is why it needs a test rather than a run.
"""

import ast

from scripts.sweep_invariants import Finding, _Sweeper


def test_key_is_separator_independent():
    windows = Finding("auramaur\\backtest\\engine.py", 91, "naive-datetime", "d")
    posix = Finding("auramaur/backtest/engine.py", 91, "naive-datetime", "d")

    assert windows.key() == posix.key()
    assert windows.key().startswith("auramaur/backtest/engine.py::")


def _dimensions(relpath: str, src: str) -> set[str]:
    tree = ast.parse(src)
    sweeper = _Sweeper(relpath, set(), set())
    sweeper.visit(tree)
    return {f.dimension for f in sweeper.findings}


def test_strategy_raw_order_covers_cancels_not_only_placements():
    """An order's lifecycle has two ends and the choke point owns both.

    cancel_order was excluded from this dimension while the gateway exposed no
    cancel contract — a pillar had nowhere to route one, so the gate could only
    have produced findings nobody could act on. ExecutionGateway.cancel_resting
    is that contract, so a raw venue cancel in a pillar is a finding again.
    """
    raw = "async def go(self):\n    await self._exchange.cancel_order('oid')\n"
    routed = "async def go(self):\n    await self._gateway.cancel_resting('oid')\n"

    assert "strategy-raw-order" in _dimensions("auramaur/strategy/market_maker.py", raw)
    assert "strategy-raw-order" not in _dimensions(
        "auramaur/strategy/market_maker.py", routed)
    # The gateway itself is where the raw venue call belongs.
    assert "strategy-raw-order" not in _dimensions(
        "auramaur/broker/execution_gateway.py", raw)
