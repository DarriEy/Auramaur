"""The sweep's baseline keys must not depend on the host's path separator.

``rglob`` yields ``auramaur\\backtest\\engine.py`` on Windows and
``auramaur/backtest/engine.py`` on POSIX, while ``.sweep-baseline.json``
records one spelling. A key built from a raw ``str(path)`` therefore matches
the baseline on exactly one platform: before 2026-08-06 the documented
invocation reported every baselined finding as new and exited 1 on Windows
while the Linux CI job stayed green — the defect was invisible precisely
where it was checked, which is why it needs a test rather than a run.
"""

from scripts.sweep_invariants import Finding


def test_key_is_separator_independent():
    windows = Finding("auramaur\\backtest\\engine.py", 91, "naive-datetime", "d")
    posix = Finding("auramaur/backtest/engine.py", 91, "naive-datetime", "d")

    assert windows.key() == posix.key()
    assert windows.key().startswith("auramaur/backtest/engine.py::")
