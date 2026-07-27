import pytest
from auramaur.strategy.ibkr_etf_controls import (
    MomentumETFAnalyzer,
    compare_controls,
    completed_closes,
    cross_sectional_winners,
    dual_momentum_score,
    regime_allows_risk,
)


def _trend(n=260, rate=0.001):
    return [100 * (1 + rate) ** i for i in range(n)]


def test_completed_closes_excludes_cutoff_and_future():
    bars = [("2026-07-17", 100), ("2026-07-18", 500), ("2026-07-19", 999)]
    assert completed_closes(bars, "2026-07-18") == [100.0]


@pytest.mark.asyncio
async def test_momentum_analyzer_excludes_as_of_return():
    class Client:
        async def get_adjusted_daily_closes(self, symbol):
            return [
                (f"2025-{i // 28 + 1:02d}-{i % 28 + 1:02d}", x) for i, x in enumerate(_trend(120))
            ] + [("2026-07-19", 1.0)]

    result = await MomentumETFAnalyzer().analyze_symbol(Client(), "SPY", "2026-07-19")
    assert result is not None and result.probability == 0.70


@pytest.mark.asyncio
async def test_control_is_silent_on_a_one_month_window():
    """momentum_control's entire view is this signal, so a starved history
    means it records nothing at all — which is what happened for four days
    from a durationStr="1 M" fetch (20 sessions, 19 completed)."""

    class Client:
        def __init__(self, sessions):
            self.sessions = sessions

        async def get_adjusted_daily_closes(self, symbol):
            return [
                (f"2025-{i // 28 + 1:02d}-{i % 28 + 1:02d}", x)
                for i, x in enumerate(_trend(self.sessions))
            ] + [("2026-07-27", 1.0)]

    assert await MomentumETFAnalyzer().analyze_symbol(
        Client(19), "SPY", "2026-07-27") is None
    full = await MomentumETFAnalyzer().analyze_symbol(
        Client(250), "SPY", "2026-07-27")
    assert full is not None and full.probability == 0.70


def test_walk_forward_and_placebo_are_stable():
    base = _trend(180)
    assert (
        compare_controls(base + [base[-1] * 2]).momentum
        > compare_controls(base + [base[-1] * 0.5]).momentum
    )
    assert compare_controls(base) == compare_controls(base)


def test_research_challenger_primitives():
    up, down = _trend(), _trend(rate=-0.001)
    assert dual_momentum_score(up, down) > 0
    assert cross_sectional_winners({"UP": up, "DOWN": down}, 1) == ["UP"]
    assert regime_allows_risk(up) and not regime_allows_risk(down)
