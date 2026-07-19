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
