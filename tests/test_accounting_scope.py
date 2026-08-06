"""Accounting that decides capital must measure what actually happened.

Four independent ways it did not: a token-blind join double-counting both
sides of a market, a fee subtracted twice, a settle that dropped the winning
leg, and a Brier gate scored against a coin instead of the instrument's drift.
"""

import inspect


from auramaur.evaluation.etf_calibration import Forecast, clearance


# --- token scoping ---------------------------------------------------------

def test_every_attribution_cost_basis_join_is_token_scoped():
    """portfolio and cost_basis are both keyed (market_id, is_paper, token).
    A join on two of three fans out per side: 4 positions / $160 exposure
    where the truth is 2 / $80."""
    from auramaur.monitoring import attribution
    src = inspect.getsource(attribution)
    joins = src.count("JOIN cost_basis cb") + src.count("JOIN portfolio p ON cb.market_id")
    scoped = src.count("cb.token = p.token") + src.count("p.token = cb.token")
    assert joins > 0
    assert scoped == joins, f"{joins - scoped} join(s) still unscoped by token"


def test_settlement_deletes_only_the_leg_it_settled():
    """With token_scope=None the SELECT reads ONE row; the DELETE removed
    every token row, so the unsettled side vanished and could never settle."""
    from auramaur.strategy import resolution_tracker
    src = inspect.getsource(resolution_tracker)
    delete = src.split("DELETE FROM portfolio WHERE market_id = ? AND is_paper = ? ", 1)[1]
    guard = src.split("# Scope to the token actually settled", 1)
    assert len(guard) == 2, "the scoping rationale must stay with the code"
    assert "if token:" in guard[1].split("await")[0]
    assert "UPPER(token) = UPPER(?)" in delete


# --- fees ------------------------------------------------------------------

def test_benchmark_does_not_subtract_fees_twice():
    """pnl_ledger.pnl is already net of fees at every writer; the fees column
    is the breakdown of what is already deducted. readiness.py:579 states
    this convention — ledger_report contradicted it."""
    from auramaur.monitoring import ledger_report
    src = inspect.getsource(ledger_report)
    call = src.split("risk_free_benchmark(", 1)[1].split(")", 1)[0]
    assert 'total["fees"]' not in call


# --- decision ids ----------------------------------------------------------

def test_paired_submission_plumbs_decision_ids():
    """Discarding them left every paired-arb snapshot filled=0 forever, so
    require_executable_fills filtered out the entire holdout cohort."""
    from auramaur.broker import execution_gateway
    src = inspect.getsource(execution_gateway.ExecutionGateway.submit_paired)
    assert "decision_id_a = await self._capture_decision" in src
    assert "decision_id=decision_id_a" in src
    assert "decision_id=decision_id_b" in src


# --- Brier benchmark -------------------------------------------------------

def _flat_forecaster(n: int, up_rate: float, reference):
    """A forecaster with ZERO information: it answers the base rate itself."""
    outcomes = [1 if i < round(n * up_rate) else 0 for i in range(n)]
    return [Forecast(up_rate, "MEDIUM", o, reference) for o in outcomes]


def test_zero_skill_arm_no_longer_clears_against_a_coin():
    """E[Brier] of a constant r under base rate q is (r-q)^2 + q(1-q), so a
    0.5 benchmark is EASIER than the drift by (0.5-q)^2 — free edge."""
    result = clearance(_flat_forecaster(400, 0.56, reference=None),
                       min_resolved=100)
    assert result.cleared is False
    assert "benchmark" in result.reason


def test_zero_skill_arm_scored_against_its_own_drift_does_not_clear():
    result = clearance(_flat_forecaster(400, 0.56, reference=0.56),
                       min_resolved=100)
    assert result.cleared is False


def test_a_genuinely_skilled_arm_still_clears():
    """The gate must not be merely shut — a real edge has to pass it."""
    forecasts = []
    for i in range(400):
        outcome = 1 if i % 2 == 0 else 0
        forecasts.append(Forecast(0.9 if outcome else 0.1, "HIGH", outcome, 0.56))
    result = clearance(forecasts, min_resolved=100)
    assert result.cleared is True
