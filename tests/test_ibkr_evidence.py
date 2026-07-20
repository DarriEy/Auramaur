from auramaur.risk.ibkr_evidence import evaluate_ibkr_evidence


def test_twenty_profitable_trades_cannot_graduate():
    evidence = evaluate_ibkr_evidence([1.0] * 20, elapsed_days=90, budget_usd=5_000)
    assert not evidence.ready
    assert any("20/200" in reason for reason in evidence.reasons)


def test_cost_adjusted_long_record_can_graduate_with_calibration():
    pnls = [1.2 if i % 3 else -0.5 for i in range(240)]
    outcomes = [1 if i % 3 else 0 for i in range(240)]
    probabilities = [0.75 if outcome else 0.25 for outcome in outcomes]
    evidence = evaluate_ibkr_evidence(
        pnls, elapsed_days=220, budget_usd=5_000,
        probabilities=probabilities, outcomes=outcomes)
    assert evidence.ready
    assert evidence.brier_score < evidence.baseline_brier_score


def test_drawdown_and_uncertainty_fail_closed():
    evidence = evaluate_ibkr_evidence(
        [5.0] * 210 + [-100.0] * 10, elapsed_days=220, budget_usd=1_000)
    assert not evidence.ready
    assert evidence.max_drawdown_usd == 1_000


# ---- daily-marks contract (pre-registered 2026-07-20) ----------------------

def test_daily_evidence_reachable_for_slow_turnover_books():
    from auramaur.risk.ibkr_evidence import evaluate_ibkr_daily_evidence
    # 180 days of steady small positive daily P&L: exactly the profile a
    # 2-6 week-hold FX book can produce but the 200-trip contract cannot see.
    daily = [3.0 + (0.5 if i % 3 else -0.5) for i in range(130)]
    out = evaluate_ibkr_daily_evidence(daily, elapsed_days=185, budget_usd=5000)
    assert out.ready, out.reasons
    assert out.observations == 130


def test_daily_evidence_rejects_short_windows_and_weak_means():
    from auramaur.risk.ibkr_evidence import evaluate_ibkr_daily_evidence
    thin = evaluate_ibkr_daily_evidence([5.0] * 50, elapsed_days=60,
                                        budget_usd=5000)
    assert not thin.ready
    assert any("50/120" in r for r in thin.reasons)
    assert any("60/180" in r for r in thin.reasons)
    noisy = evaluate_ibkr_daily_evidence(
        [100.0 if i % 2 else -99.0 for i in range(130)],
        elapsed_days=185, budget_usd=5000)
    assert not noisy.ready
    assert any("lower bound" in r for r in noisy.reasons)
