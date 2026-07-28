"""The ETF threshold decision aid, pinned against over-claiming.

The point of this report is to STOP a threshold being read off a thin sample.
Most of these tests therefore assert that it declines to find an edge, which is
the behaviour that is easy to regress by "improving" the statistics.
"""

import math

import pytest

from auramaur.evaluation.etf_calibration import (
    COIN,
    Forecast,
    brier_score,
    confidence_bands,
    probability_bands,
    samples_for_reliable_edge,
    suggested_thresholds,
    threshold_sweep,
    wilson_interval,
)


def _forecasts(spec):
    """spec: list of (probability, confidence, outcome-or-None)."""
    return [Forecast(p, c, o) for p, c, o in spec]


def _run(probability, confidence, hits, misses):
    return _forecasts([(probability, confidence, 1)] * hits
                      + [(probability, confidence, 0)] * misses)


def test_wilson_stays_inside_zero_one_where_the_normal_interval_does_not():
    # 9/10 breaks the normal approximation: 0.9 + 1.96*sqrt(.9*.1/10) = 1.086.
    lo, hi = wilson_interval(9, 10)
    assert 0.0 <= lo <= hi <= 1.0
    naive = 0.9 + 1.96 * math.sqrt(0.9 * 0.1 / 10)
    assert naive > 1.0 and hi < 1.0
    # Degenerate samples must claim nothing rather than divide by zero.
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(1, 1)[0] < 1.0


def test_a_thin_sample_does_not_beat_a_coin():
    """12/20 is a 60% hit rate and is NOT evidence of an edge."""
    band = confidence_bands(_run(0.55, "LOW", 12, 8))[0]
    assert band.n == 20 and band.realized == 0.6
    assert not band.beats_coin
    assert band.lo < COIN < band.hi
    # The same rate on a large sample is evidence.
    assert confidence_bands(_run(0.55, "LOW", 120, 80))[0].beats_coin


def test_samples_needed_answers_how_much_longer():
    needed = samples_for_reliable_edge(0.60)
    assert needed is not None and 60 < needed < 200
    assert wilson_interval(round(0.60 * needed), needed)[0] > COIN
    assert wilson_interval(round(0.60 * (needed - 1)), needed - 1)[0] <= COIN
    # A stronger edge needs fewer resolutions; a coin needs infinitely many.
    assert samples_for_reliable_edge(0.80) < needed
    assert samples_for_reliable_edge(0.50) is None
    assert samples_for_reliable_edge(0.43) is None


def test_calibration_gap_exposes_an_underconfident_arm():
    """An arm told to stay near 0.50 can still be right 65% of the time.

    That is the case the threshold decision hinges on: the fix is to lower the
    gate to meet the arm's honest scale, not to demand it inflate.
    """
    band = confidence_bands(_run(0.52, "LOW", 65, 35))[0]
    assert band.calibration_gap > 0.12
    assert band.beats_coin


def test_probability_bands_resolve_the_narrow_range_the_arms_produce():
    """Decile bins would collapse 0.43-0.56 into two buckets."""
    forecasts = _forecasts(
        [(0.44, "LOW", 0), (0.45, "LOW", 0), (0.51, "LOW", 1),
         (0.52, "LOW", 1), (0.55, "MEDIUM_LOW", 1), (0.56, "MEDIUM_LOW", 1)])
    bands = probability_bands(forecasts, width=0.02)
    assert len(bands) >= 3
    assert sum(b.n for b in bands) == 6
    assert bands[0].realized == 0.0 and bands[-1].realized == 1.0
    # Unresolved forecasts are excluded, never counted as misses.
    assert probability_bands(_forecasts([(0.5, "LOW", None)])) == []


def test_confidence_bands_order_by_rank_and_can_fail_to_separate():
    forecasts = _run(0.50, "LOW", 10, 10) + _run(0.55, "MEDIUM_LOW", 5, 5)
    bands = confidence_bands(forecasts)
    assert [b.label for b in bands] == ["LOW", "MEDIUM_LOW"]
    # Identical hit rates: stated confidence carries no information here, so
    # etf_arm_min_confidence should not be the binding gate.
    assert bands[0].realized == bands[1].realized == 0.5
    assert not any(b.beats_coin for b in bands)


def test_threshold_sweep_reads_both_halves_of_the_entry_gate():
    forecasts = (_run(0.45, "LOW", 2, 8)          # weak, often wrong
                 + _run(0.55, "MEDIUM_LOW", 8, 2))  # strong, often right
    sweep = {row.threshold: row for row in
             threshold_sweep(forecasts, [0.40, 0.50, 0.60])}
    assert sweep[0.40].n == 20 and sweep[0.40].hits == 10
    assert sweep[0.50].n == 10 and sweep[0.50].realized == 0.8
    # Above the observed ceiling a threshold is an off switch, not a filter.
    assert sweep[0.60].n == 0
    # The confidence floor applies the other half of the AND.
    floored = threshold_sweep(forecasts, [0.40], min_confidence="MEDIUM_LOW")[0]
    assert floored.n == 10 and floored.hits == 8
    # MEDIUM is what production demands today, and it selects nothing.
    assert threshold_sweep(forecasts, [0.40], min_confidence="MEDIUM")[0].n == 0


def test_suggested_thresholds_never_exceed_what_the_arms_produce():
    forecasts = _forecasts([(0.43, "LOW", 1), (0.56, "MEDIUM_LOW", 0)])
    candidates = suggested_thresholds(forecasts, width=0.02)
    assert candidates[0] == 0.43 and max(candidates) <= 0.56
    assert suggested_thresholds([]) == []


def test_brier_ignores_unresolved_and_scores_a_coin_at_quarter():
    assert brier_score(_forecasts([(0.5, "LOW", None)])) is None
    assert brier_score(_run(0.5, "LOW", 1, 1)) == 0.25
    confident_and_right = brier_score(_run(0.9, "HIGH", 1, 0))
    assert confident_and_right < 0.25


def _scored(probability, reference, hits, misses):
    return (_forecasts([(probability, "LOW", 1)] * hits)
            + _forecasts([(probability, "LOW", 0)] * misses))


def _with_reference(forecasts, reference):
    from auramaur.evaluation.etf_calibration import Forecast
    return [Forecast(f.probability, f.confidence, f.actual_outcome, reference)
            for f in forecasts]


def test_trading_stays_locked_until_enough_forecasts_resolve():
    """Forecasts are free and resolve in five sessions; trading is not free.
    The gate is demonstrated skill, not elapsed time."""
    from auramaur.evaluation.etf_calibration import clearance

    thin = _with_reference(_scored(0.60, 0.50, 30, 10), 0.50)
    verdict = clearance(thin, min_resolved=100)
    assert not verdict.cleared
    assert "40/100" in verdict.reason


def test_beating_the_benchmark_opens_the_gate_and_luck_does_not():
    from auramaur.evaluation.etf_calibration import clearance

    # A real 10pp Brier edge still needs SIZE. At n=200 the mean edge is
    # +0.010 but its lower bound is -0.0036, so the gate correctly refuses --
    # ~370 resolutions are needed before that edge is distinguishable. This is
    # the gate working, not a bug, and at ~125 forecasts/week it is ~3 weeks.
    borderline = _with_reference(_scored(0.60, 0.50, 120, 80), 0.50)
    assert clearance(borderline, min_resolved=100).brier_edge > 0
    assert not clearance(borderline, min_resolved=100).cleared

    skilled = _with_reference(_scored(0.60, 0.50, 360, 240), 0.50)
    verdict = clearance(skilled, min_resolved=100)
    assert verdict.cleared and verdict.brier_edge > 0
    assert verdict.brier_edge_lo > 0

    # Same hit rate, but the forecast adds nothing over the benchmark because
    # it IS the benchmark. No edge, so no clearance.
    noise = _with_reference(_scored(0.50, 0.50, 360, 240), 0.50)
    assert not clearance(noise, min_resolved=100).cleared

    # A forecast that is confidently WRONG must never clear.
    wrong = _with_reference(_scored(0.60, 0.50, 180, 420), 0.50)
    bad = clearance(wrong, min_resolved=100)
    assert not bad.cleared and bad.brier_edge < 0


def test_clearance_reports_conviction_because_the_economic_gate_needs_it():
    """An arm can be perfectly calibrated and still unable to trade anything:
    0.02 of conviction does not clear costs on any instrument in reach. That
    is a different failure from being wrong, and must be visible."""
    from auramaur.evaluation.etf_calibration import clearance

    timid = _with_reference(_scored(0.52, 0.50, 120, 80), 0.50)
    verdict = clearance(timid, min_resolved=100)
    assert verdict.max_conviction == pytest.approx(0.02)
    bold = _with_reference(_scored(0.60, 0.50, 120, 80), 0.50)
    assert clearance(bold, min_resolved=100).max_conviction == pytest.approx(0.10)
