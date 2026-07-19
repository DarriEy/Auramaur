from datetime import datetime, timedelta, timezone
import math

import pytest

from auramaur.research.ibkr_signals import (
    CurvePoint,
    PricePoint,
    SignalInputError,
    futures_carry_trend,
    fx_carry_trend,
    relative_value_residual_zscore,
    short_term_mean_reversion,
)


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def prices(values):
    start = NOW - timedelta(days=len(values) - 1)
    return [PricePoint(start + timedelta(days=i), value) for i, value in enumerate(values)]


def test_short_term_mean_reversion_requires_shock_and_uptrend():
    values = [100 + i for i in range(24)] + [118]
    signal = short_term_mean_reversion(
        prices(values), as_of=NOW, return_lookback=10, trend_lookback=20, entry_z=2
    )
    assert signal.direction == 1
    assert signal.score >= 2


def test_short_term_mean_reversion_trend_filter_can_leave_flat():
    values = [100 - i for i in range(24)] + [72]
    signal = short_term_mean_reversion(
        prices(values), as_of=NOW, return_lookback=10, trend_lookback=20, entry_z=2
    )
    assert signal.direction == 0


def test_relative_value_uses_prefit_residual_and_gives_leg_direction():
    xs = [100 + i for i in range(21)]
    ys = [x * (1.001 if i % 2 else 0.999) for i, x in enumerate(xs)]
    ys[-1] = 150
    signal = relative_value_residual_zscore(
        prices(ys), prices(xs), as_of=NOW, hedge_beta=1, lookback=20, entry_z=2
    )
    assert signal.direction == -1
    assert signal.score > 2


def test_relative_value_rejects_misaligned_pair_timestamps():
    left, right = prices(range(100, 106)), prices(range(100, 106))
    right[-1] = PricePoint(right[-1].observed_at - timedelta(hours=1), right[-1].value)
    with pytest.raises(SignalInputError, match="identical timestamps"):
        relative_value_residual_zscore(left, right, as_of=NOW, hedge_beta=1, lookback=5)


def test_futures_carry_and_trend_must_agree():
    curve = [
        CurvePoint(NOW + timedelta(days=30), 110),
        CurvePoint(NOW + timedelta(days=120), 100),
    ]
    assert (
        futures_carry_trend(curve, prices([100, 105, 110]), as_of=NOW, trend_lookback=2).direction
        == 1
    )
    assert (
        futures_carry_trend(curve, prices([110, 105, 100]), as_of=NOW, trend_lookback=2).direction
        == 0
    )


def test_futures_curve_rejects_expired_contract_and_bad_price():
    with pytest.raises(SignalInputError, match="expired"):
        futures_carry_trend(
            [CurvePoint(NOW, 100), CurvePoint(NOW + timedelta(days=30), 101)],
            prices([99, 100]),
            as_of=NOW,
            trend_lookback=1,
        )
    with pytest.raises(SignalInputError, match="positive"):
        futures_carry_trend(
            [
                CurvePoint(NOW + timedelta(days=1), math.nan),
                CurvePoint(NOW + timedelta(days=30), 101),
            ],
            prices([99, 100]),
            as_of=NOW,
            trend_lookback=1,
        )


def test_fx_carry_and_trend_agreement_and_disagreement():
    rising = prices([1.0, 1.05, 1.1])
    assert (
        fx_carry_trend(
            rising, as_of=NOW, base_rate=0.05, quote_rate=0.02, trend_lookback=2
        ).direction
        == 1
    )
    assert (
        fx_carry_trend(
            rising, as_of=NOW, base_rate=0.01, quote_rate=0.04, trend_lookback=2
        ).direction
        == 0
    )


@pytest.mark.parametrize("bad_as_of", [NOW.replace(tzinfo=None)])
def test_point_in_time_validation_rejects_naive_as_of(bad_as_of):
    with pytest.raises(SignalInputError, match="timezone-aware"):
        fx_carry_trend(
            prices([1, 2]),
            as_of=bad_as_of,
            base_rate=0.02,
            quote_rate=0.01,
            trend_lookback=1,
        )


def test_point_in_time_validation_rejects_future_stale_and_unsorted_data():
    with pytest.raises(SignalInputError, match="after as_of"):
        fx_carry_trend(
            [PricePoint(NOW, 1), PricePoint(NOW + timedelta(seconds=1), 1.1)],
            as_of=NOW,
            base_rate=0.02,
            quote_rate=0.01,
            trend_lookback=1,
        )
    with pytest.raises(SignalInputError, match="stale"):
        fx_carry_trend(
            prices([1, 1.1]),
            as_of=NOW + timedelta(days=2),
            base_rate=0.02,
            quote_rate=0.01,
            trend_lookback=1,
            max_age=timedelta(days=1),
        )
    with pytest.raises(SignalInputError, match="strictly increasing"):
        short_term_mean_reversion(
            list(reversed(prices(range(100, 125)))),
            as_of=NOW,
            return_lookback=10,
            trend_lookback=20,
        )
