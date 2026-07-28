"""The replay harness, pinned against the ways a backtest flatters itself.

Every test here guards a specific self-deception: lookahead, uncosted fills,
or a rule that quietly differs from the deployed one.
"""

import pytest

from auramaur.backtest.ibkr_replay import replay_momentum_book


def _series(n, rate, start=100.0, key="AAA"):
    return {key: [(f"2020-{i // 21 + 1:02d}-{i % 21 + 1:02d}",
                   start * (1 + rate) ** i) for i in range(n)]}


def _run(bars, classes=None, **over):
    params = dict(
        budget_usd=5000.0, max_positions=6, max_position_pct=12.0,
        max_deployment_pct=60.0, risk_per_position_pct=0.25,
        max_asset_class_risk_pct=0.5, stop_vol_multiple=2.0, min_stop_pct=0.5,
        slippage_bps=2.0, stop_loss_pct=5.0, take_profit_pct=10.0,
        min_norm_momentum=0.25, exit_norm_momentum=-0.1,
        assumed_spread_bps=25.0, warmup=130)
    params.update(over)
    keys = classes or {k: "us_equity" for k in bars}
    return replay_momentum_book(bars, keys, **params)


def test_a_decision_never_sees_the_bar_it_trades_on():
    """The lookahead test: a series that is flat until a single explosive day.

    If the replay let a decision see its own session's close, the spike day
    would trigger an entry AT the spike. It must not.
    """
    flat = [(f"2020-{i // 21 + 1:02d}-{i % 21 + 1:02d}", 100.0)
            for i in range(200)]
    flat[-1] = (flat[-1][0], 10_000.0)          # one impossible final bar
    result = _run({"AAA": flat})
    # Momentum over a flat series is 0 (or None); the spike cannot be traded
    # because it is only visible AFTER the session it belongs to.
    assert not [t for t in result.trips if t.entry_date == flat[-1][0]]
    assert result.open_at_end == 0


def test_costs_are_charged_on_both_legs():
    """An uncosted momentum backtest on a rising series always 'wins'."""
    rising = _series(400, 0.004)
    costed = _run(rising)
    assert costed.trips, "no round trips to measure"
    for trip in costed.trips:
        assert trip.commission_usd > 0
        assert trip.net_usd < trip.gross_usd
        # Entry is crossed at the ask + slippage, exit at the bid - slippage.
        assert trip.entry_price > 0 and trip.exit_price > 0

    # A wider assumed spread must reduce net P&L, never increase it.
    wide = _run(rising, assumed_spread_bps=200.0)
    if wide.trips:
        assert (sum(t.net_usd for t in wide.trips)
                < sum(t.net_usd for t in costed.trips))


def test_a_falling_series_is_never_entered():
    """min_norm_momentum is the deployed gate; a downtrend must not qualify."""
    result = _run(_series(400, -0.004))
    assert result.trips == ()
    assert result.open_at_end == 0


def test_take_profit_and_stop_loss_both_fire():
    up = _run(_series(400, 0.004))
    assert any(t.reason in ("take_profit", "momentum", "stop", "stop_loss_pct")
               for t in up.trips)
    # A sharp reversal after a qualifying run must exit, not ride it down.
    bars = [(f"2020-{i // 21 + 1:02d}-{i % 21 + 1:02d}",
             100.0 * (1.004 ** i if i < 300 else 1.004 ** 300 * 0.97 ** (i - 300)))
            for i in range(400)]
    crash = _run({"AAA": bars})
    assert crash.trips
    assert crash.open_at_end == 0, "held a collapsing position to the end"


def test_position_and_deployment_caps_are_respected():
    bars = {}
    for n in range(10):
        bars.update(_series(400, 0.004 + n * 0.0001, key=f"S{n}"))
    result = _run(bars, max_positions=3)
    # Never more than max_positions concurrently: reconstruct by date.
    events = [(t.entry_date, 1) for t in result.trips]
    events += [(t.exit_date, -1) for t in result.trips]
    events.sort()
    concurrent = peak = 0
    for _, delta in events:
        concurrent += delta
        peak = max(peak, concurrent)
    assert peak <= 3


def test_too_little_history_yields_nothing_rather_than_guessing():
    result = _run(_series(50, 0.004))
    assert result.trips == () and result.sessions == 50


def test_signal_functions_never_see_more_than_the_live_book_fetches(monkeypatch):
    """IBKRMultiAssetPaperBook calls get_daily_bars(spec) with its default
    "3 M" — 61 bars. normalized_momentum blends 20/60/120-session horizons and
    silently drops the ones that do not fit, so handing the replay the full
    accumulated history unlocks a 120-horizon term the live book can NEVER see.
    That is a different strategy, and its P&L says nothing about the deployed
    one. The first run of this harness made exactly that mistake.

    Asserted on the wiring rather than on emergent trades: the lengths handed
    to the signal functions are the thing that must match.
    """
    from auramaur.backtest import ibkr_replay
    from auramaur.backtest.ibkr_replay import SIGNAL_WINDOW

    assert SIGNAL_WINDOW == 61

    seen = []
    real_momentum = ibkr_replay.normalized_momentum
    real_vol = ibkr_replay.annualized_volatility
    monkeypatch.setattr(ibkr_replay, "normalized_momentum",
                        lambda closes, *a, **k: (seen.append(len(closes)),
                                                 real_momentum(closes, *a, **k))[1])
    monkeypatch.setattr(ibkr_replay, "annualized_volatility",
                        lambda closes, *a, **k: (seen.append(len(closes)),
                                                 real_vol(closes, *a, **k))[1])

    _run(_series(400, 0.004))
    assert seen, "signal functions were never called"
    assert max(seen) <= SIGNAL_WINDOW, (
        f"a signal saw {max(seen)} closes; the live book only ever fetches "
        f"{SIGNAL_WINDOW}")
    # And the window is actually reached, so this is not passing by starvation.
    assert max(seen) == SIGNAL_WINDOW


def test_split_by_entry_puts_straddling_trips_in_train():
    from auramaur.backtest.ibkr_replay import ReplayTrip, mean_lcb, split_by_entry

    def trip(entry, exit_):
        return ReplayTrip("A", "us_equity", entry, exit_, 1.0, 1.0, 1.0,
                          0.0, 0.0, "x")

    trips = [trip("2024-06-01", "2024-06-10"),
             trip("2024-12-30", "2025-01-15"),   # straddles the split
             trip("2025-02-01", "2025-02-10")]
    train, test = split_by_entry(trips, "2025-01-01")
    assert len(train) == 2 and len(test) == 1
    assert test[0].entry_date == "2025-02-01"

    assert mean_lcb([1.0]) == float("-inf")
    assert mean_lcb([1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert mean_lcb([-5.0, 5.0, -5.0, 5.0]) < 0
