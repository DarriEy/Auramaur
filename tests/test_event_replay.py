"""The Stage-1 earnings skeleton must implement the FROZEN spec exactly
(docs/ibkr-event-driven-design.md, 2026-08-03): sign-only, long-only,
timing-ruled entries, N-session exits, costs both ways, nothing dropped
silently. A replay that drifts from its spec measures nothing."""

from auramaur.backtest.event_replay import (
    EarningsEvent,
    build_event_trips,
    entry_session_index,
)

DATES = [f"2025-01-{d:02d}" for d in (2, 3, 6, 7, 8, 9, 10, 13, 14, 15)]
FLAT = [(d, 100.0) for d in DATES]


def _ev(sign=1, date="2025-01-03", timing="BMO", key="SHEL.L"):
    return EarningsEvent(home_listing=key, report_date=date,
                         timing_class=timing, surprise_sign=sign)


def test_entry_timing_rule():
    # BMO on a session day -> that day; on a holiday -> next session.
    assert entry_session_index(DATES, "2025-01-03", "BMO") == 1
    assert entry_session_index(DATES, "2025-01-04", "BMO") == 2
    # AMC/MID/UNKNOWN -> strictly the next session.
    assert entry_session_index(DATES, "2025-01-03", "AMC") == 2
    assert entry_session_index(DATES, "2025-01-03", "UNKNOWN") == 2
    # Nothing after the report date -> no entry.
    assert entry_session_index(DATES, "2025-01-15", "AMC") is None


def test_long_only_primary_skips_negatives_and_zeros_visibly():
    events = [_ev(1), _ev(-1), _ev(0)]
    trips, cov = build_event_trips(
        events, {"SHEL.L": FLAT}, {"SHEL.L": "USD"},
        assumed_spread_bps=40.0, slippage_bps=2.0)
    assert len(trips) == 1
    assert cov["eligible"] == 1
    assert cov["negative_skipped"] == 1
    assert cov["zero_surprise"] == 1


def test_flat_prices_lose_exactly_the_costs():
    """On flat closes the trip must lose spread+slippage+commissions and
    nothing else — the arithmetic that killed momentum, charged honestly."""
    trips, _ = build_event_trips(
        [_ev(1)], {"SHEL.L": FLAT}, {"SHEL.L": "USD"},
        assumed_spread_bps=40.0, slippage_bps=2.0)
    t = trips[0]
    assert t.entry_price > 100.0 > t.exit_price  # paid ask, received bid
    assert t.gross_usd < 0
    assert t.commission_usd >= 2.0  # both legs, $1 minimum each
    # ~$2,000 at 40bps spread + 2bps slippage/leg ≈ -$8.4 gross, < $15 all-in
    assert -15.0 < t.net_usd < -5.0


def test_gbx_fx_sizes_the_notional_correctly():
    """A 3,300-GBX (£33) name must buy ~48 shares for $2,000, not 0.6 —
    the pence trap the spec's static-FX table exists to avoid."""
    bars = [(d, 3300.0) for d in DATES]
    trips, _ = build_event_trips(
        [_ev(1)], {"SHEL.L": bars}, {"SHEL.L": "GBP"},
        assumed_spread_bps=40.0, slippage_bps=2.0)
    t = trips[0]
    notional_usd = t.entry_price * t.quantity * 0.0127
    assert abs(notional_usd - 2000.0) < 1.0
    assert 40 < t.quantity < 55


def test_short_side_profits_when_price_falls():
    falling = [(d, 100.0 - i) for i, d in enumerate(DATES)]
    trips, _ = build_event_trips(
        [_ev(-1)], {"SHEL.L": falling}, {"SHEL.L": "USD"},
        assumed_spread_bps=40.0, slippage_bps=2.0, long_only=False)
    assert len(trips) == 1
    assert trips[0].gross_usd > 0


def test_event_without_full_hold_window_is_counted_not_traded():
    trips, cov = build_event_trips(
        [_ev(1, date="2025-01-13")], {"SHEL.L": FLAT}, {"SHEL.L": "USD"},
        assumed_spread_bps=40.0, slippage_bps=2.0)
    assert not trips
    assert cov["no_exit_session"] == 1


def test_missing_bars_are_coverage_loss_not_silence():
    trips, cov = build_event_trips(
        [_ev(1, key="7203.T")], {"SHEL.L": FLAT}, {"SHEL.L": "USD"},
        assumed_spread_bps=40.0, slippage_bps=2.0)
    assert not trips
    assert cov["no_bars"] == 1
