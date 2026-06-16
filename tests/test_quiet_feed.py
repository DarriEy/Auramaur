"""Quiet feed: noise aggregation, source-aware unfilled lines, ticker
throttle, and the loop watchdog.

The display used to print every event identically — an expired MM quote
looked like a failed entry, the ticker line printed every 90s, and a frozen
event loop produced indistinguishable silence. These tests pin the new
contract: trades and failures print, routine churn aggregates, and the
watchdog thread speaks when the loop can't.
"""

from __future__ import annotations

import pytest

from auramaur.exchange.models import Order, OrderSide
from auramaur.monitoring import display
from auramaur.monitoring.feed import LoopWatchdog, NoiseAggregator


@pytest.fixture(autouse=True)
def _reset_display_state():
    """Isolate module-level counters/throttle between tests."""
    display.noise._counts.clear()
    display.noise.interval_seconds = 3600.0
    display._ticker_state.update(
        {"last_print": 0.0, "pnl": None, "positions": None, "mode": None}
    )
    yield
    display.noise._counts.clear()


# ---------------------------------------------------------------------------
# NoiseAggregator
# ---------------------------------------------------------------------------

def test_noise_flush_builds_labeled_summary():
    agg = NoiseAggregator()
    agg.bump("claude_call", 12)
    agg.bump("mm_quote", 24)
    agg.bump("mm_expired", 3)
    line = agg.flush()
    assert line == "12 claude calls | 24 mm quotes | 3 mm expiries"
    # Counters reset after flush
    assert agg.flush() is None


def test_noise_flush_empty_returns_none():
    assert NoiseAggregator().flush() is None


def test_noise_unknown_keys_still_render():
    agg = NoiseAggregator()
    agg.bump("mystery_event", 2)
    assert agg.flush() == "2 mystery_event"


# ---------------------------------------------------------------------------
# Source-aware ORDER UNFILLED
# ---------------------------------------------------------------------------

def test_mm_quote_expiry_is_silent_and_counted():
    with display.console.capture() as cap:
        display.show_order_unfilled(
            "BUY", 10.0, 0.17, 145.0, exchange="polymarket",
            market_id="629721", source="market_maker",
        )
    assert cap.get() == ""
    assert display.noise._counts.get("mm_expired") == 1


def test_strategy_unfilled_prints_with_source():
    with display.console.capture() as cap:
        display.show_order_unfilled(
            "BUY", 20.1, 0.09, 124.0, exchange="polymarket",
            market_id="2333694", source="news_speed",
        )
    out = cap.get()
    assert "ORDER UNFILLED" in out
    assert "news_speed" in out
    assert "2333694" in out


def test_routine_events_are_silent_and_counted():
    with display.console.capture() as cap:
        display.show_claude_thinking("kraken-dir:XBTUS")
        display.show_cache_hit()
        display.show_mm_quote("629721", 0.17, 0.19, 199)
        display.show_evidence(13, {"web": 3})
    assert cap.get() == ""
    assert display.noise._counts == {
        "claude_call": 1, "cache_hit": 1, "mm_quote": 1, "evidence": 1,
    }


def test_zero_evidence_still_warns():
    with display.console.capture() as cap:
        display.show_evidence(0, {})
    assert "No evidence found" in cap.get()


def test_zero_candidate_scan_is_silent():
    with display.console.capture() as cap:
        display.show_scan_results(300, 0, 59, exchange="kalshi")
        display.show_scan_results(100, 1, 20, exchange="polymarket")
    out = cap.get()
    assert "kalshi" not in out  # 0-candidate scan aggregated
    assert "polymarket" in out  # productive scan still prints


# ---------------------------------------------------------------------------
# Ticker throttle
# ---------------------------------------------------------------------------

def test_ticker_prints_first_then_suppresses_unchanged():
    with display.console.capture() as cap:
        display.show_portfolio(100.00, -10.00, 50, 0.0, schedule_mode="off_peak")
        display.show_portfolio(100.00, -10.00, 50, 0.0, schedule_mode="off_peak")
    out = cap.get()
    assert out.count("P&L") == 1


def test_ticker_prints_on_material_change():
    with display.console.capture() as cap:
        display.show_portfolio(100.00, -10.00, 50, 0.0)
        display.show_portfolio(100.00, -4.00, 50, 0.0)   # P&L move
        display.show_portfolio(100.00, -4.00, 51, 0.0)   # position count change
    assert cap.get().count("P&L") == 3


def test_ticker_shows_reserved_cash():
    with display.console.capture() as cap:
        display.show_portfolio(100.00, -10.00, 50, 0.0, reserved=20.00)
    out = cap.get()
    assert "$100.00" in out
    assert "$20.00" in out
    assert "in orders" in out


def test_ticker_flushes_noise_summary_when_due():
    display.noise.interval_seconds = 0.0  # always due
    display.noise.bump("mm_quote", 24)
    with display.console.capture() as cap:
        display.show_portfolio(100.00, -10.00, 50, 0.0)
    assert "24 mm quotes" in cap.get()


# ---------------------------------------------------------------------------
# Order.source threading
# ---------------------------------------------------------------------------

def test_order_source_defaults_empty():
    o = Order(market_id="m", side=OrderSide.BUY, size=10.0, price=0.5)
    assert o.source == ""


def test_order_source_settable():
    o = Order(market_id="m", side=OrderSide.BUY, size=10.0, price=0.5,
              source="market_maker")
    assert o.source == "market_maker"


# ---------------------------------------------------------------------------
# LoopWatchdog
# ---------------------------------------------------------------------------

def test_watchdog_alerts_on_stall_and_recovery():
    alerts: list[str] = []
    dog = LoopWatchdog(stall_seconds=10.0, alert=alerts.append)

    dog.beat()
    dog._check()
    assert alerts == []  # fresh beat, no alarm

    dog._last_beat -= 100.0  # simulate 100s of loop silence
    dog._check()
    assert len(alerts) == 1
    assert "STALLED" in alerts[0]

    dog._check()
    assert len(alerts) == 1  # no repeat alarm while still stalled

    dog.beat()  # loop wakes up
    dog._check()
    assert len(alerts) == 2
    assert "recovered" in alerts[1]
