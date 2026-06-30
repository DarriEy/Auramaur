"""Tests for the informed-flow (abnormal-trade-size) detector — the data layer
for the Kalshi ATS follower. The detector is pure, so these lock the logic
without any network."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from auramaur.strategy.informed_flow import (
    InformedFlowSignal,
    KalshiTradeTape,
    detect_informed_flow,
)


def _t(count, side):
    # Kalshi emits the per-trade size as count_fp (a fixed-point STRING), e.g.
    # "60.00" — mirror that so the detector tests exercise the real field.
    return {"count_fp": f"{count}", "taker_side": side}


def test_reads_count_fp_string_and_legacy_count():
    """_size handles the live count_fp string AND the legacy count field."""
    from auramaur.strategy.informed_flow import _size
    assert abs(_size({"count_fp": "225.25"}) - 225.25) < 1e-9
    assert _size({"count": 50}) == 50.0           # legacy fallback
    assert _size({"count_fp": "nope"}) == 0.0      # malformed -> 0, no raise


def _normal(n, size, side="yes"):
    return [_t(size, side) for _ in range(n)]


def test_thin_tape_gives_no_signal():
    """Below min_sample, a big print is noise, not information."""
    sig = detect_informed_flow([_t(1000, "yes")] * 3, min_sample=20)
    assert sig == InformedFlowSignal(False, None, 0, 0.0, 0.0, 0)


def test_no_abnormal_trades_no_signal():
    """A tape of uniform small trades has no abnormal flow -> the no-signal
    sentinel (nothing actionable to report)."""
    sig = detect_informed_flow(_normal(40, 5, "yes"), min_sample=20, size_mult=3.0)
    assert sig.has_signal is False
    assert sig.abnormal_count == 0
    assert sig.informed_side is None


def test_detects_abnormal_yes_flow_and_infers_side():
    # 30 small (size 5) trades + 3 large (size 50 = 10x baseline) YES takers.
    trades = _normal(30, 5, "yes") + _normal(30, 5, "no") + _normal(3, 50, "yes")
    sig = detect_informed_flow(trades, min_sample=20, size_mult=3.0, min_dominance=0.6)
    assert sig.has_signal is True
    assert sig.informed_side == "yes"
    assert sig.abnormal_count == 3
    assert sig.signal_volume == 150  # 3 x 50


def test_two_sided_large_flow_is_inconclusive():
    """Large prints on BOTH sides (a wash) -> no directional signal."""
    trades = _normal(40, 5, "yes") + _normal(3, 50, "yes") + _normal(3, 50, "no")
    sig = detect_informed_flow(trades, min_sample=20, size_mult=3.0, min_dominance=0.6)
    assert sig.has_signal is False
    assert sig.informed_side is None
    assert sig.abnormal_count == 6  # counted, but not followed


def test_ignores_malformed_and_unsided_trades():
    trades = (
        _normal(25, 4, "yes")
        + [{"count": None, "taker_side": "yes"}, {"count": 50}, {"taker_side": "no"}]
        + _normal(3, 40, "no")
    )
    sig = detect_informed_flow(trades, min_sample=20, size_mult=3.0)
    assert sig.has_signal is True
    assert sig.informed_side == "no"
    assert sig.sample == 28  # 25 + 3 well-formed; the 3 malformed dropped


def test_tape_wrapper_fetches_and_detects():
    async def run():
        ex = MagicMock()
        ex.get_trades = AsyncMock(
            return_value=_normal(30, 5, "no") + _normal(4, 60, "no"))
        tape = KalshiTradeTape(ex)
        sig = await tape.informed_flow("KXTEST", limit=100, min_sample=20)
        ex.get_trades.assert_awaited_once_with("KXTEST", limit=100)
        assert sig.has_signal and sig.informed_side == "no"
    asyncio.run(run())


def test_tape_wrapper_fail_soft_on_fetch_error():
    async def run():
        ex = MagicMock()
        ex.get_trades = AsyncMock(side_effect=RuntimeError("kalshi down"))
        tape = KalshiTradeTape(ex)
        sig = await tape.informed_flow("KXTEST")
        assert sig.has_signal is False  # no exception escapes
    asyncio.run(run())
