"""Tests for the LLM/news-driven Kraken directional signal + gate logic."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from auramaur.treasury.kraken_pillar import KrakenPillar


def _settings(**kw):
    kr = SimpleNamespace(
        directional_llm_min_confidence="MEDIUM",
        directional_stop_loss_pct=12.0,
        directional_llm_exit_prob=0.45,
        directional_llm_min_prob=0.60,
        directional_llm_refresh_hours=8.0,
        directional_llm_horizon_days=3,
        directional_fee_pct=0.26,
    )
    for k, v in kw.items():
        setattr(kr, k, v)
    return SimpleNamespace(kraken=kr, is_live=False)


def _pillar(bot=None):
    return KrakenPillar(_settings(), MagicMock(), bot=bot)


def test_conf_floor():
    p = _pillar()
    assert p._conf_ok("HIGH")
    assert p._conf_ok("MEDIUM_HIGH")
    assert p._conf_ok("MEDIUM")
    assert not p._conf_ok("MEDIUM_LOW")
    assert not p._conf_ok("LOW")


def test_llm_exit_reason():
    p = _pillar()

    async def run():
        # Hard stop fires regardless of view.
        assert await p._llm_exit_reason("XBTUSDC", -15.0, p._s.kraken) == "stop_loss"
        # Bearish view (below exit_prob) exits.
        p._llm_view = AsyncMock(return_value=(0.40, "HIGH"))
        assert await p._llm_exit_reason("XBTUSDC", 1.0, p._s.kraken) == "llm_bearish"
        # Still-bullish view holds.
        p._llm_view = AsyncMock(return_value=(0.55, "HIGH"))
        assert await p._llm_exit_reason("XBTUSDC", 1.0, p._s.kraken) is None
        # Missing view holds (don't churn on a transient gap).
        p._llm_view = AsyncMock(return_value=None)
        assert await p._llm_exit_reason("XBTUSDC", 1.0, p._s.kraken) is None

    asyncio.run(run())


def test_llm_view_returns_and_throttles():
    """_llm_view returns (prob, conf) and serves cached reads within the refresh
    window. Calibration recording is NOT done here — it is owned by
    _track_dir_signals (see test_kraken_dir_signals.py) so the feedback loop
    records exactly one horizon-resolved bet per pair."""
    agg = MagicMock()
    agg.gather = AsyncMock(return_value=[])
    analysis = SimpleNamespace(probability=0.71, confidence="HIGH", skipped_reason=None)
    analyzer = MagicMock()
    analyzer.analyze = AsyncMock(return_value=analysis)
    calib = MagicMock()
    calib.record_prediction = AsyncMock()
    bot = SimpleNamespace(_components={
        "aggregator": agg, "analyzer": analyzer, "cache": None, "calibration": calib,
    })
    p = _pillar(bot=bot)

    async def run():
        v = await p._llm_view("SOLUSDC")
        assert v == (0.71, "HIGH")
        # _llm_view itself no longer records to calibration.
        calib.record_prediction.assert_not_awaited()
        assert analyzer.analyze.await_count == 1
        # Second call within refresh window → served from cache, no new LLM call.
        v2 = await p._llm_view("SOLUSDC")
        assert v2 == (0.71, "HIGH")
        assert analyzer.analyze.await_count == 1

    asyncio.run(run())


def test_llm_view_unknown_pair_returns_none():
    bot = SimpleNamespace(_components={})
    p = _pillar(bot=bot)

    async def run():
        assert await p._llm_view("NOTAPAIRUSDC") is None

    asyncio.run(run())


def test_llm_view_skipped_analysis_returns_none():
    agg = MagicMock()
    agg.gather = AsyncMock(return_value=[])
    analysis = SimpleNamespace(probability=0.5, confidence="LOW", skipped_reason="thin evidence")
    analyzer = MagicMock()
    analyzer.analyze = AsyncMock(return_value=analysis)
    bot = SimpleNamespace(_components={
        "aggregator": agg, "analyzer": analyzer, "cache": None, "calibration": None,
    })
    p = _pillar(bot=bot)

    async def run():
        assert await p._llm_view("XBTUSDC") is None

    asyncio.run(run())


if __name__ == "__main__":
    test_conf_floor()
    test_llm_exit_reason()
    test_llm_view_returns_and_throttles()
    test_llm_view_unknown_pair_returns_none()
    test_llm_view_skipped_analysis_returns_none()
    print("ok")
