"""Deribit IV source: instrument parsing, ATM term structure, total-variance
interpolation, and the pillar's source-then-fallback behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from auramaur.data_sources.deribit_iv import (
    DeribitIVSource,
    atm_term_structure,
    interp_sigma,
    parse_instrument,
)

NOW = datetime(2026, 7, 18, 0, 0, tzinfo=timezone.utc)


def test_parse_instrument_variants():
    exp, strike, kind = parse_instrument("ETH-26SEP26-1600-C")
    assert (exp.year, exp.month, exp.day, exp.hour) == (2026, 9, 26, 8)
    assert (strike, kind) == (1600.0, "C")
    assert parse_instrument("BTC-2JAN27-100000-P")[1] == 100000.0
    assert parse_instrument("ETH_USDC-PERPETUAL") is None
    assert parse_instrument("") is None


def _summ(name, iv):
    return {"instrument_name": name, "mark_iv": iv}


def test_atm_term_structure_picks_nearest_strike_and_averages():
    rows = [
        _summ("ETH-25JUL26-1800-C", 50.0), _summ("ETH-25JUL26-1800-P", 54.0),
        _summ("ETH-25JUL26-2400-C", 80.0),   # far strike ignored for ATM
        _summ("ETH-25SEP26-1900-C", 62.0),
        _summ("ETH-25SEP26-1900-P", None),   # missing IV skipped
        _summ("ETH-18JUL26-1800-C", 95.0),   # ~0.3d tenor -> dropped? 18JUL 08:00 vs NOW 18JUL 00:00 = 8h < 12h -> dropped
    ]
    term = atm_term_structure(rows, index_price=1841.0, now=NOW)
    assert len(term) == 2
    t0, s0 = term[0]
    assert s0 == pytest.approx(0.52)         # mean(50, 54)/100
    assert term[1][1] == pytest.approx(0.62)


def test_interp_linear_in_total_variance_and_clamps():
    term = [(0.1, 0.50), (0.5, 0.70)]
    # Below/above range clamps flat.
    assert interp_sigma(term, 0.01) == pytest.approx(0.50)
    assert interp_sigma(term, 2.0) == pytest.approx(0.70)
    # Midpoint in total variance: w = 0.025 + (0.245-0.025)*0.5 = 0.135 at t=0.3
    assert interp_sigma(term, 0.3) == pytest.approx((0.135 / 0.3) ** 0.5)
    assert interp_sigma([], 0.3) is None


@pytest.mark.asyncio
async def test_source_uncovered_asset_and_failure_return_none():
    src = DeribitIVSource({"ethereum": "ETH"})
    assert await src.term_sigma("dogecoin", 0.3) is None   # uncovered
    src._fetch_json = AsyncMock(side_effect=RuntimeError("down"))
    assert await src.term_sigma("ethereum", 0.3) is None   # API failure


@pytest.mark.asyncio
async def test_source_caches_within_ttl():
    src = DeribitIVSource({"ethereum": "ETH"}, ttl_seconds=3600)
    calls = {"n": 0}

    async def fake(url):
        calls["n"] += 1
        if "index_price" in url:
            return {"result": {"index_price": 1841.0}}
        return {"result": [_summ("ETH-25SEP26-1900-C", 62.0)]}

    src._fetch_json = fake
    s1 = await src.term_sigma("ethereum", 0.2)
    s2 = await src.term_sigma("ethereum", 0.2)
    assert s1 == s2 == pytest.approx(0.62)
    assert calls["n"] == 2                    # one index + one book, then cache


@pytest.mark.asyncio
async def test_pillar_uses_iv_then_falls_back(tmp_path):
    from tests.test_vol_anchor import _market, _pillar, _future

    m = _market("m1", f"Will Ethereum reach $3,000 by {_future(175)}?", 0.10)
    pillar, db, _ = await _pillar(tmp_path, [m])
    try:
        # IV source present and answering -> deribit sigma drives pricing.
        pillar._iv_source = DeribitIVSource({"ethereum": "ETH"})
        pillar._iv_source.term_sigma = AsyncMock(return_value=0.80)
        assert await pillar.run_once() == 1    # sigma 0.80 -> fair well above crowd
        # Source answering None -> the calibrated blend takes over (no crash).
        pillar._iv_source.term_sigma = AsyncMock(return_value=None)
        n2 = await pillar.run_once()
        assert n2 == 0                         # market now claimed; no error path
    finally:
        await db.close()
