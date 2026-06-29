"""Tests for the leaderboard intelligence feed — the pure archetype classifier
and row parser (no network)."""

from __future__ import annotations

from auramaur.data_sources.polymarket_leaderboard import (
    Leader,
    classify_archetype,
    parse_leader,
)


def test_classify_archetype_by_pnl_vol_ratio():
    # directional: huge edge per dollar (Theo4 ~51%)
    assert classify_archetype(22.0e6, 43.0e6) == "directional"
    # mm/arb: thin margin x gigantic turnover (swisstony ~1.1%)
    assert classify_archetype(14.6e6, 1298e6) == "mm_arb"
    # mixed: between the thresholds (kch123 ~3.9%)
    assert classify_archetype(11.4e6, 293.7e6) == "mixed"
    # degenerate
    assert classify_archetype(100, 0) == "unknown"


def test_classify_threshold_boundaries():
    assert classify_archetype(10, 100) == "directional"   # exactly 10%
    assert classify_archetype(9.9, 100) == "mixed"         # just under 10%
    assert classify_archetype(2.9, 100) == "mm_arb"        # under 3%
    assert classify_archetype(3.0, 100) == "mixed"         # exactly 3% -> mixed


def test_parse_leader_classifies_and_handles_malformed():
    ld = parse_leader({"rank": "1", "proxyWallet": "0xabc", "userName": "Theo4",
                       "pnl": 22.0e6, "vol": 43.0e6})
    assert isinstance(ld, Leader)
    assert ld.archetype == "directional" and ld.name == "Theo4"
    assert abs(ld.pnl_vol_ratio - 0.512) < 0.01

    # missing wallet -> dropped
    assert parse_leader({"rank": "2", "pnl": 1, "vol": 2}) is None
    # non-numeric -> dropped, no exception
    assert parse_leader({"proxyWallet": "0xz", "pnl": "NaNish", "vol": "x"}) is None
