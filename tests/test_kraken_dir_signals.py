"""Tier-1 Kraken directional intelligence:

- the calibration feedback loop (_track_dir_signals): one outstanding bet per
  pair, opened from the cached LLM view and resolved at horizon (spot vs the
  reference price) into calibration.
- the conviction-weighted budget multiplier (_conviction_mult), which can only
  shrink the crypto ceiling (hold more USDC), never grow it.
"""

from auramaur.components import Components
import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

from auramaur.db.database import Database
from auramaur.treasury.kraken_pillar import KrakenPillar


def _conv_kcfg(*, enabled=True, min_mult=0.34):
    return SimpleNamespace(
        directional_llm_horizon_days=3,
        directional_llm_min_confidence="MEDIUM",
        directional_conviction_budget_enabled=enabled,
        directional_conviction_min_mult=min_mult,
    )


def _pillar(db, calib, kcfg, price):
    settings = SimpleNamespace(kraken=kcfg, is_live=False)
    k = SimpleNamespace(get_price=AsyncMock(return_value=price))
    bot = SimpleNamespace(_components=Components({"db": db, "calibration": calib}))
    return KrakenPillar(settings, k, bot=bot)


def _pillar_nodb(kcfg, price=100.0):
    settings = SimpleNamespace(kraken=kcfg, is_live=False)
    k = SimpleNamespace(get_price=AsyncMock(return_value=price))
    return KrakenPillar(settings, k, bot=None)


# ---------------------------------------------------------------------------
# Feedback loop: open
# ---------------------------------------------------------------------------

def test_track_opens_bet_and_records_prediction():
    """A warm LLM view with no outstanding bet snapshots a bet + records the
    prediction under the isolated 'kraken_spot' category."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        calib = SimpleNamespace(record_prediction=AsyncMock(), record_resolution=AsyncMock())
        p = _pillar(db, calib, _conv_kcfg(), price=100.0)
        p._llm_views = {"XBTUSDC": (time.monotonic(), 0.72, "HIGH")}

        await p._track_dir_signals(["XBTUSDC"])

        row = await db.fetchone(
            "SELECT prob, ref_price, due_at FROM kraken_dir_signals WHERE pair='XBTUSDC'")
        assert row is not None
        assert abs(row["prob"] - 0.72) < 1e-9
        assert abs(row["ref_price"] - 100.0) < 1e-9
        calib.record_prediction.assert_awaited_once()
        args = calib.record_prediction.await_args.args
        assert args[0] == "kraken-dir:XBTUSDC"
        assert args[2] == "kraken_spot"   # isolated from Polymarket 'crypto'
        await db.close()

    asyncio.run(run())


def test_track_does_not_open_without_a_warm_view():
    """No cached view -> nothing is tracked (no LLM call is forced)."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        calib = SimpleNamespace(record_prediction=AsyncMock(), record_resolution=AsyncMock())
        p = _pillar(db, calib, _conv_kcfg(), price=100.0)
        p._llm_views = {}

        await p._track_dir_signals(["XBTUSDC"])

        row = await db.fetchone("SELECT COUNT(*) c FROM kraken_dir_signals")
        assert row["c"] == 0
        calib.record_prediction.assert_not_awaited()
        await db.close()

    asyncio.run(run())


def test_track_keeps_single_outstanding_bet_per_pair():
    """A pair with an open bet is not re-opened on the next cycle."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        calib = SimpleNamespace(record_prediction=AsyncMock(), record_resolution=AsyncMock())
        p = _pillar(db, calib, _conv_kcfg(), price=100.0)
        p._llm_views = {"XBTUSDC": (time.monotonic(), 0.8, "HIGH")}

        await p._track_dir_signals(["XBTUSDC"])   # opens
        await p._track_dir_signals(["XBTUSDC"])   # must NOT re-open (still in future)

        row = await db.fetchone("SELECT COUNT(*) c FROM kraken_dir_signals WHERE pair='XBTUSDC'")
        assert row["c"] == 1
        assert calib.record_prediction.await_count == 1
        await db.close()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Feedback loop: resolve at horizon
# ---------------------------------------------------------------------------

def test_track_resolves_due_bet_went_up():
    """A past-due bet resolves to went_up=True when spot > ref_price, then is
    removed from the tracking table."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        await db.execute(
            "INSERT INTO kraken_dir_signals (pair, prob, ref_price, opened_at, due_at) "
            "VALUES ('XBTUSDC', 0.7, 100.0, datetime('now','-4 days'), datetime('now','-1 day'))")
        await db.commit()
        calib = SimpleNamespace(record_prediction=AsyncMock(), record_resolution=AsyncMock())
        p = _pillar(db, calib, _conv_kcfg(), price=120.0)   # spot now above ref
        p._llm_views = {}   # no re-open

        await p._track_dir_signals(["XBTUSDC"])

        calib.record_resolution.assert_awaited_once_with("kraken-dir:XBTUSDC", True)
        gone = await db.fetchone("SELECT * FROM kraken_dir_signals WHERE pair='XBTUSDC'")
        assert gone is None
        await db.close()

    asyncio.run(run())


def test_track_resolves_due_bet_went_down():
    """spot < ref_price -> went_up=False."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        await db.execute(
            "INSERT INTO kraken_dir_signals (pair, prob, ref_price, opened_at, due_at) "
            "VALUES ('ETHUSDC', 0.65, 100.0, datetime('now','-4 days'), datetime('now','-1 day'))")
        await db.commit()
        calib = SimpleNamespace(record_prediction=AsyncMock(), record_resolution=AsyncMock())
        p = _pillar(db, calib, _conv_kcfg(), price=80.0)
        p._llm_views = {}

        await p._track_dir_signals(["ETHUSDC"])

        calib.record_resolution.assert_awaited_once_with("kraken-dir:ETHUSDC", False)
        await db.close()

    asyncio.run(run())


def test_track_keeps_due_bet_when_unpriceable():
    """If spot can't be fetched, a due bet is retained for a later retry rather
    than dropped unresolved."""
    async def run():
        db = Database(":memory:")
        await db.connect()
        await db.execute(
            "INSERT INTO kraken_dir_signals (pair, prob, ref_price, opened_at, due_at) "
            "VALUES ('SOLUSDC', 0.7, 50.0, datetime('now','-4 days'), datetime('now','-1 day'))")
        await db.commit()
        calib = SimpleNamespace(record_prediction=AsyncMock(), record_resolution=AsyncMock())
        p = _pillar(db, calib, _conv_kcfg(), price=None)   # unpriceable
        p._llm_views = {}

        await p._track_dir_signals(["SOLUSDC"])

        calib.record_resolution.assert_not_awaited()
        still = await db.fetchone("SELECT COUNT(*) c FROM kraken_dir_signals WHERE pair='SOLUSDC'")
        assert still["c"] == 1
        await db.close()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Conviction-weighted budget multiplier
# ---------------------------------------------------------------------------

def test_conviction_mult_disabled_is_one():
    p = _pillar_nodb(_conv_kcfg(enabled=False))
    assert p._conviction_mult(["XBTUSDC", "ETHUSDC"]) == 1.0


def test_conviction_mult_cold_is_min():
    """Enabled but no warm views -> conservative floor (hold the most USDC)."""
    p = _pillar_nodb(_conv_kcfg(enabled=True, min_mult=0.34))
    p._llm_views = {}
    assert abs(p._conviction_mult(["XBTUSDC"]) - 0.34) < 1e-9


def test_conviction_mult_blends_views():
    """One max-bullish + one neutral over two eligible pairs -> conviction 0.5,
    mult = 0.34 + 0.66*0.5 = 0.67."""
    p = _pillar_nodb(_conv_kcfg(enabled=True, min_mult=0.34))
    p._llm_views = {
        "XBTUSDC": (time.monotonic(), 1.0, "HIGH"),   # score 1.0
        "ETHUSDC": (time.monotonic(), 0.5, "HIGH"),   # score 0.0
    }
    assert abs(p._conviction_mult(["XBTUSDC", "ETHUSDC"]) - 0.67) < 1e-6


def test_conviction_mult_ignores_low_confidence():
    """A bullish prob below the confidence floor contributes 0."""
    p = _pillar_nodb(_conv_kcfg(enabled=True, min_mult=0.34))
    p._llm_views = {"XBTUSDC": (time.monotonic(), 0.9, "LOW")}
    assert abs(p._conviction_mult(["XBTUSDC"]) - 0.34) < 1e-9


def test_conviction_mult_capped_at_one():
    """Full bullish conviction reaches but never exceeds the static ceiling."""
    p = _pillar_nodb(_conv_kcfg(enabled=True, min_mult=0.34))
    p._llm_views = {"XBTUSDC": (time.monotonic(), 1.0, "HIGH")}
    assert p._conviction_mult(["XBTUSDC"]) == 1.0
