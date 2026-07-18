"""The Kraken entry gate compares the CALIBRATED view probability.

Six weeks of tracked predictions lived in [0.42, 0.56] while outcomes spread
14%..43% across that band — a raw 0.60 gate could never fire (structurally
untradeable trial) while the signal itself discriminated strongly. The gate
now maps raw reads through a rolling linear fit of the signal's own resolved
calibration record."""

from __future__ import annotations

import pytest

from auramaur.db.database import Database
from auramaur.treasury.kraken_pillar import KrakenPillar
from config.settings import Settings


async def _seed(db, points):
    for i, (p, y) in enumerate(points):
        await db.execute(
            """INSERT INTO calibration (market_id, predicted_prob,
               actual_outcome, resolved_at, created_at, category)
               VALUES (?, ?, ?, datetime('now'), datetime('now'), 'kraken_spot')""",
            (f"kraken-dir:XBTUSDC:{i}", p, y))
    await db.commit()


def _pillar(db):
    s = Settings()
    p = KrakenPillar.__new__(KrakenPillar)
    p._s = s
    p._db = lambda: db
    return p


@pytest.mark.asyncio
async def test_identity_until_enough_resolved_points(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    try:
        p = _pillar(db)
        await _seed(db, [(0.5, 1)] * 10)          # only 10 points
        assert await p._calibrated_prob(0.52) == pytest.approx(0.52)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_fit_amplifies_the_compressed_signal(tmp_path):
    """The measured shape: predictions 0.44..0.54, outcomes 10%..60% —
    the fit must map the strongest raw reads ABOVE the 0.60 gate and the
    weak ones far below it."""
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    try:
        p = _pillar(db)
        pts = []
        # 30 reads at 0.44 resolving up 10%; 30 at 0.49 up 30%; 30 at 0.54 up 60%
        pts += [(0.44, 1)] * 3 + [(0.44, 0)] * 27
        pts += [(0.49, 1)] * 9 + [(0.49, 0)] * 21
        pts += [(0.54, 1)] * 18 + [(0.54, 0)] * 12
        await _seed(db, pts)
        strong = await p._calibrated_prob(0.56)
        weak = await p._calibrated_prob(0.46)
        assert strong > 0.60, f"strong read must clear the gate, got {strong}"
        assert weak < 0.30
        # Monotone and clamped.
        assert await p._calibrated_prob(0.99) <= 0.99
        assert await p._calibrated_prob(0.01) >= 0.01
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_fit_cached_and_slope_clamped(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    try:
        p = _pillar(db)
        # Degenerate: outcomes perfectly split at two nearly-identical preds
        # would explode an unclamped slope.
        pts = [(0.500, 0)] * 40 + [(0.502, 1)] * 40
        await _seed(db, pts)
        v = await p._calibrated_prob(0.53)
        assert 0.01 <= v <= 0.99
        # Cached: dropping the table does not change the mapping within TTL.
        await db.execute("DELETE FROM calibration")
        await db.commit()
        assert await p._calibrated_prob(0.53) == pytest.approx(v)
    finally:
        await db.close()
