"""Intraday-drift spike: drift math + register/observe (no trading)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from auramaur.db.database import Database
from auramaur.monitoring.intraday_drift import (
    IntradayDriftTracker,
    drift_toward,
    summarize,
)
from config.settings import Settings


def test_drift_toward_direction():
    # estimate above p0: price rising toward it = positive drift
    assert abs(drift_toward(0.70, 0.50, 0.60) - 0.10) < 1e-9
    assert drift_toward(0.70, 0.50, 0.45) < 0          # moved away (down)
    # estimate below p0: price falling toward it = positive drift
    assert abs(drift_toward(0.30, 0.50, 0.40) - 0.10) < 1e-9
    assert drift_toward(0.30, 0.50, 0.55) < 0          # moved away (up)


def test_summarize_buckets_and_fee():
    rows = [
        # signal A: estimate 0.7, p0 0.5; at ~60min mid 0.62 -> drift +0.12 (>fee)
        {"market_id": "A", "signal_at": "t", "estimate": 0.7, "p0": 0.5, "minutes": 0, "mid": 0.5},
        {"market_id": "A", "signal_at": "t", "estimate": 0.7, "p0": 0.5, "minutes": 62, "mid": 0.62},
        # signal B: estimate 0.3, p0 0.5; at ~60min mid 0.55 -> drift -0.05 (away)
        {"market_id": "B", "signal_at": "t", "estimate": 0.3, "p0": 0.5, "minutes": 58, "mid": 0.55},
    ]
    rep = summarize(rows, horizons=(60,), fee=0.02)
    assert rep["signals"] == 2
    h = rep["horizons"][60]
    assert h["n"] == 2
    assert abs(h["mean_drift"] - ((0.12 + -0.05) / 2)) < 1e-9
    assert h["share_beating_fee"] == 0.5   # only A beats fee


def _settings():
    s = Settings()
    s.intraday_drift.enabled = True
    s.intraday_drift.report_min_signals = 1
    return s


def test_tracker_registers_and_observes():
    async def run():
        db = Database(":memory:"); await db.connect()
        try:
            # seed a signal ~5 min ago (within the 15-min register window, but a
            # distinct obs_at so the registration row and the live obs don't collide)
            await db.execute(
                """INSERT INTO signals (market_id, timestamp, claude_prob, claude_confidence,
                   market_prob, edge, evidence_summary, action, strategy_source)
                   VALUES ('m1', datetime('now','-5 minutes'), 0.70, 'MEDIUM', 0.50, 20, 'x', 'BUY', 'news_speed')""")
            await db.commit()
            disc = MagicMock()
            disc.get_market = AsyncMock(return_value=SimpleNamespace(outcome_yes_price=0.60))
            tracker = IntradayDriftTracker(db, _settings(), disc)
            await tracker.run_once()
            # one t0 registration (mid=p0) + one live observation (mid=0.60)
            rows = await db.fetchall("SELECT minutes, mid FROM intraday_drift_obs WHERE market_id='m1' ORDER BY minutes")
            assert len(rows) == 2
            assert rows[0]["minutes"] == 0 and abs(rows[0]["mid"] - 0.50) < 1e-9
            assert rows[1]["minutes"] > 4               # ~5 min later
            assert abs(rows[1]["mid"] - 0.60) < 1e-9      # drift toward 0.70 captured
        finally:
            await db.close()
    asyncio.run(run())
