"""Intraday-drift measurement spike — does price drift toward the LLM estimate?

The intraday-convergence thesis: after a catalyst the market UNDER-reacts and
drifts toward the LLM's fresh estimate over hours (a speed/under-reaction edge),
rather than the LLM out-forecasting the resolution (which the bot's llm/news
paper record shows it does NOT). This measures the thesis with ZERO orders and
zero new LLM calls: it reuses the `signals` table (claude_prob = estimate,
market_prob = p0 at signal time) and just snapshots the market mid over a window,
recording how far price moved TOWARD the estimate vs fees.

Pure measurement. If post-signal drift-toward-estimate clears fees over 1-2
weeks, the intraday strategy is justified; if not, we built ~no code instead of
a trading loop + backtester.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger()


def drift_toward(estimate: float, p0: float, mid: float) -> float:
    """Signed price move TOWARD the estimate since the signal.

    Positive = the market moved toward the LLM estimate (under-reaction edge);
    negative = it moved away. Magnitude is in price units (dollars on 0-1).
    """
    direction = 1.0 if estimate >= p0 else -1.0
    return (mid - p0) * direction


def summarize(obs_rows: list[dict], horizons=(60, 120, 240), fee: float = 0.02) -> dict:
    """Aggregate drift-toward-estimate by horizon bucket.

    obs_rows: dicts with estimate, p0, minutes, mid (one per observation, many
    per (market_id, signal_at)). For each horizon H, take each signal's
    observation closest to H minutes and average drift_toward across signals;
    report the share that beats `fee` (the bar the strategy must clear).
    """
    by_signal: dict[tuple, list[dict]] = {}
    for r in obs_rows:
        if r["minutes"] <= 0:
            continue  # skip the t0 registration row
        by_signal.setdefault((r["market_id"], r["signal_at"]), []).append(r)

    out: dict = {"signals": len(by_signal), "horizons": {}}
    for h in horizons:
        drifts = []
        for obs in by_signal.values():
            closest = min(obs, key=lambda r: abs(r["minutes"] - h))
            if abs(closest["minutes"] - h) > h:   # no obs near this horizon
                continue
            drifts.append(drift_toward(closest["estimate"], closest["p0"], closest["mid"]))
        if not drifts:
            out["horizons"][h] = None
            continue
        mean = sum(drifts) / len(drifts)
        out["horizons"][h] = {
            "n": len(drifts),
            "mean_drift": round(mean, 4),
            "share_beating_fee": round(sum(1 for d in drifts if d > fee) / len(drifts), 3),
            "mean_beats_fee": mean > fee,
        }
    return out


class IntradayDriftTracker:
    def __init__(self, db, settings, discovery) -> None:
        self._db = db
        self._settings = settings
        self._discovery = discovery

    async def _ensure_table(self) -> None:
        await self._db.execute(
            """CREATE TABLE IF NOT EXISTS intraday_drift_obs (
                   id INTEGER PRIMARY KEY,
                   market_id TEXT NOT NULL, signal_at TEXT NOT NULL,
                   strategy TEXT, estimate REAL NOT NULL, p0 REAL NOT NULL,
                   obs_at TEXT NOT NULL, minutes REAL NOT NULL, mid REAL NOT NULL,
                   UNIQUE(market_id, signal_at, obs_at))""")
        await self._db.commit()

    async def run_once(self) -> int:
        cfg = self._settings.intraday_drift
        if not cfg.enabled:
            return 0
        await self._ensure_table()
        strat_filter = ",".join("'%s'" % s for s in cfg.strategies)

        # 1. Register fresh signals (t0 row, mid = p0 at signal time).
        new_rows = await self._db.fetchall(
            f"""SELECT market_id, timestamp, strategy_source, claude_prob, market_prob
                FROM signals
                WHERE strategy_source IN ({strat_filter})
                  AND timestamp >= datetime('now', '-{cfg.register_lookback_min} minutes')
                  AND NOT EXISTS (SELECT 1 FROM intraday_drift_obs o
                                  WHERE o.market_id = signals.market_id
                                    AND o.signal_at = signals.timestamp)
                LIMIT ?""", (cfg.max_tracks_per_cycle,))
        registered = 0
        for r in new_rows or []:
            await self._db.execute(
                """INSERT OR IGNORE INTO intraday_drift_obs
                   (market_id, signal_at, strategy, estimate, p0, obs_at, minutes, mid)
                   VALUES (?, ?, ?, ?, ?, ?, 0, ?)""",
                (r["market_id"], r["timestamp"], r["strategy_source"],
                 float(r["claude_prob"] or 0), float(r["market_prob"] or 0),
                 r["timestamp"], float(r["market_prob"] or 0)))
            registered += 1
        await self._db.commit()

        # 2. Observe still-active tracks (within the time-box) — snapshot mid.
        active = await self._db.fetchall(
            f"""SELECT DISTINCT market_id, signal_at, estimate, p0 FROM intraday_drift_obs
                WHERE (julianday('now') - julianday(signal_at)) * 24.0 < {cfg.time_box_hours}
                LIMIT ?""", (cfg.max_tracks_per_cycle,))
        observed = 0
        for t in active or []:
            try:
                m = await self._discovery.get_market(t["market_id"])
            except Exception:
                m = None
            if m is None or not (0.0 < m.outcome_yes_price < 1.0):
                continue
            minutes = await self._db.fetchone(
                "SELECT (julianday('now') - julianday(?)) * 1440.0 AS m", (t["signal_at"],))
            await self._db.execute(
                """INSERT OR IGNORE INTO intraday_drift_obs
                   (market_id, signal_at, strategy, estimate, p0, obs_at, minutes, mid)
                   VALUES (?, ?, '', ?, ?, datetime('now'), ?, ?)""",
                (t["market_id"], t["signal_at"], float(t["estimate"]), float(t["p0"]),
                 float(minutes["m"]), m.outcome_yes_price))
            observed += 1
        await self._db.commit()

        if registered or observed:
            log.info("intraday_drift.cycle", registered=registered, observed=observed)
        # Periodic readout so the edge (or its absence) is visible in the log.
        rows = await self._db.fetchall(
            "SELECT market_id, signal_at, estimate, p0, minutes, mid FROM intraday_drift_obs")
        rep = summarize([dict(r) for r in rows or []], fee=cfg.fee_threshold)
        if rep["signals"] >= cfg.report_min_signals:
            log.info("intraday_drift.report", **{"signals": rep["signals"],
                     **{f"h{h}": v for h, v in rep["horizons"].items()}})
        return registered + observed
