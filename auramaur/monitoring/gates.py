"""Gate dashboard — for each gated/experimental feature, show its flag state and
whether the data has earned flipping it on (GO / WAIT / NO-GO).

Everything risky in Auramaur is measure-gated. This is the one place to see, at a
glance, which switches are ready and which are still waiting on evidence.

    auramaur gates
"""

from __future__ import annotations

import sqlite3

import structlog

from rich.table import Table
from rich.text import Text

from auramaur.runtime import db_path as runtime_db_path

log = structlog.get_logger()


def _resolved_dollar_markets(db_path: str) -> int:
    """Count live markets with fills AND a resolved outcome (the $-edge sample)."""
    # busy_timeout: without it a checkpoint in flight raises SQLITE_BUSY, which
    # the except below turns into 0 — indistinguishable from "no data".
    try:
        c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        c.execute("PRAGMA busy_timeout=5000")
        n = c.execute(
            "SELECT COUNT(DISTINCT f.market_id) FROM fills f "
            "JOIN calibration cal ON cal.market_id=f.market_id "
            "WHERE f.is_paper=0 AND cal.actual_outcome IS NOT NULL").fetchone()[0]
        c.close()
        return n
    except Exception as exc:
        # Returning 0 makes an unreadable database look like an empty one, and
        # the gate dashboard renders that as "WAIT (need data)" — so an
        # operator never flips a filter the data has already earned. Surface
        # the failure instead of laundering it into a plausible number.
        log.warning("gates.resolved_dollar_markets_unreadable",
                    db_path=db_path, error=str(exc)[:160])
        return -1


def gather(settings, db_path: str | None = None) -> list[dict]:
    from auramaur.risk.tolerance import current_tolerance
    # Resolve through runtime.db_path() so this follows AURAMAUR_DB_PATH /
    # AURAMAUR_STATE_DIR the way Database.__init__ does. The old default was
    # the bare literal "auramaur.db", which only resolves under a CWD that
    # happens to be the repo root — under compose (/app/state/auramaur.db) or
    # from any other directory the connect raised, the except returned 0, and
    # the dashboard reported "0/100 resolved-$ markets -> WAIT (need data)"
    # while hundreds existed.
    n_resolved = _resolved_dollar_markets(db_path or str(runtime_db_path()))
    mc = settings.momentum_coupling
    tol = current_tolerance(settings)
    tol_label = ("most conservative" if tol <= 15 else "conservative" if tol < 45
                 else "neutral" if tol <= 55 else "aggressive" if tol < 85 else "YOLO")
    rows = [
        {
            "feature": "Risk-tolerance lever",
            "flag": f"risk_tolerance={tol:.0f}/100",
            "criterion": "0=conservative · 50=neutral · 100=YOLO (scales prob/stat/risk)",
            "status": tol_label,
            "verdict": "operational",
        },
        {
            "feature": "Divergence filter (slow loop)",
            "flag": f"divergence_filter_enabled={settings.risk.divergence_filter_enabled}",
            "criterion": "≥100 resolved-$ markets confirm mid-divergence adverse selection",
            # -1 means the database could not be read. That is a DIFFERENT
            # state from "no data yet", and rendering it as 0/100 told the
            # operator to keep waiting for a sample that already exists.
            "status": ("database unreadable" if n_resolved < 0
                       else f"{n_resolved}/100 resolved-$ markets"),
            "verdict": ("UNKNOWN (cannot read db)" if n_resolved < 0
                        else "GO" if n_resolved >= 100 else "WAIT (need data)"),
        },
        {
            "feature": "Momentum coupling — detect (fast path)",
            "flag": f"momentum_coupling.enabled={mc.enabled}",
            "criterion": "always safe (detection-only, places nothing)",
            "status": "ready",
            "verdict": "GO (safe to enable for detection)",
        },
        {
            "feature": "Momentum coupling — EXECUTE",
            "flag": f"momentum_coupling.execute={mc.execute}",
            "criterion": "coupling_tradeability.py net > 0 after cost",
            "status": "last measured: -0.13 / 9 trades (cost-dominated)",
            "verdict": "NO-GO (not tradeable yet)",
        },
        {
            "feature": "Kraken directional speculation",
            "flag": f"kraken.directional_enabled={settings.kraken.directional_enabled}",
            "criterion": "validated momentum edge vs buy-and-hold",
            "status": "measured: -5.1% vs -1.3% B&H (value-destroying)",
            "verdict": "NO-GO (kept ON by choice)",
        },
        {
            "feature": "IBKR equity directional",
            "flag": "REMOVED 2026-06-09",
            "criterion": "validated edge + funded account + market data",
            "status": "pre-failed (0W/20L on Kraken, backtests negative) — deleted",
            "verdict": "REMOVED",
        },
        {
            "feature": "Cross-venue transfers (armed)",
            "flag": f"transfers_armed={settings.transfers_armed}",
            "criterion": "operational gate (not edge): env + config + whitelist + approval",
            "status": "armed" if settings.transfers_armed else "disarmed",
            "verdict": "operational" if settings.transfers_armed else "disarmed",
        },
    ]
    return rows


def render(rows: list[dict]) -> Table:
    t = Table(title="Auramaur — feature gates", expand=True)
    t.add_column("feature", style="cyan", no_wrap=True)
    t.add_column("flag")
    t.add_column("gate criterion", style="dim")
    t.add_column("current status")
    t.add_column("verdict")
    for r in rows:
        v = r["verdict"]
        color = ("green" if v.startswith("GO") or v == "operational"
                 else "yellow" if v.startswith("WAIT") or v == "disarmed" else "red")
        t.add_row(r["feature"], r["flag"], r["criterion"], r["status"],
                  Text(v, style=color))
    return t
