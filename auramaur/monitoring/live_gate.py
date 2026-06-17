"""Operational live-readiness preflight — gate ARMING live trading.

Distinct from monitoring/readiness.py, which measures whether a strategy has
*earned* live capital on edge. This answers a different question: is it
operationally SAFE to place live orders right now?

A BLOCK-severity result means new live ENTRIES must be suppressed (the bot
downgrades them to paper); EXITS always stay live so held positions can get out.
WARN degrades visibility but does not block. The whole point is a system that
refuses to fool itself: no live capital when the cost model, marks, or risk
state are untrustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from auramaur.db.database import Database

Severity = Literal["OK", "WARN", "BLOCK"]


@dataclass
class GateResult:
    name: str
    severity: Severity
    detail: str


@dataclass
class PreflightReport:
    results: list[GateResult] = field(default_factory=list)

    @property
    def live_allowed(self) -> bool:
        return not any(r.severity == "BLOCK" for r in self.results)

    @property
    def blocks(self) -> list[GateResult]:
        return [r for r in self.results if r.severity == "BLOCK"]

    @property
    def warnings(self) -> list[GateResult]:
        return [r for r in self.results if r.severity == "WARN"]


async def preflight(settings, db: Database) -> PreflightReport:
    """Run the operational live-readiness checks. Never raises — a check that
    can't evaluate degrades to WARN rather than crashing the gate."""
    report = PreflightReport()

    def add(name: str, severity: Severity, detail: str) -> None:
        report.results.append(GateResult(name, severity, detail))

    # 1. Kill switch — halts ALL trading by design.
    if Path("KILL_SWITCH").exists():
        add("kill_switch", "BLOCK", "KILL_SWITCH file present")
    else:
        add("kill_switch", "OK", "absent")

    # 2. Database reachable — the ledger and risk state live here.
    try:
        await db.fetchone("SELECT 1")
        add("database", "OK", "reachable")
    except Exception as e:
        add("database", "BLOCK", f"query failed: {str(e)[:80]}")

    # 3. Fee model sane — the cost model must be real, or every crossing edge is
    #    overstated. Block if fees look disabled.
    try:
        from auramaur.strategy.signals import POLYMARKET_TAKER_FEES
        fees = settings.arbitrage.exchange_fees
        kalshi_fee = float(fees.get("kalshi", 0.0) or 0.0)
        poly_ok = any(v > 0 for v in POLYMARKET_TAKER_FEES.values())
        if kalshi_fee <= 0 or not poly_ok:
            add("fee_model", "BLOCK",
                f"fee model looks disabled (kalshi={kalshi_fee}, poly_taker_nonzero={poly_ok})")
        else:
            add("fee_model", "OK", f"kalshi={kalshi_fee}, polymarket per-category taker")
    except Exception as e:
        add("fee_model", "BLOCK", f"unavailable: {str(e)[:80]}")

    # 4. Drawdown — never arm fresh entries through a breached drawdown limit.
    try:
        from auramaur.risk.portfolio import PortfolioTracker
        dd = await PortfolioTracker(db).get_drawdown()
        maxdd = float(settings.risk.max_drawdown_pct)
        if dd >= maxdd:
            add("drawdown", "BLOCK", f"{dd:.1f}% >= limit {maxdd:.1f}%")
        elif dd >= maxdd * 0.8:
            add("drawdown", "WARN", f"{dd:.1f}% nearing limit {maxdd:.1f}%")
        else:
            add("drawdown", "OK", f"{dd:.1f}% (limit {maxdd:.1f}%)")
    except Exception as e:
        add("drawdown", "WARN", f"could not compute: {str(e)[:80]}")

    # 5. Mark integrity — live positions priced <=0 mean the risk gates
    #    (drawdown/exposure/Kelly) are acting on fiction (the zero-mark bug).
    try:
        row = await db.fetchone(
            "SELECT COUNT(*) AS n FROM portfolio "
            "WHERE is_paper = 0 AND size > 0 "
            "AND (current_price IS NULL OR current_price <= 0)"
        )
        zero = int(row["n"]) if row else 0
        if zero > 5:
            add("position_marks", "BLOCK", f"{zero} live positions marked <=0")
        elif zero > 0:
            add("position_marks", "WARN", f"{zero} live positions marked <=0")
        else:
            add("position_marks", "OK", "no zero-marked live positions")
    except Exception as e:
        add("position_marks", "WARN", f"could not check: {str(e)[:80]}")

    # 6. LLM budget — WARN only: the live surface (arb/MM) needs no LLM; only
    #    directional (paper-forced) does.
    try:
        from auramaur.nlp import call_budget
        budget = int(settings.nlp.daily_claude_call_budget)
        used = call_budget.calls_today()
        if budget > 0 and used >= budget:
            add("llm_budget", "WARN",
                f"daily LLM budget spent ({used}/{budget}) — directional degraded, arb/MM unaffected")
        else:
            add("llm_budget", "OK", f"{used}/{budget}")
    except Exception as e:
        add("llm_budget", "WARN", f"could not check: {str(e)[:80]}")

    # 7. Three-gate state (informational): the standing live arming gates.
    add(
        "live_gates",
        "OK" if settings.is_live else "WARN",
        f"AURAMAUR_LIVE={settings.auramaur_live} execution.live={settings.execution.live} "
        f"=> is_live={settings.is_live}",
    )

    return report
