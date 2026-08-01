"""Unified realized-P&L scorecard from the pnl_ledger table.

The ledger is the single source of truth for realized money (see
auramaur/broker/ledger.py). This report answers "where do we actually make
and lose money" by venue, strategy, category and month — plus a
reconciliation panel against the legacy sources (resolution_pnl +
cost_basis.realized_pnl) so drift between the accountings is visible
instead of silently trusted.
"""

from __future__ import annotations

from datetime import date

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def risk_free_benchmark(*, net_pnl: float, first_at, last_at,
                        settings=None) -> dict:
    """What the same capital would have earned doing nothing.

    Every graduation and profitability judgement in this system was scored
    against ZERO until 2026-08-01, which flatters any strategy that merely
    avoids losing money. Idle capital in the same account earns the risk-free
    rate for no work and no drawdown, so that is the honest hurdle.

    Returns a dict with `available=False` when the span or the book size is
    unknown. `book_capital_usd` defaults to 0 precisely so this stays silent
    rather than dividing by a guessed denominator — a wrong book size would
    produce a confidently wrong verdict, which is worse than no verdict.
    """
    out = {"available": False, "net_pnl": net_pnl}
    if settings is None:
        try:
            from config.settings import Settings
            settings = Settings()
        except Exception:
            return out
    cfg = getattr(settings, "benchmark", None)
    if cfg is None:
        return out

    def _day(v):
        try:
            return date.fromisoformat(str(v)[:10])
        except Exception:
            return None

    d0, d1 = _day(first_at), _day(last_at)
    if d0 is None or d1 is None:
        return out
    days = (d1 - d0).days
    if days < 1:
        # Under a day of record, annualising multiplies noise by ~365.
        return out

    annualised = net_pnl * 365.0 / days
    out.update({"days": days, "first_at": str(first_at)[:10],
                "last_at": str(last_at)[:10], "annualised": annualised,
                "rate": float(cfg.risk_free_annual_rate)})
    capital = float(cfg.book_capital_usd or 0.0)
    if capital <= 0:
        # Annualised dollars are still meaningful; the percentage is not.
        out["available"] = True
        return out
    rf_dollars = capital * float(cfg.risk_free_annual_rate)
    out.update({
        "available": True, "capital": capital,
        "return_pct": 100.0 * annualised / capital,
        "rf_dollars": rf_dollars,
        "excess_dollars": annualised - rf_dollars,
        "excess_pct": 100.0 * annualised / capital
        - 100.0 * float(cfg.risk_free_annual_rate),
    })
    return out


async def gather_ledger_report(db, *, is_paper: bool, settings=None) -> dict:
    flag = 1 if is_paper else 0

    async def _group(dim: str) -> list[dict]:
        rows = await db.fetchall(
            f"""SELECT COALESCE(NULLIF({dim}, ''), '(none)') AS k,
                       COUNT(*) AS n,
                       SUM(pnl) AS pnl,
                       SUM(fees) AS fees,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins
                FROM pnl_ledger WHERE is_paper = ?
                GROUP BY 1 ORDER BY pnl DESC""",
            (flag,),
        )
        return [dict(r) for r in (rows or [])]

    months = await db.fetchall(
        """SELECT substr(realized_at, 1, 7) AS k,
                  COUNT(*) AS n, SUM(pnl) AS pnl, SUM(fees) AS fees,
                  SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins
           FROM pnl_ledger WHERE is_paper = ?
           GROUP BY 1 ORDER BY 1""",
        (flag,),
    )

    total = await db.fetchone(
        """SELECT COUNT(*) AS n, COALESCE(SUM(pnl), 0) AS pnl,
                  COALESCE(SUM(fees), 0) AS fees,
                  SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins
           FROM pnl_ledger WHERE is_paper = ?""",
        (flag,),
    )

    # Span of the record, for annualising against the risk-free hurdle.
    span = await db.fetchone(
        """SELECT MIN(realized_at) AS first_at, MAX(realized_at) AS last_at
           FROM pnl_ledger WHERE is_paper = ?""",
        (flag,),
    )

    # Reconciliation vs the legacy accountings (live-scoped like they are):
    # cost_basis.realized_pnl books sells + settlements; the ledger should
    # match its mode-scoped total. resolution_pnl is whole-market (its
    # formula re-includes sell proceeds), so it is shown but expected to
    # differ wherever a market was partially sold before resolving.
    legacy_cb = await db.fetchone(
        "SELECT COALESCE(SUM(realized_pnl), 0) AS v FROM cost_basis WHERE is_paper = ?",
        (flag,),
    )
    legacy_res = await db.fetchone(
        "SELECT COALESCE(SUM(pnl), 0) AS v FROM resolution_pnl"
    )

    return {
        "is_paper": is_paper,
        "benchmark": risk_free_benchmark(
            net_pnl=float((total["pnl"] if total else 0) or 0)
            - float((total["fees"] if total else 0) or 0),
            first_at=(span["first_at"] if span else None),
            last_at=(span["last_at"] if span else None),
            settings=settings,
        ),
        "total": dict(total) if total else {"n": 0, "pnl": 0.0, "fees": 0.0, "wins": 0},
        "by_venue": await _group("venue"),
        "by_strategy": await _group("strategy_source"),
        "by_category": await _group("category"),
        "by_month": [dict(r) for r in (months or [])],
        "legacy_cost_basis": float(legacy_cb["v"]) if legacy_cb else 0.0,
        "legacy_resolution_pnl": float(legacy_res["v"]) if legacy_res else 0.0,
    }


def _pnl_text(v: float) -> Text:
    return Text(f"${v:+,.2f}", style="green" if v >= 0 else "red")


def _dim_table(title: str, rows: list[dict]) -> Table:
    t = Table(title=title, title_justify="left")
    t.add_column(title.lower())
    t.add_column("events", justify="right")
    t.add_column("win%", justify="right")
    t.add_column("fees", justify="right")
    t.add_column("realized", justify="right")
    for r in rows:
        n = int(r["n"] or 0)
        win = (int(r["wins"] or 0) / n * 100) if n else 0.0
        t.add_row(
            str(r["k"]),
            str(n),
            f"{win:.0f}%",
            f"${float(r['fees'] or 0):,.2f}",
            _pnl_text(float(r["pnl"] or 0)),
        )
    return t


def render_ledger_report(state: dict) -> Panel:
    mode = "paper" if state["is_paper"] else "LIVE"
    tot = state["total"]
    n = int(tot["n"] or 0)
    win = (int(tot["wins"] or 0) / n * 100) if n else 0.0

    head = Text()
    head.append(f"Realized P&L ledger — {mode}\n", style="bold")
    head.append(f"{n} realization events, {win:.0f}% wins, ")
    head.append(f"fees ${float(tot['fees'] or 0):,.2f}, net ")
    head.append(_pnl_text(float(tot["pnl"] or 0)))

    bench = state.get("benchmark") or {}
    if bench.get("available"):
        head.append("\n")
        # "net of fees" stated explicitly: the header's `net` above is
        # SUM(pnl) with fees broken out separately, while this annualises
        # SUM(pnl - fees) — the same figure graduation judges cells on. Same
        # word, two meanings, so name which one this is.
        head.append(
            f"\n{bench['days']}d of record ({bench['first_at']} → "
            f"{bench['last_at']}), net of fees ")
        head.append(_pnl_text(bench["net_pnl"]))
        head.append(" → annualised ")
        head.append(_pnl_text(bench["annualised"]))
        if "return_pct" in bench:
            rf = bench["rate"] * 100.0
            excess = bench["excess_pct"]
            head.append(f"  =  {bench['return_pct']:+.2f}%/yr on "
                        f"${bench['capital']:,.0f}")
            head.append("\nvs risk-free ")
            head.append(f"{rf:.2f}%/yr", style="bold")
            head.append(":  ")
            # The whole point of the line: below zero means idle cash in the
            # same account would have done better, for no work and no risk.
            head.append(
                f"{excess:+.2f}%/yr (${bench['excess_dollars']:+,.0f}/yr)",
                style="bold green" if excess >= 0 else "bold red")
            if excess < 0:
                head.append(
                    "  — behind cash", style="red")

    recon = Table(title="reconciliation", title_justify="left")
    recon.add_column("source")
    recon.add_column("total", justify="right")
    recon.add_column("note")
    recon.add_row("pnl_ledger", f"${float(tot['pnl'] or 0):+,.2f}", "events: sells + settlements")
    recon.add_row(
        "cost_basis.realized_pnl",
        f"${state['legacy_cost_basis']:+,.2f}",
        "tracks the ledger going forward; history under-counts (zeroed rows, "
        "settlements without a basis row no-op)",
    )
    recon.add_row(
        "resolution_pnl",
        f"${state['legacy_resolution_pnl']:+,.2f}",
        "whole-market formula, live-only — overlaps sells by design",
    )

    body = Group(
        head,
        Text(),
        _dim_table("venue", state["by_venue"]),
        _dim_table("strategy", state["by_strategy"]),
        _dim_table("category", state["by_category"]),
        _dim_table("month", state["by_month"]),
        recon,
    )
    return Panel(body, title="auramaur pnl", border_style="cyan")
