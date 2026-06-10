"""Unified realized-P&L scorecard from the pnl_ledger table.

The ledger is the single source of truth for realized money (see
auramaur/broker/ledger.py). This report answers "where do we actually make
and lose money" by venue, strategy, category and month — plus a
reconciliation panel against the legacy sources (resolution_pnl +
cost_basis.realized_pnl) so drift between the accountings is visible
instead of silently trusted.
"""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


async def gather_ledger_report(db, *, is_paper: bool) -> dict:
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
