"""Auramaur CLI — interim-manager proposal queue (docs/INTERIM_MANAGER.md)."""

from __future__ import annotations

import asyncio

import click
from rich.table import Table

from auramaur.cli._base import console, main
from auramaur.db.database import Database
from auramaur.runtime import db_path


@main.group()
def manager():
    """Interim-manager proposals: manual entries, ladder-evaluated."""


async def _db() -> Database:
    db = Database(str(db_path()))
    # CLI fast path (CLAUDE.md policy): no DDL/write locks against the live
    # bot's database when the schema is already current.
    await db.connect(ensure_schema=False)
    return db


@manager.command()
@click.option("--venue", type=click.Choice(["polymarket", "kalshi"]), required=True)
@click.option("--market", "market_id", required=True, help="Market id / ticker.")
@click.option("--side", type=click.Choice(["buy", "sell"]), required=True)
@click.option("--fair-prob", type=click.FloatRange(0.01, 0.99), required=True,
              help="Your calibrated fair probability for YES.")
@click.option("--stake", type=click.FloatRange(1.0, 25.0), default=10.0,
              show_default=True, help="Stake in USD (risk sizing may shrink it).")
@click.option("--thesis", required=True,
              help="The nameable mispricing mechanism (charter rule 1; >= 40 chars).")
@click.option("--thesis-class", "thesis_class",
              type=click.Choice(["forecast_divergence", "cross_venue",
                                 "conditional_inconsistency", "timeline",
                                 "base_rate", "resolution_language",
                                 "information_lag", "exclusive_basket",
                                 "unclassified"]),
              default="unclassified", show_default=True,
              help="Which anchor class the thesis belongs to (evaluated per class).")
@click.option("--ci", default=None,
              help="Fair-probability confidence interval, e.g. 0.33,0.47 "
                   "(half-width becomes the uncertainty haircut).")
@click.option("--max-entry", "max_entry", type=click.FloatRange(0.01, 0.99),
              default=None, help="Max price PAID for the side taken; protects the edge.")
@click.option("--catalyst", default="", help="What resolves or closes the gap.")
@click.option("--invalidation", default="",
              help="Observable condition that kills the thesis.")
@click.option("--sunset", default=None,
              help="ISO time after which the thesis is stale, e.g. 2026-07-20T12:00Z.")
def propose(venue: str, market_id: str, side: str, fair_prob: float,
            stake: float, thesis: str, thesis_class: str, ci: str | None,
            max_entry: float | None, catalyst: str, invalidation: str,
            sunset: str | None) -> None:
    """Queue a proposal. The pillar executes it only after the charter rules,
    the full risk gateway, and the graduation ladder (paper-forced until the
    manager earns live)."""
    if len(thesis.strip()) < 40:
        console.print("[red]Thesis too short — name the mispricing mechanism "
                      "(charter rule 1; >= 40 chars).[/]")
        raise SystemExit(1)
    lo = hi = None
    if ci:
        try:
            lo, hi = (float(x) for x in ci.split(","))
        except ValueError:
            console.print("[red]--ci must be 'lo,hi', e.g. 0.33,0.47[/]")
            raise SystemExit(1) from None
        if not (0.0 < lo <= fair_prob <= hi < 1.0):
            console.print("[red]--ci must bracket --fair-prob inside (0, 1).[/]")
            raise SystemExit(1)
    if sunset:
        from datetime import datetime
        try:
            datetime.fromisoformat(sunset.replace("Z", "+00:00"))
        except ValueError:
            console.print("[red]--sunset must be ISO 8601, e.g. 2026-07-20T12:00Z[/]")
            raise SystemExit(1) from None

    async def run() -> None:
        db = await _db()
        try:
            await db.execute(
                """INSERT INTO manager_proposals
                   (venue, market_id, side, fair_prob, stake_usd, thesis,
                    thesis_class, confidence_lo, confidence_hi, max_entry_price,
                    catalyst, invalidation, sunset_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (venue, market_id, side.upper(), fair_prob, stake, thesis.strip(),
                 thesis_class, lo, hi, max_entry, catalyst.strip(),
                 invalidation.strip(),
                 sunset.replace("Z", "+00:00") if sunset else None))
            await db.commit()
            row = await db.fetchone("SELECT last_insert_rowid() AS id")
            console.print(f"[green]Proposal {row['id']} queued[/] — "
                          f"{side.upper()} {market_id} on {venue} "
                          f"(fair {fair_prob:.2f}, class {thesis_class}, "
                          f"${stake:.2f}"
                          + (f", limit {max_entry:.2f}" if max_entry else "")
                          + ").")
        finally:
            await db.close()

    asyncio.run(run())


@manager.command("list")
@click.option("--all", "show_all", is_flag=True, help="Include decided proposals.")
def list_proposals(show_all: bool) -> None:
    """Show the proposal queue and its decisions (the audit log)."""

    async def run() -> None:
        db = await _db()
        try:
            where = "" if show_all else "WHERE status = 'pending'"
            rows = await db.fetchall(
                f"SELECT * FROM manager_proposals {where} "
                "ORDER BY created_at DESC LIMIT 50")
            table = Table(title="interim-manager proposals")
            for col in ("id", "by", "venue", "market", "side", "fair", "stake",
                        "status", "reason", "created"):
                table.add_column(col)
            for r in rows or []:
                keys = r.keys()
                by = r["proposer"] if "proposer" in keys else "operator"
                table.add_row(str(r["id"]), by, r["venue"], r["market_id"],
                              r["side"], f"{r['fair_prob']:.2f}",
                              f"{r['stake_usd']:.2f}",
                              r["status"], (r["reason"] or "")[:50],
                              str(r["created_at"])[:16])
            console.print(table)
        finally:
            await db.close()

    asyncio.run(run())


@manager.command()
@click.argument("proposal_id", type=int)
def cancel(proposal_id: int) -> None:
    """Cancel a pending proposal."""

    async def run() -> None:
        db = await _db()
        try:
            cur = await db.execute(
                """UPDATE manager_proposals SET status = 'cancelled',
                   reason = 'operator cancel', decided_at = datetime('now')
                   WHERE id = ? AND status = 'pending'""", (proposal_id,))
            await db.commit()
            if cur.rowcount:
                console.print(f"[yellow]Proposal {proposal_id} cancelled.[/]")
            else:
                console.print(f"[red]No pending proposal {proposal_id}.[/]")
        finally:
            await db.close()

    asyncio.run(run())


@manager.command()
def report() -> None:
    """Per-thesis-class scorecard: was the reasoning good, not just the P&L?"""

    async def run() -> None:
        db = await _db()
        try:
            rows = await db.fetchall(
                """SELECT mp.proposer || ':' || mp.thesis_class AS cls,
                          COUNT(*) AS proposals,
                          SUM(mp.status = 'executed') AS executed,
                          SUM(mp.status = 'skipped') AS skipped,
                          AVG(mp.robust_edge) AS avg_edge,
                          COALESCE(SUM(led.pnl), 0) AS realized
                     FROM manager_proposals mp
                     LEFT JOIN pnl_ledger led
                       ON led.market_id = mp.market_id
                      AND led.strategy_source = 'interim_manager'
                    GROUP BY mp.thesis_class ORDER BY proposals DESC""")
            table = Table(title="interim-manager scorecard (by proposer:class)")
            for col in ("class", "proposals", "executed", "skipped",
                        "avg robust edge", "realized P&L"):
                table.add_column(col)
            for r in rows or []:
                table.add_row(r["cls"], str(r["proposals"]), str(r["executed"] or 0),
                              str(r["skipped"] or 0),
                              f"{(r['avg_edge'] or 0):+.3f}",
                              f"${(r['realized'] or 0):+.2f}")
            console.print(table)
            console.print(
                "[dim]Full audit per proposal: manager list --all. Calibration "
                "of fair-prob vs outcomes: the standard calibration report "
                "(predictions are recorded per executed proposal).[/]")
        finally:
            await db.close()

    asyncio.run(run())
