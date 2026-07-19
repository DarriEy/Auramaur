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
    await db.connect()
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
def propose(venue: str, market_id: str, side: str, fair_prob: float,
            stake: float, thesis: str) -> None:
    """Queue a proposal. The pillar executes it only after the charter rules,
    the full risk gateway, and the graduation ladder (paper-forced until the
    manager earns live)."""
    if len(thesis.strip()) < 40:
        console.print("[red]Thesis too short — name the mispricing mechanism "
                      "(charter rule 1; >= 40 chars).[/]")
        raise SystemExit(1)

    async def run() -> None:
        db = await _db()
        try:
            await db.execute(
                """INSERT INTO manager_proposals
                   (venue, market_id, side, fair_prob, stake_usd, thesis)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (venue, market_id, side.upper(), fair_prob, stake, thesis.strip()))
            await db.commit()
            row = await db.fetchone("SELECT last_insert_rowid() AS id")
            console.print(f"[green]Proposal {row['id']} queued[/] — "
                          f"{side.upper()} {market_id} on {venue} "
                          f"(fair {fair_prob:.2f}, ${stake:.2f}).")
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
            for col in ("id", "venue", "market", "side", "fair", "stake",
                        "status", "reason", "created"):
                table.add_column(col)
            for r in rows or []:
                table.add_row(str(r["id"]), r["venue"], r["market_id"], r["side"],
                              f"{r['fair_prob']:.2f}", f"{r['stake_usd']:.2f}",
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
