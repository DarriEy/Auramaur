"""Auramaur CLI — market-intelligence commands (read-only)."""

from __future__ import annotations

import asyncio

import click
from rich.table import Table

from auramaur.cli._base import console, main


@main.command()
@click.option("--period", default="MONTH", type=click.Choice(["DAY", "WEEK", "MONTH", "ALL"]),
              help="Leaderboard window (default: MONTH = current winners).")
@click.option("--limit", default=25, help="How many top-PnL wallets to scan.")
@click.option("--activity/--no-activity", default=False,
              help="Also show each directional winner's most recent trade.")
def whales(period, limit, activity):
    """Whale watch — top-PnL Polymarket wallets, classified by archetype.

    INTELLIGENCE ONLY (read-only public data) — not a trading signal. The
    reverse-engineering showed winners split into DIRECTIONAL conviction (the
    replicable archetype) vs MM/ARB (turnover machine). This surfaces the
    directional winners to watch — copy-trading is a documented money-loser, so
    this is for situational awareness, not mirroring.
    """
    from auramaur.data_sources.polymarket_leaderboard import PolymarketLeaderboard

    async def _run():
        lb = PolymarketLeaderboard()
        leaders = await lb.top(period=period, limit=limit)
        if not leaders:
            console.print("[yellow]no leaderboard data (API unreachable?)[/]")
            return

        counts = {"directional": 0, "mm_arb": 0, "mixed": 0, "unknown": 0}
        for ld in leaders:
            counts[ld.archetype] = counts.get(ld.archetype, 0) + 1

        t = Table(title=f"Polymarket top-PnL wallets — {period} (intelligence, read-only)")
        for c in ("rank", "name", "PnL $M", "Vol $M", "PnL/Vol", "archetype"):
            t.add_column(c, justify="right" if "$" in c or c in ("rank", "PnL/Vol") else "left")
        for ld in leaders:
            style = "green" if ld.archetype == "directional" else (
                "dim" if ld.archetype == "mm_arb" else "")
            t.add_row(str(ld.rank), (ld.name or ld.wallet[:10])[:20],
                      f"{ld.pnl/1e6:.1f}", f"{ld.vol/1e6:.0f}",
                      f"{ld.pnl_vol_ratio:.0%}", f"[{style}]{ld.archetype}[/]" if style else ld.archetype)
        console.print(t)
        console.print(
            f"[bold]{counts['directional']}[/] directional · "
            f"{counts['mm_arb']} mm/arb · {counts['mixed']} mixed  "
            f"[dim](directional = the conviction archetype worth watching)[/]")

        if activity:
            direc = [ld for ld in leaders if ld.archetype == "directional"][:8]
            console.print("\n[bold]Recent accumulations by directional winners:[/]")
            for ld in direc:
                acts = await lb.recent_activity(ld.wallet, limit=3)
                if not acts:
                    continue
                a = acts[0]
                console.print(
                    f"  [green]{(ld.name or ld.wallet[:8])[:16]}[/]: "
                    f"{a.get('side','?')} {a.get('outcome','?')} @ "
                    f"{float(a.get('price',0) or 0):.2f} — {str(a.get('title',''))[:50]}")

    asyncio.run(_run())
