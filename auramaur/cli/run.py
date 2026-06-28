"""Auramaur CLI — run commands."""

from __future__ import annotations

import asyncio

import click
from rich.live import Live

from auramaur.db.database import Database
from config.settings import Settings

from auramaur.cli._base import console, main

@main.command()
@click.option("--agent", is_flag=True, default=False, help="Use agentic analyzer (relational reasoning + web search)")
@click.option("--hybrid", is_flag=True, default=False, help="Multi-strategy: arb + news speed + domain LLM + market making")
@click.option("--exchange", default=None, type=click.Choice(["polymarket", "kalshi", "ibkr"]), help="Run only a specific exchange (isolated instance)")
def run(agent: bool, hybrid: bool, exchange: str | None):
    """Start the bot."""
    # Hang diagnostic: `kill -USR1 <pid>` dumps every thread's Python stack
    # (including the asyncio loop thread) to stderr — even when the loop is
    # blocked in a synchronous C call. Needs no root/py-spy. Fires only on the
    # explicit signal, so it has no effect on normal operation.
    import faulthandler
    import signal as _signal

    if hasattr(_signal, "SIGUSR1"):
        faulthandler.register(_signal.SIGUSR1, all_threads=True)

    settings = Settings()

    if agent:
        settings.analysis.mode = "agent"
        if settings.is_live:
            console.print("[bold red]AGENT MODE[/] — [bold]LIVE TRADING[/]")
        else:
            console.print("[bold yellow]AGENT MODE[/] — paper trading")
    if hybrid:
        # NOTE: --hybrid used to force market_maker.enabled = True here
        # (hybrid.market_maker_auto_enable), silently reverting an explicit
        # config disable on every restart — it cost three blind restarts
        # during the 2026-06-12 incident before anyone found it. Book
        # enablement now lives in exactly one place: market_maker.enabled.
        mode = "[bold red]LIVE TRADING[/]" if settings.is_live else "paper trading"
        console.print(f"[bold cyan]HYBRID MODE[/] — {mode}")
        # The full per-book picture (modes, gates, graduation, ledger) is the
        # "strategy books" panel printed at bot startup.
        console.print(
            "[dim]  Pillars: arb + news speed + LLM + market making"
            " + bias_harvest + entailment_arb + resolution_lens[/]"
        )
    if exchange:
        console.print(f"[bold blue]Starting Auramaur bot (exchange: {exchange})...[/]")
    else:
        console.print("[bold blue]Starting Auramaur bot...[/]")
    from auramaur.cli import AuramaurBot  # call-time lookup keeps test patch working
    bot = AuramaurBot(settings=settings, exchange_filter=exchange, hybrid=hybrid)
    asyncio.run(bot.run())

@main.command()
@click.pass_context
def dashboard(ctx):
    """Deprecated alias for `cockpit` (the unified live view)."""
    console.print("[yellow]`dashboard` is deprecated — running `cockpit` "
                  "(the unified view). Use `auramaur cockpit` going forward.[/]")
    ctx.invoke(cockpit)

@main.command()
def cockpit():
    """Live cockpit — read-only fused view (DB + venue balances + pillar liveness)."""
    from auramaur.monitoring import cockpit as ck

    async def _run():
        settings = Settings()
        db = Database()
        await db.connect()
        cache: dict = {}
        with Live(console=console, refresh_per_second=2, screen=True) as live:
            while True:
                try:
                    state = await ck.gather_state(db, settings, cache)
                    live.update(ck.make_layout(state))
                except Exception as e:
                    console.print(f"[red]cockpit error: {e}[/]")
                await asyncio.sleep(2)

    asyncio.run(_run())

@main.command()
def status():
    """One-shot snapshot — same view as `cockpit`, rendered once."""
    from auramaur.monitoring import cockpit as ck

    async def _status():
        settings = Settings()
        db = Database()
        await db.connect()
        try:
            # cache=None → fresh venue balances, no Live loop.
            state = await ck.gather_state(db, settings, cache=None)
            console.print(ck.make_layout(state, compact=True))

            # Extra, live-only: confirm the CLOB connection + open-order count.
            if settings.is_live and settings.polygon_private_key:
                try:
                    from py_clob_client_v2 import ClobClient, ApiCreds
                    client = ClobClient(
                        "https://clob.polymarket.com",
                        chain_id=137,
                        key=settings.polygon_private_key,
                        creds=ApiCreds(
                            api_key=settings.polymarket_api_key,
                            api_secret=settings.polymarket_api_secret,
                            api_passphrase=settings.polymarket_passphrase,
                        ),
                    )
                    orders = client.get_open_orders()
                    n = len(orders) if isinstance(orders, list) else 0
                    console.print(f"Polymarket: [green]connected[/] — {n} open orders")
                except Exception as e:
                    console.print(f"Polymarket: [red]error[/] — {str(e)[:60]}")
        finally:
            await db.close()

    asyncio.run(_status())
