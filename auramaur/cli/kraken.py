"""Auramaur CLI — kraken commands."""

from __future__ import annotations

import asyncio

import click
from rich.table import Table

from auramaur.db.database import Database
from config.settings import Settings

from auramaur.cli._base import console, main

@main.group()
def kraken():
    """Manual Kraken spot trading (gated + typed confirmation)."""

async def _kraken_spot(pair: str, side: str, usd: float, assume_yes: bool) -> None:
    from auramaur.exchange.kraken import KrakenSpotClient
    from auramaur.exchange.models import OrderSide
    s = Settings()
    k = KrakenSpotClient(s)
    try:
        price = await k.get_price(pair)
        if not price:
            console.print(f"[red]No price for {pair}[/]")
            return
        vol = round(usd / price, 8)
        mode = "[bold red]LIVE — REAL ORDER[/]" if s.is_live else "[green]paper (validate-only)[/]"
        console.print(f"\n{side.upper()} ~${usd:.2f} of [cyan]{pair}[/] "
                      f"({vol} @ ~{price})  {mode}")
        if not assume_yes and s.is_live:
            ans = click.prompt(f"Type the amount ({usd:.2f}) to confirm", default="",
                               show_default=False).strip()
            if ans not in (f"{usd:.2f}", str(usd)):
                console.print("[yellow]Not confirmed — aborted.[/]")
                return
        side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL

        # Auto-bridge: buying a non-USDC-quoted pair (e.g. gold PAXGUSD) needs the
        # quote currency. Convert USDC -> quote to cover the shortfall first.
        if side == "buy":
            quote = await k.get_pair_quote(pair)              # e.g. 'ZUSD'
            qsym = quote[1:] if quote and quote[0] in "XZ" else quote  # ZUSD -> USD
            if quote and qsym and qsym != "USDC":
                bal = await k.get_balance()
                have = bal.get(quote, 0.0)
                if have < usd:
                    short = round(usd - have + 0.5, 2)
                    bridge_pair = f"USDC{qsym}"               # USDCUSD / USDCUSDT
                    console.print(f"[dim]bridging ~${short:.2f} USDC -> {qsym} "
                                  f"via {bridge_pair}[/]")
                    bres = await k.place_spot_order(bridge_pair, OrderSide.SELL, short,
                                                    ordertype="market", purpose="manual",
                                                    max_usd=short * 1.1)
                    if bres.status not in ("pending", "paper"):
                        console.print(f"[red]bridge failed: {bres.error_message}[/] — aborting")
                        return

        res = await k.place_spot_order(pair, side_enum, vol, ordertype="market",
                                       purpose="manual", max_usd=usd * 1.1)
        color = "green" if res.status in ("pending", "paper") else "red"
        console.print(f"Result: [{color}]{res.status}[/] | id={res.order_id}"
                      + (f" | {res.error_message}" if res.error_message else ""))
    finally:
        await k.close()

@kraken.command("balance")
def kraken_balance_cmd():
    """Show Kraken balances."""
    async def _run():
        from auramaur.exchange.kraken import KrakenSpotClient
        k = KrakenSpotClient(Settings())
        bal = await k.get_balance()
        for a, v in sorted(bal.items()):
            if v > 0:
                console.print(f"  [magenta]{a:8}[/] {v}")
        await k.close()
    asyncio.run(_run())

@kraken.command("pnl")
def kraken_pnl():
    """Performance readout for the directional (spec) book — is it net positive?"""
    from auramaur.monitoring.spec_report import (
        gather_spec_performance, render_spec_performance,
    )

    async def _run():
        db = Database()
        await db.connect()
        try:
            state = await gather_spec_performance(db, Settings())
            console.print(render_spec_performance(state))
        finally:
            await db.close()

    asyncio.run(_run())

@kraken.command("signal")
def kraken_signal():
    """Latest LLM/news directional view per pair (P(up) + LONG/hold/exit).

    Reads the views the running bot records to calibration when
    kraken.directional_llm_enabled is on. Empty until the bot has run with it.
    """
    async def _run():
        settings = Settings()
        db = Database()
        await db.connect()
        try:
            rows = await db.fetchall(
                """SELECT market_id, predicted_prob, created_at FROM calibration c
                   WHERE market_id LIKE 'kraken-dir:%'
                     AND created_at = (SELECT MAX(created_at) FROM calibration c2
                                       WHERE c2.market_id = c.market_id)
                   ORDER BY predicted_prob DESC"""
            )
            if not rows:
                console.print(
                    "[dim]No LLM directional views recorded yet. Set "
                    "kraken.directional_llm_enabled=true and let the bot run.[/]"
                )
                return
            min_prob = settings.kraken.directional_llm_min_prob
            exit_prob = settings.kraken.directional_llm_exit_prob
            paper = settings.kraken.directional_llm_paper
            mode = "PAPER" if paper else "LIVE"
            table = Table(title=f"Kraken LLM directional views  ({mode})")
            table.add_column("pair", style="magenta")
            table.add_column("P(up)", justify="right")
            table.add_column("signal", justify="right")
            table.add_column("as of")
            for r in rows:
                pair = r["market_id"].split(":", 1)[1]
                p = r["predicted_prob"]
                if p >= min_prob:
                    sig = "[green]LONG[/]"
                elif p < exit_prob:
                    sig = "[red]exit[/]"
                else:
                    sig = "[dim]hold[/]"
                table.add_row(pair, f"{p:.0%}", sig, str(r["created_at"]))
            console.print(table)
            console.print(
                f"[dim]LONG >= {min_prob:.0%} · exit < {exit_prob:.0%} · "
                f"mode={mode} (directional_llm_paper={paper})[/]"
            )

            # Calibration scoreboard — the closed feedback loop. Each tracked bet
            # is resolved at horizon (spot vs the reference price) under the
            # isolated 'kraken_spot' category, so directional edge is measurable.
            resolved = await db.fetchall(
                "SELECT predicted_prob, actual_outcome FROM calibration "
                "WHERE category='kraken_spot' AND actual_outcome IS NOT NULL"
            )
            if resolved:
                n = len(resolved)
                hits = sum(1 for r in resolved
                           if (r["predicted_prob"] >= 0.5) == (r["actual_outcome"] == 1))
                brier = sum((r["predicted_prob"] - r["actual_outcome"]) ** 2 for r in resolved) / n
                color = "green" if hits / n > 0.5 else "red"
                console.print(
                    f"[bold]Directional scoreboard[/] (resolved): "
                    f"[{color}]{hits}/{n} correct ({hits / n:.0%})[/] · "
                    f"Brier {brier:.3f} [dim](lower=better; 0.25=coin-flip)[/]"
                )
            else:
                console.print(
                    "[dim]No resolved directional bets yet — edge unmeasurable "
                    "until bets reach their horizon.[/]"
                )

            outstanding = await db.fetchall(
                "SELECT pair, prob, ref_price, due_at FROM kraken_dir_signals ORDER BY due_at"
            )
            if outstanding:
                console.print(f"[dim]{len(outstanding)} bet(s) outstanding; "
                              f"next resolves {outstanding[0]['due_at']}[/]")

            conv = settings.kraken
            if getattr(conv, "directional_conviction_budget_enabled", False):
                console.print(
                    f"[dim]Conviction budget ON (min_mult="
                    f"{getattr(conv, 'directional_conviction_min_mult', 0.34)}); "
                    f"live multiplier is logged as conv_mult on budget_full.[/]"
                )
            else:
                console.print("[dim]Conviction budget OFF (static ceiling).[/]")
        finally:
            await db.close()

    asyncio.run(_run())

@kraken.command("buy")
@click.option("--pair", required=True, help="e.g. XBTUSDC")
@click.option("--usd", required=True, type=float, help="USD/USDC amount")
@click.option("--yes", "assume_yes", is_flag=True, help="skip typed confirmation")
def kraken_buy(pair, usd, assume_yes):
    """Buy ~--usd of --pair (market order)."""
    asyncio.run(_kraken_spot(pair, "buy", usd, assume_yes))

@kraken.command("sell")
@click.option("--pair", required=True, help="e.g. TONUSDC")
@click.option("--usd", required=True, type=float, help="USD/USDC amount to sell")
@click.option("--yes", "assume_yes", is_flag=True, help="skip typed confirmation")
def kraken_sell(pair, usd, assume_yes):
    """Sell ~--usd of --pair (market order)."""
    asyncio.run(_kraken_spot(pair, "sell", usd, assume_yes))

@kraken.command("flatten")
@click.option("--yes", "assume_yes", is_flag=True, help="skip typed confirmation")
def kraken_flatten(assume_yes):
    """Sell all non-USDC crypto holdings back to USDC."""
    async def _run():
        from auramaur.exchange.kraken import KrakenSpotClient
        from auramaur.exchange.models import OrderSide
        s = Settings()
        k = KrakenSpotClient(s)
        try:
            bal = await k.get_balance()
            crypto = {a: v for a, v in bal.items()
                      if v > 0 and a != "USDC" and not a.startswith("Z")}
            plans = []
            for a, v in crypto.items():
                price = await k.get_price(f"{a}USDC")
                if price and v * price >= 2.0:
                    plans.append((f"{a}USDC", v, v * price))
            if not plans:
                console.print("Nothing to flatten (no non-dust crypto with a USDC pair).")
                return
            for pair, v, val in plans:
                console.print(f"  SELL {v} [cyan]{pair}[/] (~${val:.2f})")
            console.print(f"  total ~${sum(p[2] for p in plans):.2f}  "
                          + ("[bold red]LIVE[/]" if s.is_live else "[green]paper[/]"))
            if not assume_yes and s.is_live:
                if click.prompt("Type FLATTEN to confirm", default="").strip() != "FLATTEN":
                    console.print("[yellow]Aborted.[/]")
                    return
            for pair, v, val in plans:
                res = await k.place_spot_order(pair, OrderSide.SELL, round(v, 8),
                                               purpose="manual", max_usd=val * 1.1)
                console.print(f"  {pair}: {res.status} {res.error_message}")
        finally:
            await k.close()
    asyncio.run(_run())
