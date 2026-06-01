"""Rich CLI dashboard and Click commands."""

from __future__ import annotations

import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::RuntimeWarning"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import asyncio

import click
import structlog
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

from config.settings import Settings
from auramaur.bot import AuramaurBot
from auramaur.db.database import Database

console = Console()
log = structlog.get_logger()


def _make_dashboard_layout(stats: dict) -> Layout:
    """Build the dashboard layout."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )

    # Header
    mode = "[bold red]LIVE[/]" if stats.get("live") else "[bold green]PAPER[/]"
    header = Panel(
        Text.from_markup(f"[bold]AURAMAUR[/] Polymarket Bot  |  Mode: {mode}  |  Kill Switch: {'[red]ACTIVE[/]' if stats.get('kill_switch') else '[green]OFF[/]'}"),
        style="bold blue",
    )
    layout["header"].update(header)

    # Body: positions + signals
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )

    # Portfolio table
    portfolio_table = Table(title="Portfolio", expand=True)
    portfolio_table.add_column("Market", style="cyan", max_width=40)
    portfolio_table.add_column("Side", style="bold")
    portfolio_table.add_column("Size", justify="right")
    portfolio_table.add_column("Avg Price", justify="right")
    portfolio_table.add_column("Current", justify="right")
    portfolio_table.add_column("PnL", justify="right")

    for pos in stats.get("positions", []):
        pnl = pos.get("pnl", 0)
        pnl_style = "green" if pnl >= 0 else "red"
        portfolio_table.add_row(
            pos.get("question", pos.get("market_id", ""))[:40],
            pos.get("side", ""),
            f"${pos.get('size', 0):.2f}",
            f"{pos.get('avg_price', 0):.3f}",
            f"{pos.get('current_price', 0):.3f}",
            f"[{pnl_style}]${pnl:.2f}[/]",
        )

    layout["left"].update(Panel(portfolio_table))

    # Recent signals table
    signals_table = Table(title="Recent Signals", expand=True)
    signals_table.add_column("Market", style="cyan", max_width=35)
    signals_table.add_column("Claude P", justify="right")
    signals_table.add_column("Market P", justify="right")
    signals_table.add_column("Edge", justify="right")
    signals_table.add_column("Action", style="bold")

    for sig in stats.get("signals", [])[:10]:
        edge = sig.get("edge", 0)
        edge_style = "green" if edge > 0 else "red"
        signals_table.add_row(
            sig.get("question", sig.get("market_id", ""))[:35],
            f"{sig.get('claude_prob', 0):.3f}",
            f"{sig.get('market_prob', 0):.3f}",
            f"[{edge_style}]{edge:.1f}%[/]",
            sig.get("action", ""),
        )

    layout["right"].update(Panel(signals_table))

    # Footer
    balance = stats.get("balance")
    total_pnl = stats.get("total_pnl", 0)
    pnl_style = "green" if total_pnl >= 0 else "red"
    balance_str = "[dim]n/a[/]" if balance is None else f"[bold]${balance:.2f}[/]"
    footer = Panel(
        Text.from_markup(
            f"Balance: {balance_str}  |  "
            f"PnL: [{pnl_style}]${total_pnl:.2f}[/]  |  "
            f"Trades: {stats.get('trade_count', 0)}  |  "
            f"Drawdown: {stats.get('drawdown', 0):.1f}%  |  "
            f"Positions: {stats.get('position_count', 0)}"
        ),
    )
    layout["footer"].update(footer)

    return layout


async def _get_dashboard_stats(db: Database, settings: Settings) -> dict:
    """Gather stats for dashboard display."""
    stats = {
        "live": settings.is_live,
        "kill_switch": settings.kill_switch_active,
    }

    # Positions — scope to current mode so paper state doesn't show in live
    # dashboards (and vice versa).
    is_paper_flag = 0 if settings.is_live else 1
    rows = await db.fetchall(
        """SELECT p.*, m.question FROM portfolio p
           LEFT JOIN markets m ON p.market_id = m.id
           WHERE p.is_paper = ?""",
        (is_paper_flag,),
    )
    stats["positions"] = [
        {
            "market_id": r["market_id"], "question": r["question"] or r["market_id"],
            "side": r["side"], "size": r["size"], "avg_price": r["avg_price"],
            "current_price": r["current_price"] or r["avg_price"],
            "pnl": r["unrealized_pnl"] or 0,
        }
        for r in rows
    ]
    stats["position_count"] = len(stats["positions"])

    # Recent signals
    rows = await db.fetchall(
        """SELECT s.*, m.question FROM signals s
           LEFT JOIN markets m ON s.market_id = m.id
           ORDER BY s.timestamp DESC LIMIT 20"""
    )
    stats["signals"] = [
        {
            "market_id": r["market_id"], "question": r["question"] or r["market_id"],
            "claude_prob": r["claude_prob"], "market_prob": r["market_prob"],
            "edge": r["edge"], "action": r["action"] or "",
        }
        for r in rows
    ]

    # Balance and PnL — filter trades to the current mode.
    row = await db.fetchone(
        "SELECT COUNT(*) as cnt, COALESCE(SUM(pnl), 0) as total_pnl FROM trades WHERE is_paper = ?",
        (is_paper_flag,),
    )
    stats["trade_count"] = row["cnt"] if row else 0
    stats["total_pnl"] = row["total_pnl"] if row else 0

    # Drawdown
    row = await db.fetchone(
        "SELECT max_drawdown FROM daily_stats ORDER BY date DESC LIMIT 1"
    )
    stats["drawdown"] = row["max_drawdown"] if row else 0

    if settings.is_live:
        # Live balance should be queried from the CLOB on demand. The running
        # bot's portfolio monitor already displays on-chain cash via the
        # syncer, so this dashboard just reports realized PnL from live fills
        # and leaves balance undefined.
        stats["balance"] = None
    else:
        stats["balance"] = settings.execution.paper_initial_balance + stats["total_pnl"]

    return stats


@click.group()
def main():
    """Auramaur — Polymarket prediction market trading bot."""
    pass


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
        if settings.hybrid.market_maker_auto_enable:
            settings.market_maker.enabled = True
        mode = "[bold red]LIVE TRADING[/]" if settings.is_live else "paper trading"
        console.print(f"[bold cyan]HYBRID MODE[/] — {mode}")
        console.print(
            "[dim]  Pillars: arb (60s) + news speed (30s) + domain LLM + market making[/]"
        )
    if exchange:
        console.print(f"[bold blue]Starting Auramaur bot (exchange: {exchange})...[/]")
    else:
        console.print("[bold blue]Starting Auramaur bot...[/]")
    bot = AuramaurBot(settings=settings, exchange_filter=exchange, hybrid=hybrid)
    asyncio.run(bot.run())


@main.command()
def dashboard():
    """Show live dashboard (read-only)."""

    async def _dashboard():
        settings = Settings()
        db = Database()
        await db.connect()

        with Live(console=console, refresh_per_second=1) as live:
            while True:
                try:
                    stats = await _get_dashboard_stats(db, settings)
                    layout = _make_dashboard_layout(stats)
                    live.update(layout)
                except Exception as e:
                    console.print(f"[red]Dashboard error: {e}[/]")
                await asyncio.sleep(settings.intervals.dashboard_refresh_seconds)

    asyncio.run(_dashboard())


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
@click.argument("value", required=False, type=float)
def risk(value):
    """Show or set the global risk-tolerance lever (0=conservative..100=YOLO).

    Live — a running bot picks up the new value on its next risk check, no restart.
    """
    from auramaur.risk.tolerance import current_tolerance, set_tolerance
    s = Settings()
    if value is None:
        console.print(f"risk_tolerance = [bold]{current_tolerance(s):.0f}[/]/100")
        return
    if not 0 <= value <= 100:
        console.print("[red]value must be between 0 and 100[/]")
        return
    set_tolerance(value)
    label = ("most conservative" if value <= 15 else "conservative" if value < 45
             else "neutral" if value <= 55 else "aggressive" if value < 85 else "YOLO")
    console.print(f"risk_tolerance -> [bold]{value:.0f}[/]/100 ([cyan]{label}[/]) — "
                  "live; running bot applies it next cycle.")


@main.command()
def gates():
    """Show feature gates — which experimental switches the data has earned."""
    from auramaur.monitoring import gates as g
    console.print(g.render(g.gather(Settings())))


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


@main.command()
@click.argument("query")
@click.option("--limit", default=20, help="Number of markets to show")
def scan(query: str, limit: int):
    """Scan Polymarket for markets matching a query."""

    async def _scan():
        from auramaur.exchange.gamma import GammaClient
        gamma = GammaClient()
        try:
            if query == "top":
                markets = await gamma.get_markets(limit=limit)
            else:
                markets = await gamma.search_markets(query, limit=limit)

            table = Table(title=f"Markets: {query}")
            table.add_column("ID", style="dim", max_width=12)
            table.add_column("Question", style="cyan", max_width=50)
            table.add_column("Yes", justify="right", style="green")
            table.add_column("No", justify="right", style="red")
            table.add_column("Volume", justify="right")
            table.add_column("Liquidity", justify="right")

            for m in markets:
                table.add_row(
                    m.id[:12], m.question[:50],
                    f"{m.outcome_yes_price:.3f}", f"{m.outcome_no_price:.3f}",
                    f"${m.volume:,.0f}", f"${m.liquidity:,.0f}",
                )

            console.print(table)
        finally:
            await gamma.close()

    asyncio.run(_scan())


@main.command()
def redeem_check():
    """List Polymarket positions ready to redeem for USDC."""

    async def _check():
        from auramaur.broker.redeemer import (
            fetch_redeemable_positions, summarize_redemptions,
        )
        settings = Settings()
        proxy = settings.polymarket_proxy_address
        if not proxy:
            console.print("[red]POLYMARKET_PROXY_ADDRESS not set in environment.[/]")
            return

        console.print(f"Checking redeemable positions for [cyan]{proxy[:10]}…{proxy[-6:]}[/]\n")
        try:
            positions = await fetch_redeemable_positions(proxy)
        except Exception as e:
            console.print(f"[red]Failed to fetch positions: {e}[/]")
            return

        if not positions:
            console.print("[green]No positions to redeem — everything's settled or still open.[/]")
            return

        summary = summarize_redemptions(positions)

        now = [p for p in positions if p.redeemable_now]
        pending = [p for p in positions if p.status == "pending_oracle"]

        def _render_table(title: str, items: list) -> None:
            if not items:
                return
            table = Table(title=title, show_lines=False)
            table.add_column("Market", max_width=55)
            table.add_column("Side", width=4)
            table.add_column("Size", justify="right")
            table.add_column("Cost", justify="right")
            table.add_column("Payout", justify="right")
            table.add_column("P&L", justify="right")
            table.add_column("Type", width=8)
            for p in sorted(items, key=lambda x: -x.payout):
                pnl_color = "green" if p.realized_pnl >= 0 else "red"
                win_marker = "[green]✓[/]" if p.is_winner else "[red]✗[/]"
                market_type = "NegRisk" if p.neg_risk else "CTF"
                table.add_row(
                    f"{win_marker} {p.title[:53]}",
                    p.outcome,
                    f"{p.size:.1f}",
                    f"${p.cost_basis:.2f}",
                    f"${p.payout:.2f}",
                    f"[{pnl_color}]{p.realized_pnl:+.2f}[/]",
                    market_type,
                )
            console.print(table)
            console.print()

        _render_table("Redeemable Now (click Redeem on Polymarket)", now)
        _render_table("Pending UMA Oracle (resolved, awaiting confirmation)", pending)

        console.print(
            f"[bold]Redeemable now:[/] {summary['redeemable_now']}  "
            f"([green]${summary['payout_now_usdc']:.2f}[/] payout, "
            f"net [{'green' if summary['net_pnl_now'] >= 0 else 'red'}]"
            f"${summary['net_pnl_now']:+.2f}[/])"
        )
        console.print(
            f"[bold]Pending oracle:[/] {summary['pending_oracle']}  "
            f"([yellow]${summary['payout_pending_usdc']:.2f}[/] expected)"
        )
        if summary['neg_risk_count'] > 0:
            console.print(
                f"[yellow]Note:[/] {summary['neg_risk_count']} NegRisk positions — "
                "these need the NegRiskAdapter contract for on-chain redemption."
            )
        console.print()
        console.print(
            "[dim]To redeem, open Polymarket → Portfolio → 'Redeem All' button.[/]"
        )

    asyncio.run(_check())


@main.command("dust-exit")
@click.option("--max-notional", default=5.0, type=float,
              help="Treat open positions worth less than this (current value, $) as dust.")
@click.option("--exchange", default=None, type=click.Choice(["polymarket", "kalshi"]),
              help="Limit to a single exchange (default: all).")
@click.option("--execute", is_flag=True, default=False,
              help="Place the closing orders. Without this flag it's a preview only. "
                   "Live vs paper still follows the normal three-gate model "
                   "(AURAMAUR_LIVE + execution.live); without those gates open the "
                   "sells are simulated.")
@click.option("--yes", is_flag=True, default=False,
              help="Skip the confirmation prompt (for non-interactive use).")
@click.option("--limit", default=0, type=int,
              help="Cap how many positions to close this run (0 = no cap).")
@click.option("--category", default=None,
              help="Only positions in this category (comma-separated for several, "
                   "e.g. 'sports,esports'). The category is recomputed live with the "
                   "current classifier, not the stored label, so stale mislabels "
                   "don't leak in.")
def dust_exit(max_notional: float, exchange: str | None, execute: bool, yes: bool,
              limit: int, category: str | None):
    """List small 'dust' positions and optionally close them.

    Dust is open positions whose current value is below ``--max-notional``.
    Closing them frees capital and trims position count — which un-starves
    Kelly sizing and lowers the regime-driven minimum-edge bar.

    Use ``--category`` to target a specific category (e.g. a blocked one like
    sports) — the category is recomputed live so freshly-fixed classifications
    apply even before the next scan rewrites the stored label.

    Safety: this attaches to the primary database with an exclusive lock, so
    it refuses to run while the live bot is running (which would risk
    double-selling). Stop the bot first.
    """

    async def _run():
        from auramaur.exchange.models import ExitReason, OrderSide, Position, TokenType
        from auramaur.strategy.classifier import classify_market

        cat_filter = (
            {c.strip().lower() for c in category.split(",") if c.strip()}
            if category else None
        )

        settings = Settings()
        bot = AuramaurBot(settings=settings, db_path="auramaur.db", exchange_filter=exchange)
        try:
            await bot._init_components()
        except RuntimeError as e:
            console.print(f"[red]Cannot start dust-exit:[/] {e}")
            console.print(
                "[yellow]Auramaur looks like it's already running. Stop the bot first — "
                "running dust-exit alongside it risks double-selling the same position.[/]"
            )
            return

        comps = bot._components
        db = comps["db"]
        discoveries = comps["discoveries"]
        exchanges = comps["exchanges"]
        alerts = comps["alerts"]
        is_paper = 0 if settings.is_live else 1

        def _price_for(market, token: TokenType) -> float:
            if token == TokenType.NO:
                no = market.outcome_no_price
                return no if no > 0.01 else (1.0 - market.outcome_yes_price)
            return market.outcome_yes_price

        def _sellable(name: str, p: Position) -> bool:
            # Mirrors the hard floors in the bot's exit paths: Polymarket needs
            # >=5 tokens, Kalshi >=1 contract, and a price the book can fill.
            if (p.current_price or 0.0) < 0.01:
                return False
            return p.size >= 5 if name == "polymarket" else p.size >= 1

        try:
            # ---- Discover dust, refreshing prices for candidates only ----
            dust: dict[str, list[Position]] = {}
            for name, discovery in discoveries.items():
                if exchange and name != exchange:
                    continue
                if exchanges.get(name) is None:
                    continue
                rows = await db.fetchall(
                    "SELECT p.market_id, p.side, p.size, p.avg_price, p.current_price, "
                    "p.category, p.token, p.token_id, "
                    "COALESCE(m.question, '') AS question, "
                    "COALESCE(m.description, '') AS description "
                    "FROM portfolio p LEFT JOIN markets m ON p.market_id = m.id "
                    "WHERE p.exchange = ? AND p.is_paper = ? AND p.size > 0",
                    (name, is_paper),
                )
                found: list[Position] = []
                for row in rows:
                    tok = TokenType(row["token"]) if row["token"] in ("YES", "NO") else TokenType.YES
                    # Recompute the category with the current classifier so the
                    # filter targets the true category, not a stale stored label
                    # (the whole reason mislabeled markets got stuck blocked).
                    live_cat = (
                        classify_market(row["question"], row["description"])
                        if row["question"] else (row["category"] or "")
                    )
                    if cat_filter and live_cat.lower() not in cat_filter:
                        continue
                    p = Position(
                        market_id=row["market_id"], exchange=name,
                        side=OrderSide(row["side"]) if row["side"] else OrderSide.BUY,
                        size=float(row["size"]), avg_price=float(row["avg_price"] or 0.0),
                        current_price=float(row["current_price"] or 0.0),
                        category=live_cat,
                        token=tok, token_id=row["token_id"] or "",
                    )
                    # Pre-filter on the stored value so we only hit the API for
                    # likely-dust positions, then refresh their price to confirm.
                    if p.size * (p.current_price or 0.0) > max_notional:
                        continue
                    try:
                        m = await discovery.get_market(p.market_id)
                        if m is not None:
                            fresh = _price_for(m, p.token)
                            if fresh and fresh > 0:
                                p.current_price = round(fresh, 4)
                    except Exception:
                        pass
                    if p.size * (p.current_price or 0.0) <= max_notional:
                        found.append(p)
                found.sort(key=lambda x: x.size * (x.current_price or 0.0))
                if found:
                    dust[name] = found

            scope = f" in [{category}]" if category else ""
            total = sum(len(v) for v in dust.values())
            if total == 0:
                console.print(f"[green]No dust positions under ${max_notional:.2f}{scope}.[/]")
                return

            # ---- Show what we found ----
            sellables: list[tuple[str, object, object, Position]] = []
            grand_value = grand_pnl = 0.0
            for name, positions in dust.items():
                table = Table(title=f"{name} — dust under ${max_notional:.2f}{scope}")
                for col in ("Market", "Cat", "Tok", "Size", "Price", "Value", "PnL", ""):
                    table.add_column(col)
                for p in positions:
                    value = p.size * (p.current_price or 0.0)
                    grand_value += value
                    grand_pnl += p.unrealized_pnl
                    ok = _sellable(name, p)
                    if ok:
                        sellables.append((name, discoveries[name], exchanges[name], p))
                    table.add_row(
                        p.market_id[:30], (p.category or "?")[:12], p.token.value, f"{p.size:.1f}",
                        f"${p.current_price:.3f}", f"${value:.2f}",
                        f"${p.unrealized_pnl:+.2f}",
                        "[green]sell[/]" if ok else "[dim]redeem-only[/]",
                    )
                console.print(table)

            n_unsellable = total - len(sellables)
            console.print(
                f"\n[bold]{total}[/] dust positions — ${grand_value:.2f} value, "
                f"${grand_pnl:+.2f} PnL.  [green]{len(sellables)} sellable[/], "
                f"[dim]{n_unsellable} redeem-only[/] (resolved/near-zero — use "
                f"[cyan]auramaur redeem-check[/])."
            )

            if not execute:
                console.print(
                    f"\n[dim]Preview only. Re-run with [cyan]--execute[/] to close the "
                    f"{len(sellables)} sellable position(s).[/]"
                )
                return

            if not sellables:
                console.print("\n[yellow]Nothing sellable to close.[/]")
                return

            if limit and limit > 0:
                sellables = sellables[:limit]

            mode = "LIVE — real sell orders" if settings.is_live else "PAPER — simulated sells"
            console.print(f"\n[bold]Execution mode:[/] {mode}")
            if not yes:
                if not click.confirm(f"Close {len(sellables)} sellable dust position(s)?"):
                    console.print("[yellow]Aborted.[/]")
                    return

            closed = failed = 0
            for name, discovery, exch, p in sellables:
                try:
                    if name == "polymarket":
                        ok = await bot._execute_poly_exit(p, ExitReason.DUST_CLEANUP, discovery, exch, alerts)
                    else:
                        ok = await bot._execute_kalshi_exit(p, ExitReason.DUST_CLEANUP, discovery, exch, alerts)
                except Exception as e:
                    console.print(f"[red]ERROR closing {p.market_id[:24]}: {e}[/]")
                    ok = False
                if ok:
                    closed += 1
                    console.print(f"[green]✓[/] {name} {p.market_id[:30]} — sell submitted")
                else:
                    failed += 1
                    console.print(f"[yellow]✗[/] {name} {p.market_id[:30]} — not closed (too small / no fill path)")

            console.print(f"\n[bold]Done:[/] {closed} submitted, {failed} skipped.")
        finally:
            await bot.shutdown()

    asyncio.run(_run())


@main.command()
def status():
    """Show current bot status."""

    async def _status():
        settings = Settings()
        db = Database()
        await db.connect()
        stats = await _get_dashboard_stats(db, settings)

        console.print(f"Mode: {'[red]LIVE[/]' if stats['live'] else '[green]PAPER[/]'}")
        console.print(f"Kill Switch: {'[red]ACTIVE[/]' if stats['kill_switch'] else '[green]OFF[/]'}")

        # Show real Polymarket balance when live
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
                orders = client.get_orders()
                console.print(f"Polymarket: [green]connected[/] — {len(orders)} open orders")
            except Exception as e:
                console.print(f"Polymarket: [red]error[/] — {str(e)[:60]}")

        if stats['balance'] is None:
            console.print("Balance: [dim]n/a (live — see running bot for on-chain cash)[/]")
        else:
            console.print(f"Balance: ${stats['balance']:.2f}")
        console.print(f"PnL: ${stats['total_pnl']:.2f}")
        console.print(f"Trades: {stats['trade_count']}")
        console.print(f"Open Positions: {stats['position_count']}")
        console.print(f"Drawdown: {stats['drawdown']:.1f}%")

        await db.close()

    asyncio.run(_status())


@main.command()
@click.option("--days", default=30, help="Number of days to backtest")
@click.option("--min-edge", default=None, type=float, help="Minimum edge % to trade (overrides config)")
@click.option("--kelly-fraction", default=None, type=float, help="Kelly fraction (overrides config)")
@click.option("--compare", is_flag=True, help="Compare two strategies (default params vs aggressive)")
def backtest(days: int, min_edge: float | None, kelly_fraction: float | None, compare: bool):
    """Run backtest on historical signals."""
    from rich.columns import Columns

    from auramaur.backtest.engine import BacktestEngine

    async def _backtest():
        settings = Settings()
        db = Database()
        await db.connect()

        try:
            engine = BacktestEngine(db, settings)

            if compare:
                # A/B comparison: conservative vs aggressive
                params_a = {
                    "min_edge_pct": settings.risk.min_edge_pct,
                    "kelly_fraction": settings.kelly.fraction,
                }
                params_b = {
                    "min_edge_pct": max(1.0, settings.risk.min_edge_pct - 2.0),
                    "kelly_fraction": min(0.5, settings.kelly.fraction * 1.6),
                }
                comparison = await engine.compare_strategies(params_a, params_b, days=days)
                _display_comparison(comparison)
            else:
                result = await engine.run(
                    days=days,
                    min_edge_pct=min_edge,
                    kelly_fraction=kelly_fraction,
                )
                _display_backtest_result(result, days)
        finally:
            await db.close()

    asyncio.run(_backtest())


def _display_backtest_result(result, days: int):
    """Render backtest results with Rich tables and panels."""
    from auramaur.backtest.engine import BacktestResult

    if result.total_trades == 0:
        console.print(Panel(
            "[yellow]No resolved signals found for backtesting.[/]\n"
            "The backtest requires signals with matching calibration resolutions.\n"
            "Run the bot to collect data first.",
            title="Backtest - No Data",
            border_style="yellow",
        ))
        return

    # --- Overall Performance Panel ---
    pnl_style = "green" if result.total_pnl >= 0 else "red"
    sharpe_style = "green" if result.sharpe_ratio >= 1.0 else ("yellow" if result.sharpe_ratio >= 0 else "red")
    brier_style = "green" if result.brier_score < 0.2 else ("yellow" if result.brier_score < 0.3 else "red")
    dd_style = "green" if result.max_drawdown_pct < 10 else ("yellow" if result.max_drawdown_pct < 20 else "red")

    overview = Table(show_header=False, expand=True, box=None, padding=(0, 2))
    overview.add_column("Metric", style="bold")
    overview.add_column("Value", justify="right")
    overview.add_column("Metric", style="bold")
    overview.add_column("Value", justify="right")

    overview.add_row(
        "Total PnL", f"[{pnl_style}]${result.total_pnl:+.2f}[/]",
        "Total Trades", str(result.total_trades),
    )
    overview.add_row(
        "Win Rate", f"{result.win_rate:.1f}%",
        "Wins / Losses", f"[green]{result.winning_trades}[/] / [red]{result.losing_trades}[/]",
    )
    overview.add_row(
        "Sharpe Ratio", f"[{sharpe_style}]{result.sharpe_ratio:.2f}[/]",
        "Avg PnL/Trade", f"${result.avg_pnl_per_trade:+.2f}",
    )
    overview.add_row(
        "Brier Score", f"[{brier_style}]{result.brier_score:.4f}[/]",
        "Accuracy", f"{result.accuracy:.1f}%",
    )
    overview.add_row(
        "Max Drawdown", f"[{dd_style}]{result.max_drawdown_pct:.1f}%[/]",
        "Avg Edge", f"{result.avg_edge:.1f}%",
    )
    overview.add_row(
        "Best Trade", f"[green]${result.best_trade:+.2f}[/]",
        "Worst Trade", f"[red]${result.worst_trade:+.2f}[/]",
    )

    console.print(Panel(overview, title=f"Backtest Results ({days} days)", border_style="blue"))

    # --- Category Breakdown ---
    if result.by_category:
        cat_table = Table(title="Performance by Category", expand=True)
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Trades", justify="right")
        cat_table.add_column("Win Rate", justify="right")
        cat_table.add_column("PnL", justify="right")
        cat_table.add_column("Avg Edge", justify="right")
        cat_table.add_column("Brier", justify="right")

        sorted_cats = sorted(result.by_category.items(), key=lambda x: x[1]["pnl"], reverse=True)
        for cat_name, stats in sorted_cats:
            pnl_s = "green" if stats["pnl"] >= 0 else "red"
            cat_table.add_row(
                cat_name,
                str(stats["trades"]),
                f"{stats['win_rate']:.1f}%",
                f"[{pnl_s}]${stats['pnl']:+.2f}[/]",
                f"{stats['avg_edge']:.1f}%",
                f"{stats['brier_score']:.4f}",
            )

        console.print(cat_table)

    # --- PnL Curve (sparkline) ---
    if result.pnl_curve:
        _display_pnl_curve(result.pnl_curve)

    # --- Top/Bottom Trades ---
    if result.trade_details:
        sorted_trades = sorted(result.trade_details, key=lambda t: t["pnl"], reverse=True)

        top_n = min(5, len(sorted_trades))

        trades_table = Table(title="Top Trades", expand=True)
        trades_table.add_column("Market", style="cyan", max_width=45)
        trades_table.add_column("Claude P", justify="right")
        trades_table.add_column("Market P", justify="right")
        trades_table.add_column("Edge %", justify="right")
        trades_table.add_column("Outcome", justify="center")
        trades_table.add_column("PnL", justify="right")

        for t in sorted_trades[:top_n]:
            pnl_s = "green" if t["pnl"] >= 0 else "red"
            outcome_s = "[green]YES[/]" if t["actual_outcome"] == 1 else "[red]NO[/]"
            trades_table.add_row(
                t["question"][:45],
                f"{t['claude_prob']:.3f}",
                f"{t['market_prob']:.3f}",
                f"{t['edge_pct']:.1f}%",
                outcome_s,
                f"[{pnl_s}]${t['pnl']:+.2f}[/]",
            )

        # Add separator then worst trades
        if len(sorted_trades) > top_n:
            trades_table.add_section()
            for t in sorted_trades[-top_n:]:
                pnl_s = "green" if t["pnl"] >= 0 else "red"
                outcome_s = "[green]YES[/]" if t["actual_outcome"] == 1 else "[red]NO[/]"
                trades_table.add_row(
                    t["question"][:45],
                    f"{t['claude_prob']:.3f}",
                    f"{t['market_prob']:.3f}",
                    f"{t['edge_pct']:.1f}%",
                    outcome_s,
                    f"[{pnl_s}]${t['pnl']:+.2f}[/]",
                )

        console.print(trades_table)


def _display_pnl_curve(pnl_curve: list[float]):
    """Display a simple ASCII PnL curve."""
    if not pnl_curve:
        return

    # Normalize to a fixed height
    height = 10
    width = min(60, len(pnl_curve))

    # Resample if we have more points than width
    if len(pnl_curve) > width:
        step = len(pnl_curve) / width
        sampled = [pnl_curve[int(i * step)] for i in range(width)]
    else:
        sampled = pnl_curve

    min_val = min(sampled)
    max_val = max(sampled)
    val_range = max_val - min_val

    if val_range == 0:
        # Flat line
        line = Text("  " + "-" * len(sampled))
        console.print(Panel(line, title="Cumulative PnL", border_style="blue"))
        return

    # Build the chart rows
    rows: list[str] = []
    for row in range(height, -1, -1):
        threshold = min_val + (val_range * row / height)
        line_chars = []
        for val in sampled:
            if val >= threshold:
                line_chars.append("*")
            else:
                line_chars.append(" ")
        # Y-axis label
        label = f"${threshold:>8.2f} |"
        rows.append(label + "".join(line_chars))

    # X-axis
    rows.append(" " * 10 + "+" + "-" * len(sampled))
    rows.append(" " * 10 + f" Trade 1{' ' * max(0, len(sampled) - 14)}Trade {len(pnl_curve)}")

    chart_text = "\n".join(rows)
    console.print(Panel(chart_text, title="Cumulative PnL Curve", border_style="blue"))


def _display_comparison(comparison: dict):
    """Display A/B strategy comparison."""
    table = Table(title="Strategy Comparison (A/B Test)", expand=True)
    table.add_column("Metric", style="bold")
    table.add_column("Strategy A", justify="right")
    table.add_column("Strategy B", justify="right")
    table.add_column("Diff", justify="right")

    a = comparison["strategy_a"]
    b = comparison["strategy_b"]

    # Params header
    a_params = ", ".join(f"{k}={v}" for k, v in a["params"].items())
    b_params = ", ".join(f"{k}={v}" for k, v in b["params"].items())
    table.add_row("Parameters", a_params, b_params, "")
    table.add_section()

    metrics = [
        ("Total Trades", "total_trades", "", False),
        ("Total PnL", "total_pnl", "$", True),
        ("Win Rate", "win_rate", "%", True),
        ("Sharpe Ratio", "sharpe_ratio", "", True),
        ("Max Drawdown", "max_drawdown_pct", "%", False),
        ("Brier Score", "brier_score", "", False),
        ("Avg Edge", "avg_edge", "%", True),
        ("Best Trade", "best_trade", "$", True),
        ("Worst Trade", "worst_trade", "$", False),
    ]

    for label, key, unit, higher_better in metrics:
        a_val = a[key]
        b_val = b[key]
        diff = a_val - b_val

        if unit == "$":
            a_str = f"${a_val:.2f}"
            b_str = f"${b_val:.2f}"
            d_str = f"${diff:+.2f}"
        elif unit == "%":
            a_str = f"{a_val:.1f}%"
            b_str = f"{b_val:.1f}%"
            d_str = f"{diff:+.1f}%"
        else:
            a_str = f"{a_val}"
            b_str = f"{b_val}"
            d_str = f"{diff:+.2f}"

        if higher_better:
            diff_style = "green" if diff > 0 else ("red" if diff < 0 else "")
        else:
            diff_style = "red" if diff > 0 else ("green" if diff < 0 else "")

        table.add_row(label, a_str, b_str, f"[{diff_style}]{d_str}[/]" if diff_style else d_str)

    console.print(table)

    winner = comparison["winner"]
    console.print(Panel(
        f"[bold]Winner: Strategy {winner}[/]  |  "
        f"PnL advantage: ${comparison['pnl_diff']:+.2f}  |  "
        f"Sharpe advantage: {comparison['sharpe_diff']:+.2f}",
        border_style="green" if winner == "A" else "yellow",
    ))


@main.command()
@click.option("--exchange", default=None, help="Exchange to evaluate (e.g. kalshi)")
@click.option("--days", default=7, help="Window in days (default 7)")
@click.option("--log-file", default="auramaur.log", help="Path to structlog output file")
@click.option("--json-output", is_flag=True, default=False, help="Emit JSON instead of table")
def readiness(exchange, days, log_file, json_output):
    """Evaluate live-trading readiness criteria.

    Prints PASS/FAIL/INSUFFICIENT_DATA per criterion. Exits 0 if all pass, 1 otherwise.
    """
    from dataclasses import asdict
    from pathlib import Path
    from auramaur.monitoring.readiness import evaluate_readiness

    async def _run():
        settings = Settings()
        if exchange:
            fee_rate = settings.arbitrage.exchange_fees.get(exchange, 0.07)
        else:
            fee_rate = 0.07

        db = Database()
        await db.connect()
        try:
            return await evaluate_readiness(
                db,
                log_file=Path(log_file),
                exchange=exchange,
                days=days,
                fee_rate=fee_rate,
            )
        finally:
            await db.close()

    report = asyncio.run(_run())

    if json_output:
        payload = {
            "timestamp": report.timestamp.isoformat(),
            "exchange": report.exchange,
            "window_days": report.window_days,
            "overall_pass": report.overall_pass,
            "criteria": [asdict(c) for c in report.criteria],
        }
        import json as _json
        console.print_json(_json.dumps(payload))
    else:
        _render_readiness_table(report)

    if not report.overall_pass:
        raise click.exceptions.Exit(1)


def _render_readiness_table(report) -> None:
    status_styles = {
        "PASS": "[green]PASS[/]",
        "FAIL": "[red]FAIL[/]",
        "INSUFFICIENT_DATA": "[yellow]INSUFFICIENT_DATA[/]",
    }
    header = (
        f"Readiness — {report.exchange or 'all exchanges'} "
        f"(window: {report.window_days}d)"
    )
    table = Table(title=header, expand=True)
    table.add_column("Criterion", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Value", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Notes", overflow="fold")

    for c in report.criteria:
        table.add_row(
            c.name,
            status_styles.get(c.status, c.status),
            c.value,
            c.threshold,
            c.detail or "",
        )

    console.print(table)
    overall = "[green]READY[/]" if report.overall_pass else "[red]NOT READY[/]"
    console.print(f"\nOverall: {overall}")


@main.command()
def kill():
    """Activate the kill switch."""
    from pathlib import Path
    Path("KILL_SWITCH").touch()
    console.print("[bold red]KILL SWITCH ACTIVATED[/]")


@main.command()
def unkill():
    """Deactivate the kill switch."""
    from pathlib import Path
    ks = Path("KILL_SWITCH")
    if ks.exists():
        ks.unlink()
        console.print("[bold green]Kill switch deactivated[/]")
    else:
        console.print("Kill switch was not active.")


@main.command("redeem")
@click.option("--submit", is_flag=True, default=False,
              help="Broadcast the Safe transactions. Requires AURAMAUR_LIVE=true, "
                   "execution.live=true, and AURAMAUR_ENABLE_REDEMPTION=true. "
                   "Omit to dry-run (build + sign + show calldata, never broadcast).")
@click.option("--limit", default=10, type=int,
              help="Maximum number of redemptions to attempt in this run.")
@click.option("--min-payout", default=1.0, type=float,
              help="Skip positions with expected payout below this amount (USDC).")
def redeem_cmd(submit: bool, limit: int, min_payout: float):
    """Redeem winning conditional tokens to USDC on-chain.

    Without --submit, this is a safe dry-run that prints what would be sent
    to Polygon but does not broadcast. Review the Safe nonce, calldata size,
    and target contract before flipping the gates.
    """
    async def run():
        import aiohttp
        from auramaur.broker.onchain import OnChainRedeemer
        from auramaur.broker.redeemer import fetch_redeemable_positions

        settings = Settings()
        if not settings.polymarket_proxy_address:
            console.print("[red]polymarket_proxy_address not configured[/]")
            return

        db = Database()
        await db.connect()

        try:
            async with aiohttp.ClientSession() as session:
                positions = await fetch_redeemable_positions(
                    settings.polymarket_proxy_address,
                    session=session,
                    include_pending=False,
                )

            ready = [p for p in positions
                     if p.redeemable_now and p.is_winner and p.payout >= min_payout]
            ready.sort(key=lambda p: p.payout, reverse=True)
            ready = ready[:limit]

            if not ready:
                console.print(f"[yellow]Nothing redeemable with payout ≥ ${min_payout:.2f}[/]")
                return

            total = sum(p.payout for p in ready)
            console.print(f"[bold]{len(ready)} position(s) redeemable — total payout ${total:.2f}[/]")

            redeemer = OnChainRedeemer(settings, db)
            gates_open = redeemer._is_live_submission_allowed()

            if submit and not gates_open:
                console.print("[red]--submit passed but gates are closed. Need all of:[/]")
                console.print(f"  AURAMAUR_LIVE=true           (now: {settings.auramaur_live})")
                console.print(f"  execution.live=true          (now: {settings.execution.live})")
                console.print(f"  AURAMAUR_ENABLE_REDEMPTION=true (now: {settings.auramaur_enable_redemption})")
                console.print("  KILL_SWITCH absent")
                console.print("Running as dry-run instead.")

            do_submit = submit and gates_open

            for pos in ready:
                try:
                    result = await redeemer.redeem(pos, dry_run=not do_submit)
                except Exception as e:
                    console.print(f"[red]ERROR {pos.title[:50]}: {e}[/]")
                    continue

                if result.status == "built":
                    console.print(
                        f"[cyan]built[/]  nonce={result.safe_nonce} "
                        f"payout=${pos.payout:.2f} {pos.title[:60]}"
                    )
                elif result.status == "submitted":
                    console.print(
                        f"[green]submitted[/] tx=0x{result.tx_hash.lstrip('0x')[:10]}... "
                        f"payout=${pos.payout:.2f} {pos.title[:50]}"
                    )
                elif result.status == "confirmed":
                    console.print(
                        f"[bold green]confirmed[/] tx=0x{result.tx_hash.lstrip('0x')[:10]}... "
                        f"payout=${pos.payout:.2f} {pos.title[:50]}"
                    )
                elif result.status == "skipped":
                    console.print(f"[dim]skipped[/] (already recorded) {pos.title[:60]}")
                else:
                    console.print(f"[red]{result.status}[/] {pos.title[:50]}: {result.error}")
        finally:
            await db.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
