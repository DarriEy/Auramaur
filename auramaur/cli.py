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
@click.option("--exchange", default=None, type=click.Choice(["polymarket", "kalshi", "ibkr"]), help="Run only a specific exchange (isolated instance)")
def run(agent: bool, exchange: str | None):
    """Start the bot."""
    settings = Settings()

    if agent:
        settings.analysis.mode = "agent"
        if settings.is_live:
            console.print("[bold red]AGENT MODE[/] — [bold]LIVE TRADING[/]")
        else:
            console.print("[bold yellow]AGENT MODE[/] — paper trading")
    if exchange:
        console.print(f"[bold blue]Starting Auramaur bot (exchange: {exchange})...[/]")
    else:
        console.print("[bold blue]Starting Auramaur bot...[/]")
    bot = AuramaurBot(settings=settings, exchange_filter=exchange)
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
                from py_clob_client.client import ClobClient
                from py_clob_client.clob_types import ApiCreds
                client = ClobClient(
                    "https://clob.polymarket.com",
                    key=settings.polygon_private_key,
                    chain_id=137,
                )
                client.set_api_creds(ApiCreds(
                    api_key=settings.polymarket_api_key,
                    api_secret=settings.polymarket_api_secret,
                    api_passphrase=settings.polymarket_passphrase,
                ))
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
