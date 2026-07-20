"""Auramaur CLI — reporting commands."""

from __future__ import annotations

import asyncio

import click
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from auramaur.db.database import Database
from config.settings import Settings

from auramaur.cli._base import console, main

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

@main.command("pnl")
@click.option("--paper", "paper", is_flag=True, default=False,
              help="Show the paper book instead of live.")
@click.option("--backfill", is_flag=True, default=False,
              help="Reconstruct ledger rows from fills + resolved outcomes first "
                   "(idempotent — re-running cannot double-count).")
def pnl(paper: bool, backfill: bool):
    """Unified realized-P&L scorecard (the pnl_ledger source of truth).

    One row per realization event (sell or settlement), with venue, entry
    strategy and category on every row — by venue / strategy / category /
    month, plus reconciliation against the legacy accountings.
    """

    async def _run():
        from auramaur.broker.ledger import backfill_ledger
        from auramaur.monitoring.ledger_report import (
            gather_ledger_report, render_ledger_report,
        )

        db = Database()
        await db.connect(ensure_schema=False)
        try:
            if backfill:
                written = await backfill_ledger(db)
                console.print(
                    f"[dim]backfill: {written['sell']} sell + "
                    f"{written['settlement']} settlement rows visited "
                    f"(existing refs skipped)[/]"
                )
            state = await gather_ledger_report(db, is_paper=paper)
            console.print(render_ledger_report(state))
        finally:
            await db.close()

    asyncio.run(_run())


@main.command("ibkr-intelligence")
def ibkr_intelligence():
    """Compare Luna/Terra/Sol ETF forecast quality and paper P&L."""
    async def _run():
        db = Database()
        await db.connect(ensure_schema=False)
        try:
            rows = await db.fetchall(
                """SELECT f.model_alias, MAX(f.model) AS model,
                          COUNT(*) AS forecasts,
                          SUM(CASE WHEN f.actual_outcome IS NOT NULL THEN 1 ELSE 0 END) AS resolved,
                          AVG(CASE WHEN f.actual_outcome IS NOT NULL THEN
                            (f.probability-f.actual_outcome)*(f.probability-f.actual_outcome)
                          END) AS brier,
                          AVG(CASE WHEN f.actual_outcome IS NOT NULL THEN
                            CASE WHEN (f.probability >= .5) = (f.actual_outcome = 1)
                                 THEN 1.0 ELSE 0.0 END END) AS accuracy,
                          COALESCE((SELECT SUM(l.pnl) FROM ibkr_etf_ledger l
                            WHERE l.model_alias = f.model_alias), 0) AS realized_pnl,
                          COALESCE((SELECT -SUM(l.pnl) FROM ibkr_etf_ledger l
                            WHERE l.model_alias = f.model_alias
                              AND l.kind = 'intelligence'), 0) AS intelligence_cost,
                          COALESCE((SELECT COUNT(*) FROM ibkr_etf_positions p
                            WHERE p.model_alias = f.model_alias), 0) AS open_positions
                   FROM ibkr_etf_forecasts f GROUP BY f.model_alias
                   ORDER BY f.model_alias""")
            table = Table(title="IBKR ETF OpenAI intelligence comparison")
            for name, justify in (("cell", "left"), ("model", "left"),
                                  ("forecasts", "right"), ("resolved", "right"),
                                  ("accuracy", "right"), ("Brier", "right"),
                                  ("open", "right"), ("intelligence", "right"),
                                  ("net P&L", "right")):
                table.add_column(name, justify=justify)
            for row in rows:
                table.add_row(
                    row["model_alias"], row["model"], str(row["forecasts"]),
                    str(row["resolved"]),
                    f"{row['accuracy']:.1%}" if row["accuracy"] is not None else "—",
                    f"{row['brier']:.3f}" if row["brier"] is not None else "—",
                    str(row["open_positions"]), f"${row['intelligence_cost']:,.4f}",
                    f"${row['realized_pnl']:+,.2f}")
            console.print(table)
        finally:
            await db.close()
    asyncio.run(_run())

@main.command("agent-compare")
@click.option("--agent-db", default="agent.db",
              help="The Hermes agent-trader's isolated ledger.")
@click.option("--auramaur-db", default="auramaur.db",
              help="The bot's ledger (the opponent).")
def agent_compare(agent_db: str, auramaur_db: str):
    """Head-to-head scorecard: the Hermes agent-trader vs Auramaur.

    Agent (paper) vs Bot (paper) is the fair A/B — both simulated on the same
    universe, and the bot's paper book is the strategy ensemble the agent is
    trying to beat. The bot's live book is shown for context. Realized P&L comes
    from each database's pnl_ledger.
    """

    async def _run():
        from auramaur.agentmcp.compare import build_comparison, render_comparison

        data = await build_comparison(agent_db, auramaur_db)
        console.print(render_comparison(data))

    asyncio.run(_run())

@main.command("graduation")
def graduation():
    """Graduation ladder — which (strategy x category) cells have earned live
    capital, per the pnl_ledger record. In observe mode this shows what
    enforce WOULD do."""

    async def _run():
        from auramaur.risk.graduation import GraduationLadder

        settings = Settings()
        db = Database()
        await db.connect(ensure_schema=False)
        try:
            ladder = GraduationLadder(db, settings)
            cells = await ladder.report()
            cfg = settings.graduation
            console.print(
                f"[bold]Graduation ladder[/] — mode [cyan]{cfg.mode}[/], "
                f"min {cfg.min_markets} markets / {cfg.window_days}d window, "
                f"probation x{cfg.probation_multiplier}"
            )
            if not cells:
                console.print("[dim]No ledger history in the window — run "
                              "`auramaur pnl --backfill` first.[/]")
                return
            table = Table()
            table.add_column("strategy", style="cyan")
            table.add_column("category")
            table.add_column("live n/$", justify="right")
            table.add_column("paper n/$", justify="right")
            table.add_column("status")
            table.add_column("would trade as")
            for c in cells:
                status_style = {
                    "live": "green", "probation": "yellow", "exempt": "blue",
                }.get(c["status"], "red")
                mode_str = ("LIVE" if not c["force_paper"] else "paper") + (
                    f" x{c['multiplier']}" if c["multiplier"] != 1.0 else "")
                table.add_row(
                    c["strategy"], c["category"],
                    f"{c['live_n']} / ${c['live_pnl']:+.2f}",
                    f"{c['paper_n']} / ${c['paper_pnl']:+.2f}",
                    f"[{status_style}]{c['status']}[/]",
                    mode_str,
                )
            console.print(table)
            if cfg.mode == "observe":
                console.print("[dim]observe mode: nothing is enforced yet — flip "
                              "graduation.mode to \"enforce\" to act on this.[/]")
        finally:
            await db.close()

    asyncio.run(_run())

@main.command("entailment")
def entailment():
    """Entailment-arb status: cached LLM verdicts + ladder families visible
    in the markets table (detection view; the live scan runs in the bot)."""

    async def _run():
        from auramaur.exchange.models import Market
        from auramaur.strategy.entailment_arb import ladder_pairs

        db = Database()
        await db.connect(ensure_schema=False)
        try:
            rows = await db.fetchall(
                "SELECT * FROM entailment_verdicts ORDER BY checked_at DESC LIMIT 25")
            if rows:
                t = Table(title="LLM entailment verdicts (cached)")
                t.add_column("pair")
                t.add_column("direction")
                t.add_column("conf", justify="right")
                t.add_column("traded")
                for r in rows:
                    t.add_row(f"{r['market_id_a'][:18]} / {r['market_id_b'][:18]}",
                              r["direction"], f"{r['confidence']:.2f}",
                              "yes" if r["traded_at"] else "")
                console.print(t)
            else:
                console.print("[dim]No LLM verdicts cached yet.[/]")

            mrows = await db.fetchall(
                """SELECT id, question, outcome_yes_price, liquidity
                   FROM markets WHERE active = 1 AND question IS NOT NULL""")
            markets = [Market(id=r["id"], exchange="polymarket",
                              question=r["question"] or "",
                              outcome_yes_price=r["outcome_yes_price"] or 0.5,
                              liquidity=r["liquidity"] or 0.0)
                       for r in (mrows or [])]
            pairs = ladder_pairs(markets)
            viol = [(im, ip, why, im.outcome_yes_price - ip.outcome_yes_price)
                    for im, ip, why in pairs
                    if im.outcome_yes_price - ip.outcome_yes_price >= 0.04]
            console.print(f"\nladder families: [cyan]{len(pairs)}[/] pairs from "
                          f"{len(markets)} active markets; "
                          f"[cyan]{len(viol)}[/] showing >=4c violations "
                          "(before dead-book/liquidity guards)")
            for im, ip, why, gap in sorted(viol, key=lambda v: -v[3])[:10]:
                console.print(f"  [red]{gap:+.2f}[/] {why}")
                console.print(f"        implier: {im.question[:64]} @ {im.outcome_yes_price:.2f}")
                console.print(f"        implied: {ip.question[:64]} @ {ip.outcome_yes_price:.2f}")
        finally:
            await db.close()

    asyncio.run(_run())

@main.command("oddlot")
def oddlot():
    """Odd-lot tender opportunities detected from EDGAR (the equity pillar)."""

    async def _run():
        db = Database()
        await db.connect(ensure_schema=False)
        try:
            rows = await db.fetchall(
                "SELECT * FROM oddlot_filings ORDER BY filed_at DESC LIMIT 30")
            if not rows:
                console.print("[dim]No filings audited yet — the scanner runs "
                              "every 6h inside the bot.[/]")
                return
            t = Table(title="Odd-lot tender filings (EDGAR)")
            t.add_column("filed")
            t.add_column("ticker", style="cyan")
            t.add_column("company")
            t.add_column("priority")
            t.add_column("price", justify="right")
            t.add_column("expires")
            t.add_column("status")
            for r in rows:
                pr = "[green]YES[/]" if r["odd_lot_priority"] else "[dim]no[/]"
                price = (f"${r['tender_price']:.2f}"
                         + (f"-{r['tender_price_high']:.2f}"
                            if r["tender_price_high"] > r["tender_price"] else ""))
                t.add_row(r["filed_at"], r["ticker"] or "?",
                          (r["company"] or "")[:36], pr, price,
                          r["expiration"] or "—", r["status"])
            console.print(t)
            console.print("[dim]Tendering is MANUAL — submit in TWS before "
                          "expiration. Entries are paper-forced.[/]")
        finally:
            await db.close()

    asyncio.run(_run())

@main.command()
@click.option("--days", default=30, help="Number of days to backtest")
@click.option("--min-edge", default=None, type=float, help="Minimum edge % to trade (overrides config)")
@click.option("--kelly-fraction", default=None, type=float, help="Kelly fraction (overrides config)")
@click.option("--compare", is_flag=True, help="Compare two strategies (default params vs aggressive)")
def backtest(days: int, min_edge: float | None, kelly_fraction: float | None, compare: bool):
    """Run backtest on historical signals."""
    from auramaur.backtest.engine import BacktestEngine

    async def _backtest():
        settings = Settings()
        db = Database()
        await db.connect(ensure_schema=False)

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
