"""Auramaur CLI — diagnostics commands."""

from __future__ import annotations

import asyncio

import click
from rich.table import Table

from auramaur.db.database import Database
from config.settings import Settings

from auramaur.cli._base import console, main

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

@main.command()
def attribution():
    """Where the bot makes/loses money — per-category and per-strategy P&L."""
    from auramaur.monitoring.attribution import PerformanceAttributor
    from auramaur.monitoring.diagnostics import render_attribution

    async def _run():
        settings = Settings()
        db = Database()
        await db.connect()
        try:
            attr = PerformanceAttributor(db=db)
            venues = await attr.get_venue_summary(is_live=settings.is_live)
            cats = await attr.get_category_summary(is_live=settings.is_live)
            strats = await attr.get_strategy_summary(is_live=settings.is_live)
            mode = "live" if settings.is_live else "paper"
            console.print(render_attribution(cats, strats, mode=mode, venue_rows=venues))
        finally:
            await db.close()

    asyncio.run(_run())

@main.command()
def doctor():
    """Operational health snapshot — is anything broken right now?"""
    from auramaur.monitoring.diagnostics import gather_doctor, render_doctor

    async def _run():
        settings = Settings()
        db = Database()
        await db.connect()
        try:
            state = await gather_doctor(settings, db)
            console.print(render_doctor(state))
        finally:
            await db.close()

    asyncio.run(_run())

@main.command()
@click.option("--mb", "max_mb", default=8.0, type=float,
              help="How many MB from the end of the log to scan.")
@click.option("--top", default=15, help="Number of distinct events to show.")
@click.option("--all", "include_benign", is_flag=True,
              help="Include benign visibility-warnings (order.live, kraken.order, …).")
def errors(max_mb: float, top: int, include_benign: bool):
    """Digest of recent errors/warnings from the JSON log — top events by count."""
    from auramaur.monitoring.diagnostics import gather_log_errors, render_error_digest
    settings = Settings()
    s = gather_log_errors(settings.logging.file, max_bytes=int(max_mb * 1_000_000),
                          top=top, include_benign=include_benign)
    console.print(render_error_digest(s))

@main.command("health")
def health():
    """Operational live-readiness preflight — is it safe to arm live now?

    Runs the live_gate checks (kill switch, DB, fee model, drawdown, mark
    integrity, LLM budget, arming gates) and prints a verdict. Exit code 0 if
    live is allowed, 1 if any BLOCK condition is present.
    """
    import sys

    async def _run() -> int:
        from auramaur.monitoring.live_gate import preflight

        settings = Settings()
        db = Database()
        await db.connect()
        try:
            report = await preflight(settings, db)
        finally:
            await db.close()

        colour = {"OK": "green", "WARN": "yellow", "BLOCK": "red"}
        for r in report.results:
            console.print(f"  [{colour[r.severity]}]{r.severity:<5}[/] {r.name}: {r.detail}")
        if report.live_allowed:
            warns = len(report.warnings)
            console.print("\n[bold green]LIVE ALLOWED[/]"
                          + (f" ([yellow]{warns} warning(s)[/])" if warns else ""))
            return 0
        console.print("\n[bold red]LIVE BLOCKED[/] — "
                      + ", ".join(b.name for b in report.blocks))
        return 1

    sys.exit(asyncio.run(_run()))


@main.command("ibkr-etf-preflight")
def ibkr_etf_preflight():
    """Verify IBKR ETF paper isolation, data, OpenAI access, and DB state."""
    import sys

    async def _run() -> int:
        from auramaur.monitoring.ibkr_etf_preflight import preflight

        settings = Settings()
        db = Database()
        await db.connect()
        try:
            report = await preflight(settings, db)
        finally:
            await db.close()
        colour = {"OK": "green", "WARN": "yellow", "BLOCK": "red"}
        for result in report.results:
            console.print(
                f"  [{colour[result.severity]}]{result.severity:<5}[/] "
                f"{result.name}: {result.detail}")
        console.print("\n[bold green]ETF PAPER READY[/]" if report.ready else
                      "\n[bold red]ETF PAPER BLOCKED[/]")
        return 0 if report.ready else 1

    sys.exit(asyncio.run(_run()))


@main.command("ibkr-multiasset-preflight")
@click.option("--book", "book_names", multiple=True,
              type=click.Choice(("global_etf", "fx", "futures",
                                 "international_equity", "options", "bonds")))
def ibkr_multiasset_preflight(book_names):
    """Verify contract resolution, data, isolation, and schema for six books."""
    import sys

    async def _run() -> int:
        from auramaur.monitoring.ibkr_multiasset_preflight import preflight
        from auramaur.exchange.ibkr_instruments import IBKRBook

        settings = Settings()
        db = Database()
        await db.connect()
        try:
            books = tuple(IBKRBook(name) for name in book_names) or None
            report = await preflight(settings, db, books=books)
        finally:
            await db.close()
        colour = {"OK": "green", "WARN": "yellow", "BLOCK": "red"}
        for result in report.results:
            console.print(
                f"  [{colour[result.severity]}]{result.severity:<5}[/] "
                f"{result.book}: {result.detail}")
        console.print("\n[bold green]MULTI-ASSET PAPER READY[/]" if report.ready else
                      "\n[bold red]MULTI-ASSET PAPER BLOCKED[/]")
        return 0 if report.ready else 1

    sys.exit(asyncio.run(_run()))


@main.command("ibkr-contract-approve")
@click.argument("instrument_key")
@click.option("--reason", required=True,
              help="Operator rationale recorded with this contract identity.")
def ibkr_contract_approve(instrument_key, reason):
    """Approve a pending discovered identity (currently corporate bonds)."""
    import asyncio
    import sys
    from auramaur.db.database import Database
    from auramaur.exchange.ibkr_registry import approve

    async def _run() -> int:
        db = Database()
        await db.connect()
        try:
            changed = await approve(db, instrument_key, reason)
        finally:
            await db.close()
        if not changed:
            console.print("[bold red]NOT APPROVED[/] — no pending current identity")
            return 1
        console.print(f"[bold green]APPROVED[/] {instrument_key}")
        return 0

    sys.exit(asyncio.run(_run()))

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
