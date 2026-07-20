"""Auramaur CLI — maintenance commands."""

from __future__ import annotations

import asyncio

import click
from rich.table import Table

from auramaur.db.database import Database
from config.settings import Settings

from auramaur.cli._base import console, main


@main.command("data-audit")
def data_audit():
    """Check point-in-time data contracts without mutating data."""
    async def _run():
        from auramaur.data_quality import audit_data_contracts

        db = Database()
        await db.connect()
        try:
            violations = await audit_data_contracts(db)
            if not violations:
                console.print("[green]All data contracts pass.[/]")
                return
            table = Table(title="Data contract violations")
            table.add_column("Contract")
            table.add_column("Count", justify="right")
            table.add_column("Detail")
            for violation in violations:
                table.add_row(violation.contract, str(violation.count), violation.detail)
            console.print(table)
            raise click.ClickException(f"{len(violations)} data contracts failed")
        finally:
            await db.close()

    asyncio.run(_run())


@main.command("information-graduation")
def information_graduation_report():
    """Report source/category/horizon information graduation cells."""
    async def _run():
        from auramaur.information_graduation import InformationGraduation

        db = Database()
        await db.connect()
        try:
            rows = await InformationGraduation(db).report()
            table = Table(title="Information graduation")
            for name in ("Source", "Category", "Horizon", "Status", "Influence", "Reason"):
                table.add_column(name)
            for row in rows:
                table.add_row(
                    row["source"], row["category"], row["horizon"], row["status"],
                    f"{row['influence_multiplier']:.2f}", row["reason"],
                )
            console.print(table)
        finally:
            await db.close()

    asyncio.run(_run())

@main.command("repair-categories")
@click.option("--write", is_flag=True,
              help="Persist classified categories (default is a dry-run).")
@click.option("--all", "reclassify_all", is_flag=True,
              help="Re-classify EVERY market and overwrite stored labels "
                   "that differ (default: only empty/NULL labels).")
def repair_categories(write: bool, reclassify_all: bool):
    """Classify markets stored with an empty/NULL category.

    Empty categories bypass blocked_categories (the check is `category in
    blocked`) and pollute graduation cells as '(none)'. Insert sites now
    classify on write; this backfills history (markets + portfolio rows).

    With --all, every stored label is re-derived from the current classifier
    and overwritten when it differs. This is the recovery path for the
    2026-06 keyword mislabels (tennis matches stored as politics_us via
    "primary" in resolution boilerplate, "American League" hitting the old
    US-politics "american" marker, governor primaries stored as sports via
    the long-removed "winner" marker).
    """

    async def _run():
        from collections import Counter

        from auramaur.strategy.classifier import classify_market

        db = Database()
        await db.connect()
        try:
            scope_sql = ("" if reclassify_all
                         else " WHERE category = '' OR category IS NULL")
            rows = await db.fetchall(
                "SELECT id, question, description, category FROM markets"
                + scope_sql)
            counts: Counter = Counter()
            transitions: Counter = Counter()
            changed = 0
            for r in rows or []:
                cat = classify_market(r["question"] or "", r["description"] or "")
                counts[cat] += 1
                old = r["category"] or ""
                if cat == old:
                    continue
                changed += 1
                transitions[f"{old or '(none)'} -> {cat}"] += 1
                if write:
                    await db.execute(
                        "UPDATE markets SET category = ? WHERE id = ?",
                        (cat, r["id"]))
                    await db.execute(
                        "UPDATE portfolio SET category = ? WHERE market_id = ?",
                        (cat, r["id"]))
            if write:
                await db.commit()
            mode = "WRITE" if write else "DRY RUN"
            scope = "all" if reclassify_all else "uncategorized"
            console.print(f"[bold]{mode}[/] {len(rows or [])} {scope} markets, "
                          f"{changed} relabeled")
            for cat, n in counts.most_common():
                console.print(f"  {cat:16} {n}")
            if transitions:
                console.print("\n[bold]Transitions[/] (old -> new):")
                for tr, n in transitions.most_common(25):
                    console.print(f"  {tr:42} {n}")
            if not write and changed:
                console.print("\n[dim]Re-run with [cyan]--write[/] to persist.[/]")
        finally:
            await db.close()

    asyncio.run(_run())

@main.command("settle-stale")
@click.option("--write", is_flag=True,
              help="Perform the settlements (default is a dry-run).")
def settle_stale(write: bool):
    """Settle live positions the venue says resolved but the tracker missed.

    The resolution tracker can't settle markets the Gamma API no longer
    returns (archived) or whose active flag is stale — those positions sit
    at phantom marks (-100% on BOTH sides of the same market). This sweeps
    them against Polymarket's data-api per-token resolution state, through
    the same idempotent settlement path the tracker uses.
    """

    async def _run():
        from auramaur.strategy.resolution_tracker import ResolutionTracker

        settings = Settings()
        proxy = settings.polymarket_proxy_address
        if not proxy:
            console.print("[red]POLYMARKET_PROXY_ADDRESS not set.[/]")
            return

        db = Database()
        await db.connect()
        try:
            tracker = ResolutionTracker(db=db, calibration=None,
                                        discoveries={}, proxy_address=proxy)
            settlements = await tracker.settle_via_venue(
                proxy, dry_run=not write)
            if not settlements:
                console.print("[green]No venue-resolved positions to settle.[/]")
                return
            t = Table(title=f"{'Settled' if write else 'Would settle (dry run)'} "
                            f"{len(settlements)} positions")
            t.add_column("Market", max_width=50)
            t.add_column("Token", width=5)
            t.add_column("Size", justify="right")
            t.add_column("Avg", justify="right")
            t.add_column("P&L", justify="right")
            t.add_column("Status")
            total = 0.0
            for s in settlements:
                total += s["pnl"]
                color = "green" if s["pnl"] >= 0 else "red"
                t.add_row(s["title"][:50], s["token"], f"{s['size']:.1f}",
                          f"{s['avg_price']:.2f}",
                          f"[{color}]{s['pnl']:+.2f}[/]", s["status"])
            console.print(t)
            color = "green" if total >= 0 else "red"
            console.print(f"[bold]Net realized: [{color}]${total:+.2f}[/][/]")
            if not write:
                console.print("\n[dim]Re-run with [cyan]--write[/] to settle.[/]")
        finally:
            await db.close()

    asyncio.run(_run())

@main.command("kalshi-settlements")
@click.option("--write", is_flag=True,
              help="Book the settlements into the ledger (default dry-run).")
def kalshi_settlements(write: bool):
    """Backfill/book Kalshi settlements from the venue's settlement feed.

    Kalshi realized P&L was never recorded (the tracker keyed off a Market
    field that didn't exist, and the syncer dropped settled positions before
    booking). This walks GET /portfolio/settlements and books every
    settlement missing from pnl_ledger, idempotently (source_ref dedupes).
    Settlements whose cost basis we never recorded are listed but skipped.
    The bot also runs this sweep automatically every resolution cycle.
    """

    async def _run():
        from auramaur.broker.kalshi_settlements import sweep_kalshi_settlements
        from auramaur.exchange.kalshi import KalshiClient
        from auramaur.exchange.paper import PaperTrader

        settings = Settings()
        db = Database()
        await db.connect()
        try:
            paper = PaperTrader(db=db)
            client = KalshiClient(settings=settings, paper_trader=paper)
            rows = await sweep_kalshi_settlements(db, client, dry_run=not write)
            if not rows:
                console.print("[green]No unbooked Kalshi settlements found.[/]")
                return
            t = Table(title=f"{'Booked' if write else 'Would book (dry run)'} "
                            f"Kalshi settlements")
            t.add_column("Settled", width=10)
            t.add_column("Ticker", max_width=30)
            t.add_column("Result", width=6)
            t.add_column("Qty", justify="right")
            t.add_column("Payout", justify="right")
            t.add_column("P&L", justify="right")
            total, skipped = 0.0, 0
            for r in sorted(rows, key=lambda x: x["settled"]):
                if r["pnl"] is None:
                    skipped += 1
                    t.add_row(r["settled"][:10], r["ticker"][:30], r["result"],
                              f"{r['qty']:.0f}", "—",
                              f"[yellow]skip: {r['reason']}[/]")
                    continue
                total += r["pnl"]
                color = "green" if r["pnl"] >= 0 else "red"
                t.add_row(r["settled"][:10], r["ticker"][:30], r["result"],
                          f"{r['qty']:.0f}", f"${r['payout']:.2f}",
                          f"[{color}]{r['pnl']:+.2f}[/]")
            console.print(t)
            color = "green" if total >= 0 else "red"
            console.print(f"[bold]Net realized: [{color}]${total:+.2f}[/][/]"
                          + (f"  ({skipped} skipped, no cost basis)" if skipped else ""))
            if not write:
                console.print("\n[dim]Re-run with [cyan]--write[/] to book.[/]")
        finally:
            await db.close()

    asyncio.run(_run())

@main.command("repair-pnl")
@click.option("--write", is_flag=True,
              help="Persist resolution_pnl rows and rebuild category_stats dollar fields.")
def repair_pnl(write: bool):
    """Backfill realized PnL from fills for resolved markets.

    Default mode is a dry-run. Use ``--write`` after reviewing the summary.
    """

    async def _run():
        from auramaur.broker.pnl_repair import (
            backfill_resolution_pnl,
            rebuild_category_stats_from_resolution_pnl,
        )

        db = Database()
        await db.connect()
        try:
            dry_run = not write
            backfill = await backfill_resolution_pnl(db, is_paper=0, dry_run=dry_run)
            category = await rebuild_category_stats_from_resolution_pnl(db, dry_run=dry_run)

            mode = "WRITE" if write else "DRY RUN"
            console.print(f"[bold]{mode}[/] PnL repair")
            console.print(
                f"Resolved markets scanned: [cyan]{backfill.scanned_markets}[/]  "
                f"with live fills: [cyan]{backfill.markets_with_fills}[/]"
            )
            console.print(
                f"resolution_pnl rows {'written' if write else 'would write'}: "
                f"[cyan]{backfill.written_resolution_rows if write else backfill.markets_with_fills}[/]  "
                f"total PnL: [{'green' if backfill.total_pnl >= 0 else 'red'}]"
                f"${backfill.total_pnl:+.2f}[/]"
            )

            category_pnl = category.by_category if write else backfill.by_category
            if category_pnl:
                table = Table(title="Resolution PnL by Category")
                table.add_column("Category", style="cyan")
                table.add_column("PnL", justify="right")
                for cat, pnl in sorted(
                    category_pnl.items(), key=lambda item: item[1], reverse=True,
                ):
                    style = "green" if pnl >= 0 else "red"
                    table.add_row(cat or "other", f"[{style}]${pnl:+.2f}[/]")
                console.print(table)

            if not write:
                console.print("\n[dim]Preview only. Re-run with [cyan]--write[/] to persist.[/]")
        finally:
            await db.close()

    asyncio.run(_run())

@main.command("reconcile-kalshi-orders")
@click.option("--write", is_flag=True,
              help="Persist the reconciled trade statuses (default is a dry-run).")
def reconcile_kalshi_orders(write: bool):
    """Reconcile stale 'pending' Kalshi trade rows against the venue.

    Kalshi orders inserted at placement stayed 'pending' because the client
    historically wasn't monitored. This re-queries each one and flips it to its
    real terminal status (filled/cancelled/expired). Ledger hygiene only — it
    does not touch P&L. Default mode is a dry-run.
    """

    async def _run():
        from auramaur.broker.order_reconcile import reconcile_pending_kalshi_orders
        from auramaur.exchange.kalshi import KalshiClient
        from auramaur.exchange.paper import PaperTrader

        settings = Settings()
        db = Database()
        await db.connect()
        try:
            paper = PaperTrader(db=db)
            exchange = KalshiClient(settings=settings, paper_trader=paper)
            res = await reconcile_pending_kalshi_orders(
                db, exchange, dry_run=not write,
            )
            mode = "WRITE" if write else "DRY RUN"
            console.print(f"[bold]{mode}[/] Kalshi order reconcile")
            console.print(
                f"Pending rows scanned: [cyan]{res.scanned}[/]  "
                f"{'updated' if write else 'would update'}: [cyan]{res.updated}[/]  "
                f"still pending: [cyan]{res.still_pending}[/]  "
                f"errors (left untouched): [yellow]{res.errors}[/]"
            )
            if res.by_status:
                table = Table(title="Reconciled by status")
                table.add_column("status", style="cyan")
                table.add_column("count", justify="right")
                for status, n in sorted(res.by_status.items()):
                    table.add_row(status, str(n))
                console.print(table)
            if not write:
                console.print("\n[dim]Preview only. Re-run with [cyan]--write[/] to persist.[/]")
        finally:
            await db.close()

    asyncio.run(_run())
