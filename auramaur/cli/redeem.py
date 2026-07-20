"""Auramaur CLI — redeem commands."""

from __future__ import annotations

import asyncio

import click
from rich.table import Table

from auramaur.db.database import Database
from config.settings import Settings

from auramaur.cli._base import console, main

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
        from auramaur.cli import AuramaurBot  # call-time lookup keeps test patch working
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
def kill():
    """Activate the kill switch."""
    from auramaur.killswitch import KILL_SWITCH_PATH
    # Arm at the canonical repo-root path so every check finds it regardless of
    # the launch directory.
    KILL_SWITCH_PATH.touch()
    console.print("[bold red]KILL SWITCH ACTIVATED[/]")

@main.command()
def unkill():
    """Deactivate the kill switch."""
    from pathlib import Path
    from auramaur.killswitch import KILL_SWITCH_PATH
    # Remove both the canonical and any CWD-relative switch so disarm is total.
    removed = False
    for ks in (KILL_SWITCH_PATH, Path("KILL_SWITCH")):
        if ks.exists():
            ks.unlink()
            removed = True
    if removed:
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
        await db.connect(ensure_schema=False)

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
