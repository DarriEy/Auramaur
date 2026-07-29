"""Auramaur CLI — diagnostics commands."""

from __future__ import annotations

import asyncio

import click
from rich.panel import Panel
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
        await db.connect(ensure_schema=False)
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
        await db.connect(ensure_schema=False)
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
        await db.connect(ensure_schema=False)
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
        await db.connect(ensure_schema=False)
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
        await db.connect(ensure_schema=False)
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
        await db.connect(ensure_schema=False)
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
@click.option("--log-file", default=None,
              help="Path to structlog output file (default: the configured "
                   "logging.file / LOGGING__FILE setting)")
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
        await db.connect(ensure_schema=False)
        try:
            return await evaluate_readiness(
                db,
                log_file=Path(log_file) if log_file else None,
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


@main.command("ibkr-backtest")
@click.option("--book", default="global_etf",
              type=click.Choice(("global_etf", "international_equity")))
@click.option("--years", default=5, show_default=True,
              help="Years of daily history to replay.")
@click.option("--spread-bps", default=None, type=float,
              help="Assumed round-trip spread. Defaults to the book's "
                   "max_spread_bps, i.e. the worst the live book would accept.")
def ibkr_backtest(book: str, years: int, spread_bps):
    """Replay the deployed momentum rules over history for an early read.

    The live gate needs 180 elapsed days of FORWARD paper record, so it cannot
    say whether this strategy has any edge until 2027. The rules are
    deterministic, so history can answer that today.

    This writes NOTHING. ibkr_paper_round_trips and ibkr_paper_daily_marks are
    the pre-registered forward evidence the live gate reads; filling them with
    replayed history would manufacture the very evidence the contract demands.
    """
    async def _run():
        from auramaur.backtest.ibkr_replay import replay_momentum_book
        from auramaur.exchange.alpaca_multiasset import build_multiasset_market_data
        from auramaur.exchange.ibkr_instruments import BY_BOOK, IBKRBook
        from auramaur.risk.ibkr_evidence import evaluate_ibkr_evidence

        settings = Settings()
        cfg = settings.ibkr.multiasset_books[book]
        spread = cfg.max_spread_bps if spread_bps is None else spread_bps
        specs = BY_BOOK[IBKRBook(book)]
        client = build_multiasset_market_data(
            settings, client_id=settings.ibkr.multiasset_preflight_client_id)
        bars_by_key, classes = {}, {}
        try:
            for spec in specs:
                try:
                    bars = await client.get_daily_bars(spec, duration=f"{years} Y")
                except Exception as exc:  # noqa: BLE001
                    console.print(f"  [yellow]{spec.key}: {str(exc)[:70]}[/]")
                    continue
                rows = [(str(d)[:10], float(c)) for d, c in (bars or []) if c and c > 0]
                if len(rows) < 150:
                    console.print(f"  [dim]{spec.key}: only {len(rows)} bars, skipped[/]")
                    continue
                bars_by_key[spec.key] = sorted(rows)
                classes[spec.key] = spec.asset_class or "unknown"
        finally:
            await client.close()
        if not bars_by_key:
            console.print("[red]No usable history fetched.[/]")
            return

        result = replay_momentum_book(
            bars_by_key, classes,
            budget_usd=cfg.budget_usd, max_positions=cfg.max_positions,
            max_position_pct=cfg.max_position_pct,
            max_deployment_pct=cfg.max_deployment_pct,
            risk_per_position_pct=cfg.risk_per_position_pct,
            max_asset_class_risk_pct=cfg.max_asset_class_risk_pct,
            stop_vol_multiple=cfg.stop_vol_multiple,
            min_stop_pct=cfg.min_stop_pct, slippage_bps=cfg.slippage_bps,
            stop_loss_pct=cfg.stop_loss_pct, take_profit_pct=cfg.take_profit_pct,
            min_norm_momentum=settings.ibkr.multiasset_min_normalized_momentum,
            exit_norm_momentum=settings.ibkr.multiasset_exit_normalized_momentum,
            assumed_spread_bps=spread)

        net = result.net_pnls
        wins = [p for p in net if p > 0]
        console.print(Panel(
            f"[bold]{book}[/] — {len(bars_by_key)} instruments, "
            f"{result.sessions} sessions {result.first_date} to "
            f"{result.last_date}\nassumed spread {spread:.0f}bps + "
            f"{cfg.slippage_bps:.0f}bps slippage per leg, commission charged "
            f"both ways", expand=False))
        if not net:
            console.print("[yellow]No round trips. The entry gate never "
                          "qualified on this history.[/]")
            return
        table = Table(title="replayed round trips", expand=False)
        for c in ("trips", "gross P&L", "commission", "net P&L", "mean",
                  "win rate"):
            table.add_column(c, justify="right")
        gross = sum(t.gross_usd for t in result.trips)
        fees = sum(t.commission_usd for t in result.trips)
        table.add_row(str(len(net)), f"${gross:,.2f}", f"${-fees:,.2f}",
                      f"${sum(net):,.2f}", f"${sum(net) / len(net):,.2f}",
                      f"{len(wins) / len(net):.1%}")
        console.print(table)

        # Same contract the live gate applies, on replayed instead of forward
        # evidence. Passing here is NOT graduation; it is a reason to keep the
        # forward clock running rather than abandon the book.
        from datetime import date
        span = (date.fromisoformat(result.last_date)
                - date.fromisoformat(result.first_date)).days
        verdict = evaluate_ibkr_evidence(
            net, elapsed_days=span, budget_usd=cfg.budget_usd,
            min_observations=30)
        state = "[green]would clear[/]" if verdict.ready else "[yellow]would NOT clear[/]"
        console.print(f"  Against the live contract on this history: {state}")
        for reason in getattr(verdict, "reasons", ()) or ():
            console.print(f"    [dim]- {reason}[/]")
        console.print("\n[dim]Backtest evidence only. Nothing was written: the "
                      "gate's tables hold forward paper record, and replayed "
                      "history must never be mistaken for it.[/]")
    asyncio.run(_run())


@main.command("ibkr-exit-study")
@click.option("--book", default="global_etf",
              type=click.Choice(("global_etf", "international_equity")))
@click.option("--years", default=5, show_default=True)
@click.option("--split", default="2025-01-01", show_default=True,
              help="Trips entered before this date tune; on/after only score.")
@click.option("--spread-bps", default=3.0, show_default=True)
def ibkr_exit_study(book: str, years: int, split: str, spread_bps: float):
    """Does the exit geometry explain the loss? Tuned on train, scored on test.

    The deployed book wins 27% of round trips with winners and losers the same
    size — a trend follower needs winners ~2.7x losers at that hit rate, which
    points at the 5%-stop / 10%-target geometry cutting winners short.

    Configurations are ranked ONLY on trips entered before --split, and the
    winner is then scored ONCE on trips entered after it. Searching N
    configurations and reporting the best inflates apparent edge, so the
    number searched is printed, and the winner is compared against the MEDIAN
    test result: a pick that is no better than the median did not generalise,
    it was luck.
    """
    async def _run():
        from statistics import median

        from auramaur.backtest.ibkr_replay import (
            mean_lcb, replay_momentum_book, split_by_entry,
        )
        from auramaur.exchange.alpaca_multiasset import build_multiasset_market_data
        from auramaur.exchange.ibkr_instruments import BY_BOOK, IBKRBook

        settings = Settings()
        cfg = settings.ibkr.multiasset_books[book]
        client = build_multiasset_market_data(
            settings, client_id=settings.ibkr.multiasset_preflight_client_id)
        bars_by_key, classes = {}, {}
        try:
            for spec in BY_BOOK[IBKRBook(book)]:
                try:
                    bars = await client.get_daily_bars(spec, duration=f"{years} Y")
                except Exception:  # noqa: BLE001
                    continue
                rows = [(str(d)[:10], float(c)) for d, c in (bars or []) if c and c > 0]
                if len(rows) >= 150:
                    bars_by_key[spec.key] = sorted(rows)
                    classes[spec.key] = spec.asset_class or "unknown"
        finally:
            await client.close()
        if not bars_by_key:
            console.print("[red]No usable history.[/]")
            return

        base = dict(
            budget_usd=cfg.budget_usd, max_positions=cfg.max_positions,
            max_position_pct=cfg.max_position_pct,
            max_deployment_pct=cfg.max_deployment_pct,
            risk_per_position_pct=cfg.risk_per_position_pct,
            max_asset_class_risk_pct=cfg.max_asset_class_risk_pct,
            stop_vol_multiple=cfg.stop_vol_multiple,
            min_stop_pct=cfg.min_stop_pct, slippage_bps=cfg.slippage_bps,
            min_norm_momentum=settings.ibkr.multiasset_min_normalized_momentum,
            assumed_spread_bps=spread_bps)

        NO_TARGET = 1e9
        grid = [(sl, tp, ex)
                for sl in (3.0, 5.0, 8.0, 12.0)
                for tp in (10.0, 20.0, 40.0, NO_TARGET)
                for ex in (-0.1, -0.5, 0.0)]
        deployed = (cfg.stop_loss_pct, cfg.take_profit_pct,
                    settings.ibkr.multiasset_exit_normalized_momentum)

        rows = []
        for sl, tp, ex in grid:
            result = replay_momentum_book(
                bars_by_key, classes, stop_loss_pct=sl, take_profit_pct=tp,
                exit_norm_momentum=ex, **base)
            train, test = split_by_entry(result.trips, split)
            rows.append({
                "cfg": (sl, tp, ex),
                "train_n": len(train), "test_n": len(test),
                "train_lcb": mean_lcb([t.net_usd for t in train]),
                "train_net": sum(t.net_usd for t in train),
                "test_net": sum(t.net_usd for t in test),
                "test_mean": (sum(t.net_usd for t in test) / len(test)
                              if test else 0.0),
                "test_lcb": mean_lcb([t.net_usd for t in test]),
                "test_win": (sum(1 for t in test if t.net_usd > 0) / len(test)
                             if test else 0.0),
            })

        def label(cfg_tuple):
            sl, tp, ex = cfg_tuple
            return f"stop {sl:g}% / target {'none' if tp >= NO_TARGET else f'{tp:g}%'} / mom-exit {ex:g}"

        eligible = [r for r in rows if r["train_n"] >= 30]
        console.print(Panel(
            f"[bold]{book}[/] — {len(bars_by_key)} instruments, "
            f"{len(grid)} configurations searched\n"
            f"tune on entries before {split}; score on entries after\n"
            f"spread {spread_bps:g}bps + {cfg.slippage_bps:g}bps slippage per "
            f"leg, commission both ways", expand=False))
        if not eligible:
            console.print("[yellow]No configuration produced 30+ training "
                          "trips; nothing is selectable.[/]")
            return

        base_row = next((r for r in rows if r["cfg"] == deployed), None)
        winner = max(eligible, key=lambda r: r["train_lcb"])
        med_test = median([r["test_net"] for r in eligible])

        table = Table(title="chosen on train, scored on test", expand=False)
        for c in ("configuration", "train n", "train LCB", "test n",
                  "test net", "test mean", "test win"):
            table.add_column(c, justify="right")
        for tag, r in (("deployed", base_row), ("best-on-train", winner)):
            if r is None:
                continue
            table.add_row(f"{tag}: {label(r['cfg'])}", str(r["train_n"]),
                          f"${r['train_lcb']:.2f}", str(r["test_n"]),
                          f"${r['test_net']:,.2f}", f"${r['test_mean']:.2f}",
                          f"{r['test_win']:.0%}")
        console.print(table)
        console.print(
            f"  median test net across the {len(eligible)} eligible "
            f"configurations: [bold]${med_test:,.2f}[/]")
        if winner["test_net"] <= med_test:
            console.print("  [yellow]The train-selected configuration is no "
                          "better than the median out of sample — the ranking "
                          "did not generalise. Treat it as noise.[/]")
        elif winner["test_lcb"] <= 0:
            console.print("  [yellow]Beats the median but its own 95% lower "
                          "bound is still not positive: suggestive, not "
                          "evidence.[/]")
        else:
            console.print("  [green]Beats the median AND clears a positive "
                          "lower bound out of sample.[/]")
        console.print("\n[dim]Backtest evidence only; nothing written. One "
                      "split is one experiment — re-running with a different "
                      "--split until it passes is how this lies to you.[/]")
    asyncio.run(_run())


@main.command("ibkr-signal-study")
@click.option("--book", default="global_etf",
              type=click.Choice(("global_etf", "international_equity")))
@click.option("--years", default=5, show_default=True)
@click.option("--horizon", default=10, show_default=True,
              help="Forward sessions to measure the signal against.")
def ibkr_signal_study(book: str, years: int, horizon: int):
    """Does normalized_momentum predict anything on this universe?

    Measures information content, not a tuned rule: no search, no best-of-N.
    Forward returns are EXCESS of the equal-weight universe on the same
    session, so a rising market cannot masquerade as skill, and the t-statistic
    samples only non-overlapping windows.
    """
    async def _run():
        from auramaur.backtest.ibkr_signal_study import (
            decile_buckets, gate_contrast, information_coefficient, observations,
        )
        from auramaur.exchange.alpaca_multiasset import build_multiasset_market_data
        from auramaur.exchange.ibkr_instruments import BY_BOOK, IBKRBook

        settings = Settings()
        threshold = settings.ibkr.multiasset_min_normalized_momentum
        client = build_multiasset_market_data(
            settings, client_id=settings.ibkr.multiasset_preflight_client_id)
        bars_by_key = {}
        try:
            for spec in BY_BOOK[IBKRBook(book)]:
                try:
                    bars = await client.get_daily_bars(spec, duration=f"{years} Y")
                except Exception:  # noqa: BLE001
                    continue
                rows = [(str(d)[:10], float(c)) for d, c in (bars or []) if c and c > 0]
                if len(rows) >= 150:
                    bars_by_key[spec.key] = sorted(rows)
        finally:
            await client.close()
        if not bars_by_key:
            console.print("[red]No usable history.[/]")
            return

        obs = observations(bars_by_key, horizon=horizon)
        if not obs:
            console.print("[yellow]No observations.[/]")
            return
        mean_ic, t_stat, used, total = information_coefficient(obs, horizon=horizon)
        console.print(Panel(
            f"[bold]{book}[/] — {len(bars_by_key)} instruments, "
            f"{len(obs):,} signal/forward pairs, {horizon}-session horizon\n"
            f"returns are EXCESS of the equal-weight universe that session",
            expand=False))

        table = Table(title="mean excess forward return by momentum quintile",
                      expand=False)
        for c in ("bucket", "n", "mean momentum", "mean excess", "95% low",
                  "hit rate"):
            table.add_column(c, justify="right")
        for b in decile_buckets(obs):
            table.add_row(b.label, f"{b.n:,}", f"{b.mean_momentum:+.3f}",
                          f"{b.mean_excess:+.3%}", f"{b.lcb:+.3%}",
                          f"{b.hit_rate:.0%}")
        console.print(table)

        passing, failing = gate_contrast(obs, threshold)
        gate = Table(title=f"the deployed entry gate (min_norm_momentum "
                           f"{threshold:g})", expand=False)
        for c in ("side", "n", "mean excess", "95% low", "hit rate"):
            gate.add_column(c, justify="right")
        for b in (passing, failing):
            gate.add_row(b.label, f"{b.n:,}", f"{b.mean_excess:+.3%}",
                         f"{b.lcb:+.3%}", f"{b.hit_rate:.0%}")
        console.print(gate)

        console.print(f"  information coefficient: [bold]{mean_ic:+.4f}[/] "
                      f"(t={t_stat:+.2f} on {used} non-overlapping sessions of "
                      f"{total})")
        if passing.lcb > 0:
            console.print("  [green]The gate's selections beat the universe "
                          "with a positive lower bound.[/]")
        elif passing.mean_excess > failing.mean_excess:
            console.print("  [yellow]Selections lean better than rejections, "
                          "but the lower bound does not clear zero — "
                          "suggestive, not evidence.[/]")
        else:
            console.print("  [red]The gate does not separate winners from "
                          "losers: what it buys does no better than what it "
                          "skips.[/]")
        console.print("\n[dim]No selection was performed here, so there is "
                      "nothing to overfit — but this measures the SIGNAL, not "
                      "the tradeable rule. Costs, sizing and exits still "
                      "apply on top.[/]")
    asyncio.run(_run())


@main.command("ibkr-costs")
@click.option("--ingest/--no-ingest", default=True,
              help="Pull this session's fills from IBKR before reporting.")
@click.option("--probe-label", default="",
              help="Tag the ingested fills with which calibration probe they are.")
@click.option("--mid", "mids", multiple=True, metavar="ORDERREF=PRICE",
              help="Mid price at submission, per orderRef. Cannot be recovered "
                   "later, so supply it here or slippage stays unknown.")
def ibkr_costs(ingest, probe_label, mids):
    """Measure what IBKR execution actually costs, from real fills.

    Read-only against the broker: asks what happened, records it, reports it.
    There is no order path here and there must never be one.

    The screen that decides the tradeable universe currently runs on ASSUMED
    commission/spread/slippage. At USD 800 notional the difference between
    4bps and 32bps decides whether anything clears its costs at all, so these
    numbers are worth measuring rather than guessing.
    """
    async def _run():
        from auramaur.broker.execution_costs import ingest_fills, measure

        settings = Settings()
        db = Database()
        await db.connect(ensure_schema=False)
        try:
            if ingest:
                parsed = {}
                for item in mids:
                    ref, _, price = item.partition("=")
                    try:
                        parsed[ref.strip()] = float(price)
                    except ValueError:
                        console.print(f"[red]bad --mid {item!r}, expected REF=PRICE[/]")
                        return
                from ib_async import IB
                cfg = settings.ibkr
                port = cfg.paper_port if cfg.environment == "paper" else cfg.live_port
                ib = IB()
                try:
                    await asyncio.wait_for(ib.connectAsync(
                        host=cfg.host, port=port,
                        clientId=cfg.balance_client_id + 1, readonly=True),
                        timeout=30)
                except Exception as exc:  # noqa: BLE001
                    console.print(f"[red]IBKR connect failed:[/] {str(exc)[:160]}")
                    return
                try:
                    n = await ingest_fills(db, ib, probe_label=probe_label,
                                           mids=parsed)
                    console.print(f"  ingested [cyan]{n}[/] new fill(s)")
                finally:
                    ib.disconnect()

            rows = await measure(db)
            if not rows:
                console.print("[dim]No fills recorded yet — place the "
                              "calibration probes, then re-run.[/]")
                return
            table = Table(title="Measured execution cost by venue class")
            table.add_column("venue class", style="cyan")
            table.add_column("fills", justify="right")
            table.add_column("mean notional", justify="right")
            table.add_column("commission", justify="right")
            table.add_column("comm bps", justify="right")
            table.add_column("slippage", justify="right")
            for r in rows:
                # Dollars AND bps: dollars exposes the fixed minimum that
                # dominates at small size, bps the marginal rate. Only one of
                # them is how a $1 floor gets mistaken for a 0.1% rate.
                slip = (f"{r.slippage_bps:.1f} bps ({r.mids_available})"
                        if r.slippage_bps is not None else "[dim]no mids[/]")
                table.add_row(
                    r.venue_class, str(r.fills), f"${r.mean_notional:,.0f}",
                    f"${r.commission_usd:.2f}", f"{r.commission_bps:.1f}", slip)
            console.print(table)
            console.print("[dim]Feed commission/slippage into "
                          "ibkr_edge_economics.round_trip_cost_bps() and re-run "
                          "the universe screen on measured numbers.[/]")
        finally:
            await db.close()

    asyncio.run(_run())
