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


@main.command("intelligence-eval")
@click.option("--all-streams", is_flag=True,
              help="Compare production, intelligence, and information-trial streams.")
def intelligence_eval_report(all_streams: bool):
    """Show the resolved, market-relative shadow evaluation scorecard."""
    async def _run():
        from auramaur.evaluation.store import EvaluationStore
        from auramaur.web.db import ReadOnlyDatabase

        # Pure report: mode=ro + query_only makes writes impossible at SQLite,
        # and avoids Database.connect()'s write PRAGMAs/DDL on the live file.
        db = ReadOnlyDatabase()
        await db.connect()
        try:
            if all_streams:
                # prompt_version partitions AND groups: pooling separate
                # prompt conditions would let the earlier one win the dedup
                # window and hide the newer one completely. Abstentions are
                # counted but never averaged into a Brier.
                rows = await db.fetchall(
                    """WITH ranked AS (
                         SELECT *,ROW_NUMBER() OVER (
                           PARTITION BY stream,arm,probability_kind,event_family,
                                        horizon_bucket,prompt_version
                           ORDER BY observed_at,forecast_key) rn
                         FROM forecast_score_facts
                         WHERE score_version='binary-proper-v1')
                       SELECT stream,arm,probability_kind,horizon_bucket,
                              prompt_version,
                              COUNT(*) forecasts,
                              SUM(abstained) abstains,
                              AVG(CASE WHEN abstained=0 THEN brier END) brier,
                              AVG(CASE WHEN abstained=0 THEN market_brier END) market_brier
                         FROM ranked WHERE rn=1
                        GROUP BY stream,arm,probability_kind,horizon_bucket,
                                 prompt_version
                        ORDER BY stream,arm,probability_kind,horizon_bucket,
                                 prompt_version""")
                rows = [dict(row) for row in rows]
            else:
                rows = await EvaluationStore(db).summary()
            table = Table(title="Intelligence × Exploration Evaluation")
            table.add_column("Arm")
            if all_streams:
                table.add_column("Kind")
            table.add_column("Model")
            table.add_column("Prompt")
            table.add_column("N", justify="right")
            table.add_column("Abst", justify="right")
            table.add_column("Brier", justify="right")
            table.add_column("Market", justify="right")
            table.add_column("Δ vs market", justify="right")
            for row in rows:
                arm = row.get("arm_name") or f"{row['stream']}:{row['arm']}"
                model = row.get("model") or ""
                values = [arm]
                if all_streams:
                    values.append(f"{row['probability_kind']} {row['horizon_bucket']}")
                n = int(row["forecasts"] or 0)
                abstains = int(row["abstains"] or 0)
                # "forecast-v2" -> "v2". The whole point of this column is
                # telling conditions apart at a glance, which a truncated
                # "forecast-…" defeats on a narrow terminal.
                prompt = (row["prompt_version"] or "").replace("forecast-", "")
                values.extend([model, prompt or "—", str(n),
                               f"{abstains}/{n}" if abstains else "—"])
                # NULL brier means every observation in this group was an
                # abstention. Report that, rather than printing a 0.0000 the
                # arm never earned.
                if row["brier"] is None or row["market_brier"] is None:
                    values.extend(["—", "—", "[dim]all abstained[/]"])
                else:
                    brier, market = float(row["brier"]), float(row["market_brier"])
                    values.extend([f"{brier:.4f}", f"{market:.4f}",
                                   f"{market - brier:+.4f}"])
                table.add_row(*values)
            console.print(table)
        finally:
            await db.close()
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


@main.command("ibkr-calibration")
@click.option("--arm", default="", help="Restrict to one model_alias.")
@click.option("--width", default=0.02, show_default=True,
              help="Probability bucket width for the calibration curve.")
def ibkr_calibration(arm: str, width: float):
    """Where to set the ETF arms' entry thresholds, read from resolved outcomes.

    The gate is AND(confidence >= floor, probability >= min_prob) and its
    defaults were tuned for a deterministic analyzer that returns 0.70/HIGH by
    construction. The OpenAI arms produce 0.43-0.56 at MEDIUM_LOW, so the gate
    rejects everything. This report is how the per-arm overrides get chosen
    from evidence rather than from four days of unscored output.

    Read the LOWER bound, never the hit rate. A 5-session directional call is a
    coin flip until proven otherwise.
    """
    async def _run():
        from auramaur.evaluation.etf_calibration import (
            COIN, CONFIDENCE_RANKS, Forecast, brier_score, clearance,
            confidence_bands, probability_bands, samples_for_reliable_edge,
            suggested_thresholds, threshold_sweep,
        )
        from auramaur.strategy.ibkr_edge_economics import (
            required_conviction, round_trip_cost_bps,
        )
        from auramaur.web.db import ReadOnlyDatabase

        settings = Settings()
        db = ReadOnlyDatabase()
        await db.connect()
        try:
            clause, params = "", []
            if arm:
                clause, params = " WHERE model_alias = ?", [arm]
            rows = await db.fetchall(
                f"""SELECT model_alias, probability, confidence, actual_outcome,
                           due_at
                    FROM ibkr_etf_forecasts{clause}
                    ORDER BY model_alias, id""", tuple(params))
            if not rows:
                console.print("[yellow]No ETF forecasts recorded.[/]")
                return
            by_arm: dict[str, list] = {}
            pending: dict[str, list[str]] = {}
            for row in rows:
                by_arm.setdefault(row["model_alias"], []).append(Forecast(
                    float(row["probability"]), str(row["confidence"]),
                    None if row["actual_outcome"] is None
                    else int(row["actual_outcome"])))
                if row["actual_outcome"] is None:
                    pending.setdefault(row["model_alias"], []).append(
                        str(row["due_at"]))

            for alias in sorted(by_arm):
                forecasts = by_arm[alias]
                resolved = [f for f in forecasts if f.resolved]
                # The gate is ECONOMIC now: expected edge against cost. The
                # old prob/confidence thresholds no longer decide anything, and
                # printing them as "the gate" was actively misleading.
                cap = settings.ibkr.etf_paper_budget_usd
                horizon = int(settings.ibkr.etf_signal_horizon_days)
                cost_bps = round_trip_cost_bps(
                    cap, commission_usd=settings.ibkr.etf_fee_per_order_usd,
                    spread_bps=3.0, slippage_bps=settings.ibkr.etf_slippage_bps)
                ceiling = max(abs(f.probability - 0.5) for f in forecasts)
                brier = brier_score(forecasts)
                head = (f"[bold]{alias}[/] — {len(resolved)}/{len(forecasts)} "
                        f"resolved | horizon {horizon} sessions | "
                        f"${cap:,.0f} at {cost_bps:.0f}bps round trip | "
                        f"peak conviction {ceiling:.3f}")
                if brier is not None:
                    head += f" | Brier {brier:.3f}"
                console.print(Panel(head, expand=False))
                # What the arm's peak conviction can actually reach, given
                # this capital and horizon. An instrument needing more than the
                # model has ever produced is out of the universe, not behind a
                # tighter threshold.
                reach = [(sym, required_conviction(vol, horizon, cost_bps))
                         for sym, vol in (("SLV", 0.644), ("GLD", 0.284),
                                          ("XLE", 0.210), ("QQQ", 0.190),
                                          ("SPY", 0.127), ("TLT", 0.093))]
                inside = [f"{s_}({n:.3f})" for s_, n in reach if n <= ceiling]
                console.print(
                    "  reachable at this arm's peak conviction: "
                    + (", ".join(inside) if inside else
                       "[red]nothing — no instrument's edge covers its fees[/]"))
                verdict = clearance(
                    forecasts,
                    min_resolved=settings.ibkr.etf_min_resolved_to_trade)
                console.print(
                    f"  trading: {'[green]CLEARED[/]' if verdict.cleared else '[yellow]LOCKED[/]'}"
                    f" — {verdict.reason}")
                if not resolved:
                    nxt = min(pending.get(alias, [])) if pending.get(alias) else "—"
                    console.print(
                        f"  [dim]Nothing resolved yet; first matures {nxt[:10]}. "
                        f"No threshold is choosable from this.[/]\n")
                    continue

                conf_table = Table(title="by stated confidence", expand=False)
                for col in ("band", "n", "hit rate", "95% low", "95% high",
                            "vs forecast", "beats coin"):
                    conf_table.add_column(col, justify="right")
                for band in confidence_bands(resolved):
                    conf_table.add_row(
                        band.label, str(band.n), f"{band.realized:.1%}",
                        f"{band.lo:.1%}", f"{band.hi:.1%}",
                        f"{band.calibration_gap:+.1%}",
                        "[green]yes[/]" if band.beats_coin else "[dim]no[/]")
                console.print(conf_table)

                curve = Table(title="calibration curve", expand=False)
                for col in ("bucket", "n", "mean forecast", "realized",
                            "95% low", "gap"):
                    curve.add_column(col, justify="right")
                for band in probability_bands(resolved, width=width):
                    curve.add_row(
                        band.label, str(band.n), f"{band.mean_forecast:.3f}",
                        f"{band.realized:.1%}", f"{band.lo:.1%}",
                        f"{band.calibration_gap:+.1%}")
                console.print(curve)

                candidates = suggested_thresholds(forecasts, width=width)
                # Always sweep the floor the arm can actually reach, not only
                # the configured one. A floor that selects nothing tells you
                # the gate is shut but not where to move it. Use the most
                # PERMISSIVE observed band: it maximises the sample and lets
                # min_prob do the filtering, which is the knob with a
                # continuum. Whether tightening confidence would help instead
                # is what the band table above answers.
                reachable = min(
                    (b.label for b in confidence_bands(resolved)),
                    key=lambda c: CONFIDENCE_RANKS.get(c, 99), default="LOW")
                # Confidence no longer gates; sweep the reachable band only.
                floors = [reachable]
                sweeps = {floor: threshold_sweep(resolved, candidates,
                                                 min_confidence=floor)
                          for floor in floors}
                for floor, sweep in sweeps.items():
                    tag = " (reachable)"
                    sweep_table = Table(
                        title=f"threshold sweep — confidence floor {floor}{tag}",
                        expand=False)
                    for col in ("min_prob", "selected", "hit rate", "95% low",
                                "n needed"):
                        sweep_table.add_column(col, justify="right")
                    for entry in sweep:
                        sweep_table.add_row(
                            f"{entry.threshold:.2f}", str(entry.n),
                            f"{entry.realized:.1%}" if entry.n else "—",
                            f"{entry.lo:.1%}" if entry.n else "—",
                            "—" if entry.samples_needed is None
                            else str(entry.samples_needed))
                    console.print(sweep_table)
                    if not any(e.n for e in sweep):
                        console.print(
                            f"  [red]Confidence floor {floor} selects zero "
                            f"resolved forecasts at every threshold[/] — it is "
                            f"the binding gate, and it is checked BEFORE "
                            f"probability. Moving min_prob alone changes "
                            f"nothing.")

                verdict = sweeps[reachable]
                best = [e for e in verdict if e.n and e.lo > COIN]
                if best:
                    pick = max(best, key=lambda e: e.lo)
                    console.print(
                        f"  [green]Supported[/]: at confidence floor "
                        f"{reachable}, min_prob {pick.threshold:.2f} selects "
                        f"{pick.n} forecasts at {pick.realized:.1%} "
                        f"(95% low {pick.lo:.1%} > 50%).\n")
                else:
                    strongest = max((e for e in verdict if e.n),
                                    key=lambda e: e.realized, default=None)
                    note = ""
                    if strongest is not None:
                        need = samples_for_reliable_edge(strongest.realized)
                        note = (f" Best is {strongest.realized:.1%} on "
                                f"n={strongest.n} at "
                                f"min_prob {strongest.threshold:.2f}; "
                                + (f"~{need} resolutions at that rate would be "
                                   f"needed to distinguish it from chance."
                                   if need else
                                   "no sample size rescues a sub-coin rate."))
                    console.print(
                        f"  [yellow]Not yet supported[/]: at floor {reachable}, "
                        f"no threshold's lower bound clears a coin flip.{note}\n")

            console.print(
                "[dim]These intervals assume independent forecasts and the "
                "ETF arms' are not: ~28 correlated instruments (SPY/VTI/QQQ/XLK "
                "co-move) are forecast on overlapping 5-session windows, so the "
                "effective sample is well below the row count and the true "
                "interval is WIDER than shown. Treat every bound here as "
                "optimistic — if it does not clear a coin flip on these "
                "numbers, it certainly does not in reality.[/]")
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


# Pre-registered operator checks. SOURCE OF TRUTH is the dated comment on
# each decision in runtime/config/defaults.local.yaml — update BOTH places
# when a decision is made or retired. Each row:
# (strategy_source, venue, category, epoch, settle_bar, usd_floor,
#  review_by) — the operator pre-committed to re-examine the cell when live
# settlements since epoch reach settle_bar OR cumulative live realized PnL
# reaches usd_floor, whichever comes first. review_by (nullable ISO date)
# is a CALENDAR checkpoint for cells whose entries settle too far out for
# the settlement counter to ever adjudicate them: the llm_kalshi lane's
# first live entries (2026-08-02) settle in 2028-29, so its cells get a
# 90-day mark-to-market + exit-realized review instead of waiting years.
_PREREGISTERED_CHECKS = (
    # llm exemption + bounded live categories (2026-07-28, widened 07-29)
    ("llm", "polymarket", "politics_us", "2026-07-28", 30, -50.0, None),
    ("llm", "polymarket", "politics_intl", "2026-07-29", 25, -40.0, None),
    ("llm", "polymarket", "crypto", "2026-07-29", 25, -40.0, None),
    ("llm", "polymarket", "legal", "2026-07-29", 15, -25.0, None),
    # llm_kalshi lane (split 2026-07-28, widened 2026-08-02); calendar
    # checkpoint 90d after the widening that produced the first entries
    ("llm_kalshi", "kalshi", "politics_us", "2026-07-28", 30, -50.0,
     "2026-10-31"),
    ("llm_kalshi", "kalshi", "politics_intl", "2026-08-02", 25, -40.0,
     "2026-10-31"),
    ("llm_kalshi", "kalshi", "science", "2026-08-02", 15, -25.0,
     "2026-10-31"),
)


def preregistered_check_status(settled: int, realized: float,
                               settle_bar: int, usd_floor: float,
                               review_due: bool = False) -> str:
    """'' | 'NEAR' | 'REVIEW' | 'FIRED'.

    FIRED means a pre-committed trigger crossed; REVIEW means the calendar
    checkpoint arrived (judge the cell on marks + exit realizations — its
    settlements are years out); NEAR (>= 80% of either numeric trigger)
    exists so the re-examination can be prepared before the bar is crossed,
    not discovered after. FIRED outranks REVIEW outranks NEAR.
    """
    if settled >= settle_bar or realized <= usd_floor:
        return "FIRED"
    if review_due:
        return "REVIEW"
    if settled >= 0.8 * settle_bar or realized <= 0.8 * usd_floor:
        return "NEAR"
    return ""


@main.command("preregistered-checks")
def preregistered_checks():
    """Where does each open pre-registered operator check stand?

    Every operator promotion or category widening carries a dated
    re-examine trigger in the runtime override (N settled entries or -$X
    cumulative, whichever first). A trigger only works if someone counts;
    this report is the counter. FIRED demotes nothing by itself — it means
    the operator owes the cell the decision they pre-committed to.
    """
    async def _run():
        from auramaur.web.db import ReadOnlyDatabase

        db = ReadOnlyDatabase()
        await db.connect()
        try:
            from datetime import date

            today = date.today().isoformat()
            table = Table(title="pre-registered checks — live cells",
                          expand=False)
            for c in ("cell", "epoch", "settled", "bar", "realized $",
                      "floor $", "review by", "status"):
                table.add_column(c, justify="right")
            for (src, venue, cat, epoch, bar, floor,
                 review_by) in _PREREGISTERED_CHECKS:
                row = await db.fetchone(
                    """SELECT SUM(CASE WHEN kind='settlement' THEN 1 ELSE 0
                                  END) AS settled,
                              COALESCE(SUM(pnl), 0.0) AS realized
                         FROM pnl_ledger
                        WHERE strategy_source=? AND venue=? AND category=?
                          AND is_paper=0 AND realized_at >= ?""",
                    (src, venue, cat, epoch))
                settled = int(row["settled"] or 0)
                realized = float(row["realized"] or 0.0)
                status = preregistered_check_status(
                    settled, realized, bar, floor,
                    review_due=bool(review_by) and today >= review_by)
                colour = {"FIRED": "red", "REVIEW": "magenta",
                          "NEAR": "yellow"}.get(status, "green")
                table.add_row(
                    f"{src} × {cat}", epoch, str(settled), str(bar),
                    f"{realized:+.2f}", f"{floor:+.0f}", review_by or "—",
                    f"[{colour}]{status or 'ok'}[/]")
            console.print(table)
            console.print(
                "  [dim]settled counts live kind='settlement' rows since the "
                "epoch; realized $ sums ALL live realized pnl (early exits "
                "included — a loss counts however it was taken). FIRED = the "
                "pre-committed re-examination is due, not an automatic "
                "demotion. REVIEW = the calendar checkpoint arrived: judge "
                "the cell on mark-to-market + exit realizations (its entries "
                "settle years out; marks on thin books are soft — treat as "
                "direction, not verdict). Triggers' source of truth: the "
                "dated comments in runtime/config/defaults.local.yaml — "
                "update both when a decision changes.[/]")
        finally:
            await db.close()
    asyncio.run(_run())


@main.command("evidence-accrual")
@click.option("--days", default=14, show_default=True,
              help="Window for the recent-accrual columns.")
def evidence_accrual(days: int):
    """Is countable graduation evidence actually accumulating?

    `require_executable_fills` admits only venue_fill / book_cross /
    trade_through. A paper fill is stamped `book_cross` only when it CROSSED
    the book — a resting maker fill is `synthetic` and can never count, which
    is correct: live might never have filled it.

    Until 2026-07-28 the gateway resolved the book from `orderbook_snapshots`,
    a table a separate recorder populates on its own cadence, so 109 of 111
    filled decisions had no book at all and were uncountable. This report
    exists so that failure is visible rather than discovered months later.
    """
    async def _run():
        from auramaur.web.db import ReadOnlyDatabase

        settings = Settings()
        credible = set(settings.graduation.credible_fill_evidence)
        db = ReadOnlyDatabase()
        await db.connect()
        try:
            rows = await db.fetchall(
                f"""WITH latest AS (
                        SELECT strategy_source,
                               MAX(holdout_starts_at) AS holdout_starts_at
                          FROM strategy_experiments e
                         WHERE registered_at = (SELECT MAX(registered_at)
                                                  FROM strategy_experiments e2
                                                 WHERE e2.strategy_source
                                                       = e.strategy_source)
                         GROUP BY strategy_source
                    )
                    SELECT d.strategy_source,
                           COUNT(*) AS decisions,
                           SUM(d.filled = 1) AS filled,
                           SUM(d.best_ask IS NOT NULL) AS with_book,
                           SUM(d.filled = 1 AND d.fill_evidence IN
                               ({",".join("?" * len(credible))})) AS countable,
                           SUM(d.is_holdout = 1) AS holdout,
                           MAX(l.holdout_starts_at) AS holdout_opens,
                           MAX(l.holdout_starts_at) <= datetime('now')
                               AS holdout_open_now,
                           MAX(d.observed_at) AS last
                      FROM decision_snapshots d
                      LEFT JOIN latest l
                             ON l.strategy_source = d.strategy_source
                     WHERE d.observed_at >= datetime('now', ?)
                     GROUP BY d.strategy_source
                     ORDER BY countable DESC, filled DESC""",
                (*sorted(credible), f"-{int(days)} days"))
            if not rows:
                console.print(f"[yellow]No decisions in the last {days}d.[/]")
                return
            table = Table(title=f"countable evidence accrual — last {days}d",
                          expand=False)
            for c in ("strategy", "decisions", "filled", "book", "countable",
                      "taker rate", "holdout", "holdout opens", "last"):
                table.add_column(c, justify="right")
            for r in rows:
                filled = int(r["filled"] or 0)
                countable = int(r["countable"] or 0)
                rate = f"{countable / filled:.0%}" if filled else "—"
                colour = ("green" if countable and countable == filled
                          else "yellow" if countable else "red")
                if r["holdout_opens"] is None:
                    opens = "[dim]—[/]"
                elif r["holdout_open_now"]:
                    opens = "[green]open[/]"
                else:
                    opens = f"[yellow]{str(r['holdout_opens'])[5:16]}[/]"
                table.add_row(
                    str(r["strategy_source"])[:24], str(r["decisions"]),
                    str(filled), str(int(r["with_book"] or 0)),
                    f"[{colour}]{countable}[/]", rate,
                    str(int(r["holdout"] or 0)), opens, str(r["last"])[5:16])
            console.print(table)
            console.print(
                "  [dim]book = best_ask captured (the fix). countable = filled "
                "AND evidence in credible_fill_evidence. taker rate = share of "
                "fills that CROSSED; a resting maker fill is correctly "
                "uncountable. holdout = decisions inside the scored window — "
                "everything before holdout_starts_at is permanently "
                "uncountable however good it is. holdout opens = the LATEST "
                "version's clock: ANY change to the strategy's frozen config "
                "(or the shared risk keys) re-versions it and restarts the "
                "14-day warmup, discarding accrued evidence.[/]")
        finally:
            await db.close()
    asyncio.run(_run())
