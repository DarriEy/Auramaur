"""Rich console display for bot events."""

from __future__ import annotations

from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# Track activity for status line
_cycle_stats: dict = {"signals": 0, "trades": 0, "filtered": 0, "analyzed": 0, "errors": 0}


def show_banner(mode: str, version: str) -> None:
    banner = Text()
    banner.append("AURAMAUR", style="bold cyan")
    banner.append(f" v{version}", style="dim")
    banner.append("  |  Mode: ", style="")
    if mode == "LIVE":
        banner.append("LIVE", style="bold red")
    else:
        banner.append("PAPER", style="bold green")
    banner.append(f"  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", style="dim")
    console.print(Panel(banner, style="cyan"))


def show_startup(sources: list[str], balance: float) -> None:
    console.print(f"  Data sources: [cyan]{', '.join(sources)}[/]")
    console.print(f"  Balance: [bold]${balance:,.2f}[/]")
    console.print()


def show_scan_results(total: int, candidates: int, filtered: int = 0, exchange: str = "") -> None:
    ex = f"[magenta]{exchange}[/] " if exchange else ""
    msg = f"[dim]{_ts()}[/] {ex}Scanned [bold]{total}[/] markets, [cyan]{candidates}[/] candidates"
    if filtered > 0:
        msg += f", [yellow]{filtered}[/] filtered"
    console.print(msg)
    _cycle_stats["filtered"] = filtered


def show_analyzing(question: str, market_id: str) -> None:
    q = question[:70] + "..." if len(question) > 70 else question
    console.print(f"[dim]{_ts()}[/] [dim]>>>[/] [cyan]{q}[/]")
    _cycle_stats["analyzed"] = _cycle_stats.get("analyzed", 0) + 1


def show_evidence(count: int, sources: dict[str, int]) -> None:
    if count == 0:
        console.print(f"         [yellow]No evidence found[/]")
    else:
        parts = [f"{s}:{n}" for s, n in sources.items() if n > 0]
        console.print(f"         [dim]Evidence:[/] [green]{count}[/] items ({', '.join(parts)})")


def show_analysis(claude_prob: float, market_prob: float, edge: float,
                  confidence: str, second_prob: float | None, divergence: float | None) -> None:
    edge_color = "green" if edge > 0 else "red"
    conf_color = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}.get(confidence, "white")

    # Show edge direction with arrow
    if edge > 10:
        arrow = "[bold green]>>>[/]"
    elif edge > 5:
        arrow = "[green]>>[/]"
    elif edge > 0:
        arrow = "[green]>[/]"
    elif edge > -5:
        arrow = "[red]<[/]"
    else:
        arrow = "[red]<<[/]"

    line = (
        f"         {arrow} Claude: [bold]{claude_prob:.0%}[/] vs Market: {market_prob:.0%}  "
        f"Edge: [{edge_color}]{edge:+.1f}%[/]  "
        f"[{conf_color}]{confidence}[/]"
    )
    if second_prob is not None and divergence is not None:
        line += f"  [dim]2nd:{second_prob:.0%} div:{divergence:.0%}[/]"
    console.print(line)

    _cycle_stats["signals"] = _cycle_stats.get("signals", 0) + 1


def show_risk_decision(approved: bool, reason: str, passed: int, failed: int, size: float = 0) -> None:
    total = passed + failed
    if approved:
        console.print(
            f"         [bold green]APPROVED[/] ({passed}/{total}) "
            f"Size: [bold]${size:.2f}[/]"
        )
    else:
        # Show concise rejection reason
        short_reason = reason.split(";")[0].strip() if ";" in reason else reason
        console.print(f"         [red]REJECTED[/] ({passed}/{total}) [dim]{short_reason}[/]")


def show_order(status: str, order_id: str, side: str, size: float, price: float, is_paper: bool, exchange: str = "", error_message: str = "") -> None:
    mode = "[green]PAPER[/]" if is_paper else "[bold red]LIVE[/]"
    side_color = "green" if side == "BUY" else "red"
    ex = f"[magenta]{exchange}[/] " if exchange else ""

    # SKIP_DUP is the sentinel order_id returned when the exchange's duplicate
    # guard suppresses an order because an equivalent resting order already
    # exists — the "failure" is actually our own prior request still working.
    if order_id == "SKIP_DUP":
        console.print(
            f"         [yellow]EXIT PENDING[/] {mode} {ex}[{side_color}]{side}[/] "
            f"${size * price:.2f} ({size:.1f} tokens @ ${price:.3f}) "
            f"[dim]resting order in flight[/]"
        )
    elif status == "rejected" or order_id == "ERROR":
        err_suffix = f" [red]{error_message}[/]" if error_message else ""
        console.print(
            f"         [bold red]ORDER FAILED[/] {mode} {ex}[{side_color}]{side}[/] "
            f"${size * price:.2f} ({size:.1f} tokens @ ${price:.3f}){err_suffix}"
        )
    else:
        console.print(
            f"         [bold]ORDER[/] {mode} {ex}[{side_color}]{side}[/] "
            f"${size * price:.2f} ({size:.1f} tokens @ ${price:.3f}) "
            f"[dim]{order_id[:16]}[/]"
        )
        _cycle_stats["trades"] = _cycle_stats.get("trades", 0) + 1


def show_order_dropped(market_id: str, reason: str) -> None:
    console.print(
        f"         [yellow]DROPPED[/] [dim]{market_id}[/] — {reason}"
    )


def show_cycle_summary(signals: int, trades: int, elapsed: float, exchange: str = "") -> None:
    trade_color = "green" if trades > 0 else "dim"
    ex = f"[magenta]{exchange}[/] " if exchange else ""
    console.print(
        f"[dim]{_ts()}[/] {ex}Cycle: "
        f"[cyan]{signals}[/] signals, [{trade_color}]{trades} trades[/] "
        f"[dim]({elapsed:.0f}s)[/]"
    )
    console.print()
    # Reset cycle stats
    _cycle_stats.update({"signals": 0, "trades": 0, "analyzed": 0})


def show_portfolio(balance: float, pnl: float, positions: int, drawdown: float, schedule_mode: str = "") -> None:
    pnl_color = "green" if pnl >= 0 else "red"

    # Build a compact but informative status line
    parts = [
        f"[dim]{_ts()}[/]",
        f"[bold]${balance:,.2f}[/]",
        f"PnL:[{pnl_color}]{pnl:+,.2f}[/]",
        f"Pos:{positions}",
    ]
    if drawdown > 0:
        dd_color = "yellow" if drawdown < 10 else "red"
        parts.append(f"DD:[{dd_color}]{drawdown:.1f}%[/]")

    if schedule_mode:
        mode_styles = {
            "peak": "[bold green]PEAK[/]",
            "off_peak": "[yellow]OFF-PEAK[/]",
            "quiet": "[dim]QUIET[/]",
            "starved": "[red]LOW CASH[/]",
        }
        parts.append(mode_styles.get(schedule_mode, schedule_mode))

    console.print(" | ".join(parts))


def show_claude_thinking(market_id: str, stage: str = "primary") -> None:
    label = "Asking Claude" if stage == "primary" else "Second opinion"
    console.print(f"         [dim]{label}...[/]")


def show_cache_hit() -> None:
    console.print(f"         [dim]Cache hit[/]")


def show_source_error(source: str, error: str) -> None:
    console.print(f"[dim]{_ts()}[/] [yellow]![/] {source}: {error}")


def show_error(msg: str) -> None:
    console.print(f"[dim]{_ts()}[/] [bold red]ERROR[/] {msg}")
    _cycle_stats["errors"] = _cycle_stats.get("errors", 0) + 1


def show_api_budget(calls_today: int, budget: int) -> None:
    remaining = budget - calls_today
    color = "green" if remaining > budget * 0.5 else "yellow" if remaining > budget * 0.2 else "red"
    console.print(f"[dim]{_ts()}[/] API budget: [{color}]{calls_today}/{budget}[/] calls today ({remaining} remaining)")


def show_world_model_update(cycle: int, beliefs: int, patterns: int, themes: list[str]) -> None:
    """Show world model update summary."""
    theme_str = ", ".join(themes[:4])
    if len(themes) > 4:
        theme_str += f" +{len(themes) - 4} more"
    console.print(
        f"         [bold cyan]World Model[/] cycle {cycle} | "
        f"{beliefs} beliefs, {patterns} patterns"
    )
    if theme_str:
        console.print(f"         [dim]Themes: {theme_str}[/]")


def show_arb_opportunity(exchange_a: str, exchange_b: str, question: str, spread: float) -> None:
    """Show arbitrage opportunity found."""
    q = question[:50] + "..." if len(question) > 50 else question
    console.print(
        f"[dim]{_ts()}[/] [bold yellow]ARB[/] {exchange_a}/{exchange_b} "
        f"spread: [bold]{spread:.1f}%[/]  {q}"
    )


def show_mm_quote(market_id: str, bid: float, ask: float, spread_bps: int) -> None:
    """Show market making quote placed."""
    console.print(
        f"[dim]{_ts()}[/] [blue]MM[/] {market_id[:12]} "
        f"bid:${bid:.3f} ask:${ask:.3f} spread:{spread_bps}bps"
    )


_BAR_CHARS = " ▏▎▍▌▋▊▉█"


def _alloc_bar(fraction: float, width: int = 10) -> str:
    """Render a proportional bar using Unicode block characters."""
    filled = fraction * width
    full_blocks = int(filled)
    remainder = filled - full_blocks
    partial_idx = int(remainder * 8)

    bar = _BAR_CHARS[-1] * full_blocks
    if full_blocks < width:
        bar += _BAR_CHARS[partial_idx]
        bar += " " * (width - full_blocks - 1)

    return bar


def build_category_stats_from_positions(
    positions: list,
    accuracy_map: dict[str, float | None] | None = None,
    kelly_map: dict[str, float] | None = None,
    category_lookup: dict[str, str] | None = None,
) -> list[dict]:
    """Aggregate LivePosition objects into per-category stats."""
    if accuracy_map is None:
        accuracy_map = {}
    if kelly_map is None:
        kelly_map = {}
    if category_lookup is None:
        category_lookup = {}

    seen: set[str] = set()
    cats: dict[str, dict] = {}
    for pos in positions:
        if pos.market_id in seen:
            continue
        seen.add(pos.market_id)
        cat = pos.category or category_lookup.get(pos.market_id) or "other"
        if cat not in cats:
            cats[cat] = {"category": cat, "positions": 0, "exposure": 0.0, "unrealized_pnl": 0.0}
        cats[cat]["positions"] += 1
        cats[cat]["exposure"] += pos.size * pos.avg_cost
        cats[cat]["unrealized_pnl"] += pos.unrealized_pnl

    result = sorted(cats.values(), key=lambda s: s["exposure"], reverse=True)
    for s in result:
        s["accuracy"] = accuracy_map.get(s["category"])
        s["kelly_multiplier"] = kelly_map.get(s["category"], 1.0)
    return result


def show_category_performance(stats: list[dict]) -> None:
    """Show per-category performance summary."""
    if not stats:
        return

    total_exposure = sum(s.get("exposure", 0) for s in stats)

    table = Table(
        title="Portfolio Allocation",
        show_header=True,
        title_style="bold cyan",
        border_style="dim",
        header_style="bold",
        pad_edge=True,
        padding=(0, 1),
    )
    table.add_column("Category", style="cyan", min_width=14)
    table.add_column("Pos", justify="right", style="dim")
    table.add_column("Exposure", justify="right")
    table.add_column("Alloc", justify="left", min_width=14)
    table.add_column("PnL", justify="right")
    table.add_column("Acc", justify="right")
    table.add_column("Kelly", justify="right")

    total_positions = 0
    total_pnl = 0.0

    for s in stats:
        positions = s.get("positions", 0)
        exposure = s.get("exposure", 0)
        pnl = s.get("unrealized_pnl", 0)
        accuracy = s.get("accuracy")
        mult = s.get("kelly_multiplier", 1.0) or 1.0

        total_positions += positions
        total_pnl += pnl

        frac = exposure / total_exposure if total_exposure > 0 else 0
        bar = _alloc_bar(frac)
        alloc_str = f"[cyan]{bar}[/] [dim]{frac:>4.0%}[/]"

        pnl_color = "green" if pnl >= 0 else "red"
        acc_str = f"{accuracy:.0%}" if accuracy is not None else "—"
        mult_color = "green" if mult >= 1.0 else ("yellow" if mult >= 0.5 else "red")

        table.add_row(
            s["category"],
            str(positions),
            f"${exposure:,.2f}",
            alloc_str,
            f"[{pnl_color}]${pnl:+,.2f}[/]",
            acc_str,
            f"[{mult_color}]{mult:.2f}x[/]",
        )

    total_pnl_color = "green" if total_pnl >= 0 else "red"
    table.add_section()
    table.add_row(
        "[bold]Total[/]",
        f"[bold]{total_positions}[/]",
        f"[bold]${total_exposure:,.2f}[/]",
        "",
        f"[bold {total_pnl_color}]${total_pnl:+,.2f}[/]",
        "",
        "",
    )

    console.print(table)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")
