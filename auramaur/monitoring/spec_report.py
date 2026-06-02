"""Performance readout for the Kraken directional (spec) book.

Now that directional entries/exits record fills + realized P&L into cost_basis,
this answers the question that should drive any strategy redesign: is the spec
book net positive? It reports realized P&L (cost_basis), per-trip win/loss
walked from fills (weighted-average cost, matching PnLTracker), fees paid, and
open unrealized from the portfolio mirror — scoped to the configured
directional pairs and the current paper/live mode.
"""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _trips_from_fills(rows: list) -> list[float]:
    """Per-close realized P&L by walking fills per pair.

    Weighted-average cost; buy fees are not capitalized (matches PnLTracker), so
    these trip P&Ls reconcile with cost_basis.realized_pnl. ``rows`` must be
    ordered by timestamp.
    """
    by_pair: dict[str, list] = {}
    for r in rows:
        by_pair.setdefault(r["market_id"], []).append(r)
    trips: list[float] = []
    for fills in by_pair.values():
        size = 0.0
        cost = 0.0
        for f in fills:
            side = (f["side"] or "").upper()
            fsize = float(f["size"] or 0)
            fprice = float(f["price"] or 0)
            ffee = float(f["fee"] or 0)
            if side == "BUY":
                size += fsize
                cost += fsize * fprice
            elif size > 0:  # SELL — realize against running average cost
                avg = cost / size
                sell = min(fsize, size)
                trips.append((fprice - avg) * sell - ffee)
                size -= sell
                cost = avg * size
    return trips


async def gather_spec_performance(db, settings) -> dict:
    """Collect the spec book's performance for the current mode."""
    pairs = list(getattr(settings.kraken, "directional_pairs", []) or [])
    flag = 0 if settings.is_live else 1
    out = {
        "enabled": bool(getattr(settings.kraken, "directional_enabled", False)),
        "mode": "live" if settings.is_live else "paper",
        "pairs": pairs,
        "realized": 0.0, "fees": 0.0, "unrealized": 0.0, "net": 0.0,
        "trips": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
        "open_count": 0, "open_value": 0.0, "per_pair": [],
    }
    if not pairs:
        return out
    ph = ",".join("?" * len(pairs))

    cb_rows = await db.fetchall(
        f"SELECT market_id, realized_pnl FROM cost_basis "
        f"WHERE is_paper = ? AND market_id IN ({ph})", (flag, *pairs))
    fill_rows = await db.fetchall(
        f"SELECT market_id, side, size, price, fee, timestamp FROM fills "
        f"WHERE is_paper = ? AND market_id IN ({ph}) ORDER BY timestamp", (flag, *pairs))
    pos_rows = await db.fetchall(
        "SELECT market_id, size, current_price, unrealized_pnl "
        "FROM portfolio WHERE exchange = 'kraken' AND is_paper = ?", (flag,))

    realized = sum(float(r["realized_pnl"] or 0) for r in cb_rows)
    fees = sum(float(r["fee"] or 0) for r in fill_rows)
    trips = _trips_from_fills(fill_rows)
    wins = sum(1 for p in trips if p > 0)
    losses = sum(1 for p in trips if p < 0)

    unrealized = sum(float(r["unrealized_pnl"] or 0) for r in pos_rows)
    open_value = sum(float(r["size"] or 0) * float(r["current_price"] or 0) for r in pos_rows)

    pair_realized = {r["market_id"]: float(r["realized_pnl"] or 0) for r in cb_rows}
    pos_by_pair = {r["market_id"]: r for r in pos_rows}
    pair_trips: dict[str, int] = {}
    for r in fill_rows:
        if (r["side"] or "").upper() == "SELL":
            pair_trips[r["market_id"]] = pair_trips.get(r["market_id"], 0) + 1

    per_pair = []
    for pair in pairs:
        pos = pos_by_pair.get(pair)
        rl = pair_realized.get(pair, 0.0)
        unr = float(pos["unrealized_pnl"] or 0) if pos else 0.0
        ov = (float(pos["size"] or 0) * float(pos["current_price"] or 0)) if pos else 0.0
        if rl == 0 and ov == 0 and pair_trips.get(pair, 0) == 0:
            continue  # untouched pair — keep the table tight
        per_pair.append({
            "pair": pair, "trips": pair_trips.get(pair, 0),
            "realized": rl, "unrealized": unr, "open_value": ov,
        })

    out.update(
        realized=realized, fees=fees, unrealized=unrealized, net=realized + unrealized,
        trips=len(trips), wins=wins, losses=losses,
        win_rate=(wins / len(trips) * 100.0) if trips else 0.0,
        open_count=len(pos_rows), open_value=open_value,
        per_pair=sorted(per_pair, key=lambda x: x["realized"] + x["unrealized"]),
    )
    return out


def _money(v: float) -> str:
    return f"[{'green' if v >= 0 else 'red'}]${v:+.2f}[/]"


def render_spec_performance(s: dict) -> Panel:
    """Render the spec performance dict as a rich panel."""
    state = "enabled" if s["enabled"] else "[yellow]disabled[/]"
    head = Text.from_markup(f"[bold]Kraken spec — directional book[/]  ({s['mode']}, {state})")

    summary = Table.grid(padding=(0, 2))
    summary.add_column(justify="right", style="dim")
    summary.add_column()
    summary.add_row("Realized", f"{_money(s['realized'])}  "
                    f"[dim]{s['trips']} trips, {s['wins']}W/{s['losses']}L, "
                    f"{s['win_rate']:.0f}% win[/]")
    summary.add_row("Fees paid", f"[red]-${s['fees']:.2f}[/]")
    summary.add_row("Open", f"{s['open_count']} pos, ${s['open_value']:.2f} value, "
                    f"unrealized {_money(s['unrealized'])}")
    summary.add_row("Net", f"[bold]{_money(s['net'])}[/]")

    parts = [head, Text(""), summary]
    if s["per_pair"]:
        t = Table(title="by pair", expand=True)
        t.add_column("pair", style="magenta")
        t.add_column("trips", justify="right")
        t.add_column("realized", justify="right")
        t.add_column("open $", justify="right")
        t.add_column("unrealized", justify="right")
        for p in s["per_pair"]:
            t.add_row(p["pair"], str(p["trips"]), _money(p["realized"]),
                      f"${p['open_value']:.2f}", _money(p["unrealized"]))
        parts += [Text(""), t]
    elif not s["pairs"]:
        parts += [Text.from_markup("\n[dim]No directional pairs configured.[/]")]
    else:
        parts += [Text.from_markup("\n[dim]No spec activity recorded yet.[/]")]

    return Panel(Group(*parts), title="spec performance", border_style="magenta")
