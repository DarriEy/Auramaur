"""Live cockpit — a read-only rich.Live view of the running bot.

Fuses three sources into one screen so the bot is legible at a glance and never
*looks* stopped: DB state (positions / P&L), live venue balances, and pillar
liveness inferred from the JSON log tail ("kraken alive 4s ago", etc.).

Run in a second terminal alongside `auramaur run`:

    auramaur cockpit
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from datetime import datetime, timezone

from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# event-name substring -> pillar label. First match wins.
_PILLARS: "OrderedDict[str, list[str]]" = OrderedDict([
    ("polymarket", ["polymarket", "gamma", "strategic", "allocator", "engine."]),
    ("kalshi", ["kalshi"]),
    ("kraken", ["kraken."]),
    ("ibkr", ["ibkr"]),
    ("arb", ["arb"]),
    ("news", ["news", "reactor", "aggregator"]),
    ("market_maker", ["market_maker", "mm."]),
])

# events worth showing in the activity feed (action/decision-level)
_ACTIVITY = ["order.", "kraken.directional", "kraken.treasury.convert", "redemption",
             "arb.execute", "allocator.allocated", "refill", "kill_switch"]


def _ago(ts: datetime | None, now: datetime) -> tuple[str, str]:
    if ts is None:
        return "—", "dim"
    s = (now - ts).total_seconds()
    color = "green" if s < 90 else ("yellow" if s < 600 else "red")
    if s < 60:
        return f"{int(s)}s ago", color
    if s < 3600:
        return f"{int(s // 60)}m ago", color
    return f"{int(s // 3600)}h ago", color


def _parse_ts(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def read_log_tail(path: str, kb: int = 256) -> list[dict]:
    """Parse the last ~kb of the JSON log into dicts (newest last)."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - kb * 1024))
            chunk = f.read().decode("utf-8", errors="ignore")
    except OSError:
        return []
    out = []
    for line in chunk.splitlines()[1:]:  # first line may be partial
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def pillar_liveness(lines: list[dict]) -> "OrderedDict[str, datetime]":
    last: "OrderedDict[str, datetime]" = OrderedDict((p, None) for p in _PILLARS)
    for rec in lines:
        ev = rec.get("event", "")
        ts = _parse_ts(rec.get("timestamp", ""))
        if ts is None:
            continue
        for pillar, subs in _PILLARS.items():
            if any(sub in ev for sub in subs):
                if last[pillar] is None or ts > last[pillar]:
                    last[pillar] = ts
                break
    return last


def recent_activity(lines: list[dict], limit: int = 12) -> list[tuple[str, str]]:
    out = []
    for rec in lines:
        ev = rec.get("event", "")
        if not any(a in ev for a in _ACTIVITY):
            continue
        ts = _parse_ts(rec.get("timestamp", ""))
        hhmm = ts.strftime("%H:%M:%S") if ts else "--:--:--"
        detail = rec.get("market_id") or rec.get("pair") or rec.get("symbol") or ""
        extra = ""
        for k in ("side", "status", "usd", "size", "momentum", "amount"):
            if k in rec:
                extra += f" {k}={rec[k]}"
        out.append((hhmm, f"{ev} {detail}{extra}".strip()))
    return out[-limit:]


async def _portfolio_pnl(db, settings, is_paper_flag: int) -> dict:
    """Authoritative positions + P&L, ported from the old dashboard path.

    P&L = resolved (live book only) + realized cost-basis + unrealized
    mark-to-market. Deliberately does NOT sum ``trades.pnl`` — that legacy
    mirror field is unreliable, which is exactly why the cockpit and dashboard
    used to disagree. Which components apply is keyed off ``is_paper_flag``
    (the BOOK being read), not the process's own live gates — the web
    dashboard reads either book from a paper-gated process.
    """
    live_book = is_paper_flag == 0
    # cost_basis PK is (market_id, is_paper, token) — the join MUST match all
    # three. Joining without token fans each portfolio row out per cost_basis
    # side when a market holds both YES and NO, duplicating positions and
    # double-counting P&L (found via the web UI's duplicate-key warning).
    rows = await db.fetchall(
        """SELECT p.*, m.question, cb.avg_cost AS cb_avg_cost FROM portfolio p
           LEFT JOIN markets m ON p.market_id = m.id
           LEFT JOIN cost_basis cb ON cb.market_id = p.market_id
                                  AND cb.is_paper = p.is_paper
                                  AND cb.token = p.token
           WHERE p.is_paper = ?""",
        (is_paper_flag,),
    )
    positions = [
        {
            "market_id": r["market_id"], "question": r["question"] or r["market_id"],
            # token matters for identity: one market can hold BOTH YES and NO
            # (portfolio PK is market_id+is_paper+token), so market_id+side
            # alone does not identify a row.
            "token": r["token"],
            "exchange": r["exchange"] or "polymarket",
            "side": r["side"], "size": r["size"], "avg_price": r["avg_price"],
            "current_price": r["current_price"] or r["avg_price"],
            "pnl": (
                ((r["current_price"] or r["avg_price"]) - (r["cb_avg_cost"] or r["avg_price"]))
                * (r["size"] or 0)
            ),
        }
        for r in rows
    ]
    position_value = sum((p["current_price"] or 0) * (p["size"] or 0) for p in positions)

    # Recent signals (edge feed)
    sig_rows = await db.fetchall(
        """SELECT s.*, m.question FROM signals s
           LEFT JOIN markets m ON s.market_id = m.id
           ORDER BY s.timestamp DESC LIMIT 20"""
    )
    signals = [
        {
            "market_id": r["market_id"], "question": r["question"] or r["market_id"],
            "claude_prob": r["claude_prob"], "market_prob": r["market_prob"],
            "edge": r["edge"], "action": r["action"] or "",
            "strategy_source": r["strategy_source"] or "llm",
        }
        for r in sig_rows
    ]

    # Trade count from fills (authoritative), not the trades mirror.
    cnt_row = await db.fetchone(
        "SELECT COUNT(*) as cnt FROM fills WHERE is_paper = ?", (is_paper_flag,))
    trade_count = cnt_row["cnt"] if cnt_row else 0

    # Authoritative total P&L.
    resolved_component_sql = (
        "COALESCE((SELECT SUM(r.pnl) FROM resolution_pnl r), 0)"
        if live_book else "0"
    )
    resolved_exclusion_sql = (
        "AND cb.market_id NOT IN (SELECT market_id FROM resolution_pnl)"
        if live_book else ""
    )
    portfolio_resolved_exclusion_sql = (
        "AND p.market_id NOT IN (SELECT market_id FROM resolution_pnl)"
        if live_book else ""
    )
    pnl_row = await db.fetchone(
        f"""
        SELECT
            {resolved_component_sql}
          + COALESCE((
                SELECT SUM(cb.realized_pnl)
                FROM cost_basis cb
                WHERE cb.is_paper = ?
                  {resolved_exclusion_sql}
            ), 0)
          + COALESCE((
                SELECT SUM((COALESCE(p.current_price, p.avg_price)
                            - COALESCE(cb.avg_cost, p.avg_price)) * p.size)
                FROM portfolio p
                LEFT JOIN cost_basis cb ON cb.market_id = p.market_id
                                      AND cb.is_paper = p.is_paper
                                      AND cb.token = p.token
                WHERE p.is_paper = ?
                  {portfolio_resolved_exclusion_sql}
            ), 0) AS total_pnl
        """,
        (is_paper_flag, is_paper_flag),
    )
    total_pnl = pnl_row["total_pnl"] if pnl_row else 0

    dd_row = await db.fetchone(
        "SELECT max_drawdown FROM daily_stats ORDER BY date DESC LIMIT 1")
    drawdown = dd_row["max_drawdown"] if dd_row else 0

    if live_book:
        balance = None  # on-chain cash is shown by the running bot's syncer
    else:
        balance = settings.execution.paper_initial_balance + total_pnl

    return {
        "positions": positions,
        "position_count": len(positions),
        "position_value": position_value,
        "signals": signals,
        "trade_count": trade_count,
        "total_pnl": total_pnl,
        "drawdown": drawdown,
        "balance": balance,
    }


async def gather_state(db, settings, cache: dict | None = None,
                       book: str | None = None) -> dict:
    """Collect cockpit state. `cache` persists venue balances between refreshes.

    Pass ``cache=None`` for a one-shot snapshot (e.g. ``status``); venue
    balances are then always fetched fresh. ``book`` ("paper"/"live") reads a
    specific book regardless of the process's own gates — the web dashboard
    uses it to serve both views; ``None`` keeps the TUI behavior (the book the
    bot is actually trading).
    """
    now = datetime.now(timezone.utc)
    if book is None:
        is_paper_flag = 0 if settings.is_live else 1
    else:
        is_paper_flag = 0 if book == "live" else 1

    pf = await _portfolio_pnl(db, settings, is_paper_flag)

    # Live venue balances — refreshed at most every 20s (rate-limit friendly).
    if cache is None:
        venues = await venue_balances(settings)
    else:
        if time.time() - cache.get("bal_ts", 0) > 20:
            cache["bal_ts"] = time.time()
            cache["venues"] = await venue_balances(settings)
        venues = cache.get("venues", {})

    lines = read_log_tail(settings.logging.file)
    from auramaur.monitoring.diagnostics import summarize_errors
    return {
        "now": now,
        "is_live": settings.is_live,
        "transfers_armed": settings.transfers_armed,
        "kill_switch": settings.kill_switch_active,
        "venues": venues,
        "pillars": pillar_liveness(lines),
        "activity": recent_activity(lines),
        "health": summarize_errors(lines, top=4),
        **pf,
    }


async def venue_balances(settings) -> dict:
    """Fetch live venue balances. Public: the web dashboard's broker refreshes
    these on its own cadence so they never block a state request."""
    import asyncio
    out: dict[str, str] = {}
    if settings.kalshi.enabled:
        try:
            from auramaur.exchange.kalshi import KalshiClient
            kc = KalshiClient(settings=settings, paper_trader=None)
            # wait_for guards against the SDK blocking the whole cockpit.
            out["kalshi"] = f"${await asyncio.wait_for(kc.get_balance(), 8):.2f}"
            await kc.close()
        except Exception:
            out["kalshi"] = "—"
    if settings.kraken.enabled:
        try:
            from auramaur.exchange.kraken import KrakenSpotClient
            kk = KrakenSpotClient(settings)
            bal = await asyncio.wait_for(kk.get_balance(), 8)
            usdc = bal.get("USDC", 0.0)
            cad = bal.get("ZCAD", 0.0)
            crypto = [a for a, v in bal.items() if v > 0 and a != "USDC" and not a.startswith("Z")]
            out["kraken"] = (f"${usdc:.0f} USDC + {cad:.0f} CAD"
                             + (f" | spec: {','.join(crypto)}" if crypto else ""))
            await kk.close()
        except Exception:
            out["kraken"] = "—"
    return out


# --- panel builders (shared by the live layout and the one-shot snapshot) ---

def _header_panel(s: dict) -> Panel:
    mode = "[bold red]● LIVE[/]" if s["is_live"] else "[bold green]● PAPER[/]"
    gates = []
    if s["kill_switch"]:
        gates.append("[bold red]KILL SWITCH[/]")
    if s["transfers_armed"]:
        gates.append("[yellow]transfers armed[/]")
    return Panel(Text.from_markup(
        f"  AURAMAUR cockpit   {mode}   {s['now'].strftime('%H:%M:%S UTC')}   "
        + "  ".join(gates)))


def _venues_panel(s: dict) -> Panel:
    venues = Table.grid(padding=(0, 1))
    venues.add_column(style="magenta")
    venues.add_column()
    venues.add_row("polymarket", f"{s['position_count']} pos, ${s['position_value']:.0f}")
    for name, val in s["venues"].items():
        venues.add_row(name, val)
    return Panel(venues, title="venues")


def _pillars_panel(s: dict) -> Panel:
    pillars = Table(title="pillars", expand=True)
    pillars.add_column("pillar", style="cyan")
    pillars.add_column("last seen")
    for pillar, ts in s["pillars"].items():
        txt, color = _ago(ts, s["now"])
        pillars.add_row(pillar, f"[{color}]{txt}[/]")
    return Panel(pillars)


def _positions_panel(s: dict) -> Panel:
    t = Table(title="positions", expand=True)
    t.add_column("market", style="cyan", max_width=34, no_wrap=True)
    t.add_column("side", style="bold")
    t.add_column("size", justify="right")
    t.add_column("avg", justify="right")
    t.add_column("cur", justify="right")
    t.add_column("pnl", justify="right")
    # Show the biggest movers first so the live view is information-dense.
    for p in sorted(s["positions"], key=lambda p: abs(p.get("pnl", 0)), reverse=True)[:12]:
        pnl = p.get("pnl", 0)
        pc = "green" if pnl >= 0 else "red"
        t.add_row(
            (p.get("question") or p.get("market_id") or "")[:34],
            p.get("side", ""),
            f"{p.get('size', 0):.0f}",
            f"{p.get('avg_price', 0):.3f}",
            f"{p.get('current_price', 0):.3f}",
            f"[{pc}]${pnl:+.2f}[/]",
        )
    return Panel(t)


def _signals_panel(s: dict) -> Panel:
    t = Table(title="recent signals", expand=True)
    t.add_column("market", style="cyan", max_width=34, no_wrap=True)
    t.add_column("clP", justify="right")
    t.add_column("mkP", justify="right")
    t.add_column("edge", justify="right")
    t.add_column("action", style="bold")
    for sig in s["signals"][:10]:
        edge = sig.get("edge", 0) or 0
        ec = "green" if edge > 0 else "red"
        t.add_row(
            (sig.get("question") or sig.get("market_id") or "")[:34],
            f"{sig.get('claude_prob', 0) or 0:.3f}",
            f"{sig.get('market_prob', 0) or 0:.3f}",
            f"[{ec}]{edge:+.1f}%[/]",
            sig.get("action", ""),
        )
    return Panel(t)


def _activity_panel(s: dict) -> Panel:
    feed = Table.grid(padding=(0, 1))
    feed.add_column(style="dim", no_wrap=True)
    feed.add_column()
    for hhmm, txt in s["activity"]:
        feed.add_row(hhmm, txt[:70])
    return Panel(feed, title="activity")


def _health_panel(s: dict) -> Panel:
    from auramaur.monitoring.diagnostics import error_panel_compact
    return error_panel_compact(s.get("health", {"errors": 0, "warnings": 0, "top": []}))


def _footer_panel(s: dict) -> Panel:
    pnl = s["total_pnl"]
    pc = "green" if pnl >= 0 else "red"
    bal = s.get("balance")
    bal_str = "n/a" if bal is None else f"${bal:,.2f}"
    return Panel(Text.from_markup(
        f"  Balance: [bold]{bal_str}[/]   PnL: [{pc}]${pnl:+.2f}[/]   "
        f"Trades: {s['trade_count']}   Positions: {s['position_count']} "
        f"(${s['position_value']:.0f})   Drawdown: {s['drawdown']:.1f}%"))


def make_layout(s: dict, *, compact: bool = False):
    """Render cockpit state.

    ``compact=False`` → a full-screen ``rich.Layout`` for the live cockpit.
    ``compact=True``  → a stacked ``rich.Group`` snapshot for one-shot ``status``.
    Both share the same panel builders, so the numbers can never diverge.
    """
    if compact:
        return Group(
            _header_panel(s),
            _venues_panel(s),
            _pillars_panel(s),
            _health_panel(s),
            _positions_panel(s),
            _signals_panel(s),
            _footer_panel(s),
        )

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="center", ratio=2),
        Layout(name="right", ratio=1),
    )

    layout["header"].update(_header_panel(s))

    left = Layout()
    left.split_column(
        Layout(_venues_panel(s), size=len(s["venues"]) + 4),
        Layout(_pillars_panel(s)),
        Layout(_health_panel(s), size=8),
    )
    layout["left"].update(left)

    center = Layout()
    center.split_column(Layout(_positions_panel(s)), Layout(_signals_panel(s)))
    layout["center"].update(center)

    layout["right"].update(_activity_panel(s))
    layout["footer"].update(_footer_panel(s))
    return layout
