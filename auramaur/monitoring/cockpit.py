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


async def gather_state(db, settings, cache: dict) -> dict:
    """Collect cockpit state. `cache` persists balances between refreshes."""
    now = datetime.now(timezone.utc)
    is_paper_flag = 0 if settings.is_live else 1

    # Positions + P&L from DB
    rows = await db.fetchall(
        "SELECT COUNT(*) c, COALESCE(SUM(size*COALESCE(current_price,avg_price)),0) v "
        "FROM portfolio WHERE is_paper = ?", (is_paper_flag,))
    pos = rows[0] if rows else {"c": 0, "v": 0}
    pnl_row = await db.fetchone(
        "SELECT COUNT(*) cnt, COALESCE(SUM(pnl),0) pnl FROM trades WHERE is_paper = ?",
        (is_paper_flag,))

    # Live venue balances — refreshed at most every 20s (rate-limit friendly)
    if time.time() - cache.get("bal_ts", 0) > 20:
        cache["bal_ts"] = time.time()
        cache["venues"] = await _venue_balances(settings)

    lines = read_log_tail(settings.logging.file)
    return {
        "now": now,
        "is_live": settings.is_live,
        "transfers_armed": settings.transfers_armed,
        "kill_switch": settings.kill_switch_active,
        "position_count": pos["c"], "position_value": pos["v"],
        "trade_count": pnl_row["cnt"] if pnl_row else 0,
        "total_pnl": pnl_row["pnl"] if pnl_row else 0,
        "venues": cache.get("venues", {}),
        "pillars": pillar_liveness(lines),
        "activity": recent_activity(lines),
    }


async def _venue_balances(settings) -> dict:
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


def make_layout(s: dict) -> Layout:
    now = s["now"]
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(Layout(name="left"), Layout(name="right"))

    # Header — mode + gates
    mode = "[bold red]● LIVE[/]" if s["is_live"] else "[bold green]● PAPER[/]"
    gates = []
    if s["kill_switch"]:
        gates.append("[bold red]KILL SWITCH[/]")
    if s["transfers_armed"]:
        gates.append("[yellow]transfers armed[/]")
    layout["header"].update(Panel(Text.from_markup(
        f"  AURAMAUR cockpit   {mode}   {now.strftime('%H:%M:%S UTC')}   "
        + "  ".join(gates))))

    # Left — venues + pillar liveness
    venues = Table.grid(padding=(0, 1))
    venues.add_column(style="magenta")
    venues.add_column()
    venues.add_row("polymarket", f"{s['position_count']} pos, ${s['position_value']:.0f}")
    for name, val in s["venues"].items():
        venues.add_row(name, val)

    pillars = Table(title="pillars", expand=True)
    pillars.add_column("pillar", style="cyan")
    pillars.add_column("last seen")
    for pillar, ts in s["pillars"].items():
        txt, color = _ago(ts, now)
        pillars.add_row(pillar, f"[{color}]{txt}[/]")

    left = Layout()
    left.split_column(Layout(Panel(venues, title="venues"), size=len(s["venues"]) + 4),
                      Layout(Panel(pillars)))
    layout["left"].update(left)

    # Right — activity feed
    feed = Table.grid(padding=(0, 1))
    feed.add_column(style="dim", no_wrap=True)
    feed.add_column()
    for hhmm, txt in s["activity"]:
        feed.add_row(hhmm, txt[:70])
    layout["right"].update(Panel(feed, title="activity"))

    # Footer — P&L
    pnl = s["total_pnl"]
    pc = "green" if pnl >= 0 else "red"
    layout["footer"].update(Panel(Text.from_markup(
        f"  PnL: [{pc}]${pnl:+.2f}[/]   Trades: {s['trade_count']}   "
        f"Positions: {s['position_count']}   (${s['position_value']:.0f})")))
    return layout
