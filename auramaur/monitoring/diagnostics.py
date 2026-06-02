"""Diagnostics: error digest + attribution rendering.

Turns the firehose JSON log into an actionable "what's erroring" digest and
renders the (otherwise invisible) performance attribution. Shared by the
`errors` / `attribution` CLI commands and the cockpit health panel.

``summarize_errors`` is pure over already-parsed log records, so the cockpit can
feed it the tail it already read without a second file read.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Log levels treated as problems worth surfacing.
_PROBLEM_LEVELS = ("warning", "error", "critical")

# Events logged at warning level purely for visibility (not problems). Filtered
# from the digest by default so real errors aren't drowned out — these are the
# bot announcing normal live-order activity.
_BENIGN_EVENTS = frozenset({
    "order.live",            # live CLOB/venue order placement
    "kraken.order",          # kraken spot order placement
    "ibkr_equity.order.live",
})


def summarize_errors(records: list[dict], *, top: int = 8, include_benign: bool = False) -> dict:
    """Aggregate problem-level log records by event.

    ``records`` are parsed JSON log dicts in chronological order. Returns counts
    by level plus the most frequent problem events with their latest message.
    Benign visibility-warnings (``_BENIGN_EVENTS``) are excluded unless
    ``include_benign`` so the digest highlights real failures.
    """
    counts: Counter = Counter()
    by_level: Counter = Counter()
    latest: dict[str, tuple[str, str]] = {}
    suppressed = 0
    for rec in records:
        lvl = str(rec.get("level", "")).lower()
        if lvl not in _PROBLEM_LEVELS:
            continue
        ev = rec.get("event", "?")
        if not include_benign and ev in _BENIGN_EVENTS:
            suppressed += 1
            continue
        counts[ev] += 1
        by_level[lvl] += 1
        msg = rec.get("error") or rec.get("err") or rec.get("reason") or ""
        latest[ev] = (rec.get("timestamp", ""), str(msg)[:90])
    top_events = [
        {"event": ev, "count": n, "last_ts": latest[ev][0], "last_msg": latest[ev][1]}
        for ev, n in counts.most_common(top)
    ]
    return {
        "errors": by_level.get("error", 0) + by_level.get("critical", 0),
        "warnings": by_level.get("warning", 0),
        "total": sum(counts.values()),
        "suppressed": suppressed,
        "top": top_events,
    }


def _read_tail_records(path: str, max_bytes: int) -> list[dict]:
    """Parse the last ``max_bytes`` of a JSON log into dicts (newest last)."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - max_bytes))
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


def gather_log_errors(path: str, *, max_bytes: int = 8_000_000, top: int = 15,
                      include_benign: bool = False) -> dict:
    """Error digest over the tail of the JSON log (default last ~8 MB)."""
    records = _read_tail_records(path, max_bytes)
    summary = summarize_errors(records, top=top, include_benign=include_benign)
    try:
        summary["scanned_mb"] = round(min(max_bytes, os.path.getsize(path)) / 1e6, 1)
    except OSError:
        summary["scanned_mb"] = 0.0
    summary["records"] = len(records)
    return summary


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _money(v: float) -> str:
    return f"[{'green' if v >= 0 else 'red'}]${v:+.2f}[/]"


def render_error_digest(s: dict) -> Panel:
    suppressed = s.get("suppressed", 0)
    supp = f", {suppressed} benign hidden" if suppressed else ""
    head = Text.from_markup(
        f"[bold]{s['errors']}[/] errors · [bold]{s['warnings']}[/] warnings  "
        f"[dim](last ~{s.get('scanned_mb', 0)} MB, {s.get('records', 0)} records{supp})[/]")
    if not s["top"]:
        body = Group(head, Text.from_markup("\n[green]No errors or warnings in the scanned window.[/]"))
        return Panel(body, title="error digest", border_style="green")

    t = Table(expand=True)
    t.add_column("count", justify="right", style="bold")
    t.add_column("event", style="cyan", no_wrap=True)
    t.add_column("latest message", style="dim")
    t.add_column("when", justify="right", style="dim", no_wrap=True)
    for e in s["top"]:
        when = (e["last_ts"] or "")[11:19] or "—"
        t.add_row(str(e["count"]), e["event"], e["last_msg"][:60] or "—", when)
    border = "red" if s["errors"] else ("yellow" if s["warnings"] else "green")
    return Panel(Group(head, Text(""), t), title="error digest", border_style=border)


def error_panel_compact(s: dict) -> Panel:
    """Tight error panel for the cockpit: counts + the top few offenders."""
    lines = [Text.from_markup(
        f"[{'red' if s['errors'] else 'dim'}]{s['errors']} err[/]  "
        f"[{'yellow' if s['warnings'] else 'dim'}]{s['warnings']} warn[/]")]
    for e in s["top"][:4]:
        lines.append(Text.from_markup(f"[dim]{e['count']:>3}[/] [cyan]{e['event'][:30]}[/]"))
    if not s["top"]:
        lines.append(Text.from_markup("[green]clean[/]"))
    border = "red" if s["errors"] else ("yellow" if s["warnings"] else "green")
    return Panel(Group(*lines), title="health", border_style=border)


# --------------------------------------------------------------------------
# Doctor — operational health snapshot
# --------------------------------------------------------------------------

# Substrings that mark a problem event as a data-source failure.
_SOURCE_ERROR_HINTS = ("rate_limited", "api_error", "_failed", "source_error", "_error")

_STATUS_RANK = {"ok": 0, "info": 0, "warn": 1, "fail": 2}
_STATUS_ICON = {"ok": "[green]✓[/]", "info": "[dim]·[/]", "warn": "[yellow]![/]", "fail": "[red]✗[/]"}


def _chk(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}


async def gather_doctor(settings, db, *, max_bytes: int = 8_000_000) -> dict:
    """Operational health snapshot: is the bot running, alive, and quiet?

    Complements `readiness` (go-live edge criteria) — this is the "is anything
    broken right now" check: kill switch, pillar liveness, log freshness, error
    rate, data-source failures, and basic P&L/position sanity.
    """
    from auramaur.monitoring.cockpit import pillar_liveness, _parse_ts

    now = datetime.now(timezone.utc)
    checks: list[dict] = []

    checks.append(_chk("mode", "info", "LIVE" if settings.is_live else "PAPER"))
    checks.append(_chk("kill switch", "fail", "ACTIVE — trading halted")
                  if settings.kill_switch_active else _chk("kill switch", "ok", "off"))

    records = _read_tail_records(settings.logging.file, max_bytes)

    # Log freshness — is the process even writing?
    stamps = [t for t in (_parse_ts(r.get("timestamp", "")) for r in records) if t]
    if not stamps:
        checks.append(_chk("log freshness", "warn", "no recent log lines"))
    else:
        age = (now - max(stamps)).total_seconds()
        checks.append(_chk("log freshness", "ok" if age < 600 else "warn",
                           f"last line {int(age)}s ago"))

    # Pillar liveness — anything stale (>15m) or never-seen.
    pillars = pillar_liveness(records)
    alive, stale = [], []
    for p, ts in pillars.items():
        (alive if ts is not None and (now - ts).total_seconds() <= 900 else stale).append(p)
    checks.append(_chk("pillars", "ok", f"all {len(alive)} alive")
                  if not stale else
                  _chk("pillars", "warn", f"{len(alive)} alive; stale/never: {', '.join(stale)}"))

    # Error rate + data-source failures.
    err = summarize_errors(records, top=50)
    elvl = "fail" if err["errors"] > 500 else ("warn" if err["errors"] or err["warnings"] else "ok")
    checks.append(_chk("errors", elvl,
                       f"{err['errors']} err / {err['warnings']} warn (last ~{round(max_bytes/1e6,1)} MB)"))
    bad_sources = sorted({
        e["event"] for e in err["top"]
        if any(h in e["event"] for h in _SOURCE_ERROR_HINTS)
    })
    checks.append(_chk("data sources", "ok", "no source errors")
                  if not bad_sources else
                  _chk("data sources", "warn", f"{len(bad_sources)} erroring: {', '.join(bad_sources[:5])}"))

    # P&L / positions sanity (current mode).
    flag = 0 if settings.is_live else 1
    try:
        pos = await db.fetchone(
            "SELECT COUNT(*) c, COALESCE(SUM(size*COALESCE(current_price,avg_price)),0) v "
            "FROM portfolio WHERE is_paper = ?", (flag,))
        checks.append(_chk("positions", "info", f"{pos['c']} open, ${pos['v']:.0f} exposure"))
    except Exception:  # noqa: BLE001
        checks.append(_chk("positions", "warn", "could not read portfolio"))

    verdict = max((c["status"] for c in checks), key=lambda s: _STATUS_RANK.get(s, 0))
    return {"checks": checks, "verdict": verdict, "now": now}


def render_doctor(s: dict) -> Panel:
    verdict = s["verdict"]
    vtext = {"ok": "[green]HEALTHY[/]", "info": "[green]HEALTHY[/]",
             "warn": "[yellow]DEGRADED[/]", "fail": "[red]PROBLEM[/]"}.get(verdict, verdict)
    head = Text.from_markup(f"[bold]auramaur doctor[/] — {vtext}  "
                            f"[dim]{s['now'].strftime('%H:%M:%S UTC')}[/]")
    t = Table.grid(padding=(0, 2))
    t.add_column(justify="center")
    t.add_column(style="bold")
    t.add_column()
    for c in s["checks"]:
        t.add_row(_STATUS_ICON.get(c["status"], "?"), c["name"], c["detail"])
    border = {"fail": "red", "warn": "yellow"}.get(verdict, "green")
    return Panel(Group(head, Text(""), t), title="doctor", border_style=border)


def render_attribution(category_rows: list[dict], strategy_rows: list[dict], *, mode: str) -> Panel:
    """Per-category and per-strategy P&L / accuracy / Kelly."""
    cat = Table(title="by category", expand=True)
    cat.add_column("category", style="cyan")
    cat.add_column("pos", justify="right")
    cat.add_column("exposure", justify="right")
    cat.add_column("realized", justify="right")
    cat.add_column("unrealized", justify="right")
    cat.add_column("acc", justify="right")
    cat.add_column("kelly", justify="right")
    for r in category_rows:
        acc = r.get("accuracy")
        acc_str = f"{acc * 100:.0f}%" if acc is not None else "—"
        cat.add_row(
            r["category"], str(r["positions"]),
            f"${r['exposure']:.0f}",
            _money(r.get("realized_pnl", 0) or 0),
            _money(r.get("unrealized_pnl", 0) or 0),
            acc_str, f"{r.get('kelly_multiplier', 1.0):.2f}x",
        )

    strat = Table(title="by strategy", expand=True)
    strat.add_column("strategy", style="magenta")
    strat.add_column("trades", justify="right")
    strat.add_column("wins", justify="right")
    strat.add_column("win%", justify="right")
    strat.add_column("pnl", justify="right")
    for r in strategy_rows:
        n = r.get("trade_count", 0) or 0
        w = r.get("wins", 0) or 0
        wr = f"{w / n * 100:.0f}%" if n else "—"
        strat.add_row(
            r.get("strategy_source", "?") or "?", str(n), str(w), wr,
            _money(r.get("total_pnl", 0) or 0),
        )

    head = Text.from_markup(f"[bold]Performance attribution[/]  ([dim]{mode}[/])")
    return Panel(Group(head, Text(""), cat, Text(""), strat),
                 title="attribution", border_style="blue")
