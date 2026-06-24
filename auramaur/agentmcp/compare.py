"""S3 — head-to-head scorecard: the agent's book vs Auramaur's.

The fair A/B is **agent (paper) vs bot (paper)**: both are simulated against the
same market universe with no live-execution noise, and the bot's paper book *is*
the strategy ensemble the agent is trying to beat. The bot's **live** book is
shown alongside for context (its real-money arm), not as the head-to-head.

Realized P&L is sourced from ``pnl_ledger`` (the source of truth, one row per
realization), segmented exactly as ``auramaur pnl`` / the strategy-books panel.
"""

from __future__ import annotations

from auramaur.db.database import Database


async def _book_stats(db: Database, *, is_paper: bool) -> dict:
    """Realized ledger + open exposure for one mode of one database."""
    flag = 1 if is_paper else 0
    row = await db.fetchone(
        """SELECT COUNT(*) AS n,
                  COALESCE(SUM(pnl), 0) AS realized,
                  COALESCE(SUM(fees), 0) AS fees,
                  SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins
           FROM pnl_ledger WHERE is_paper = ?""",
        (flag,),
    )
    n = int(row["n"] or 0) if row else 0
    realized = float(row["realized"] or 0.0) if row else 0.0
    wins = int(row["wins"] or 0) if row else 0

    open_row = await db.fetchone(
        """SELECT COUNT(*) AS open_n,
                  COALESCE(SUM(size * avg_price), 0) AS open_usd
           FROM portfolio WHERE is_paper = ?""",
        (flag,),
    )
    cats = await db.fetchall(
        """SELECT COALESCE(NULLIF(category, ''), '(uncat)') AS category,
                  SUM(pnl) AS pnl, COUNT(*) AS n
           FROM pnl_ledger WHERE is_paper = ?
           GROUP BY 1 ORDER BY pnl DESC""",
        (flag,),
    )
    return {
        "events": n,
        "realized": round(realized, 2),
        "fees": round(float(row["fees"] or 0.0) if row else 0.0, 2),
        "win_pct": round(wins / n * 100.0, 1) if n else None,
        "per_event": round(realized / n, 4) if n else None,
        "open_n": int(open_row["open_n"] or 0) if open_row else 0,
        "open_usd": round(float(open_row["open_usd"] or 0.0) if open_row else 0.0, 2),
        "by_category": [
            {"category": c["category"], "pnl": round(float(c["pnl"] or 0.0), 2),
             "n": int(c["n"] or 0)}
            for c in (cats or [])
        ],
    }


def _verdict(agent: dict, bot_paper: dict) -> dict:
    """Score the head-to-head (agent paper vs bot paper)."""
    def _lead(a, b):
        if a is None and b is None:
            return None
        return "agent" if (a or 0) > (b or 0) else ("bot" if (b or 0) > (a or 0) else "tie")

    return {
        "realized_leader": _lead(agent["realized"], bot_paper["realized"]),
        "realized_gap": round((agent["realized"] or 0) - (bot_paper["realized"] or 0), 2),
        "per_event_leader": _lead(agent["per_event"], bot_paper["per_event"]),
        "win_pct_leader": _lead(agent["win_pct"], bot_paper["win_pct"]),
        "agent_has_history": agent["events"] > 0,
    }


async def build_comparison(agent_db_path: str, auramaur_db_path: str) -> dict:
    """Assemble the full agent-vs-bot scorecard from both ledgers."""
    agent_db = Database(agent_db_path)
    await agent_db.connect()
    try:
        agent = await _book_stats(agent_db, is_paper=True)
    finally:
        await agent_db.close()

    bot_db = Database(auramaur_db_path)
    await bot_db.connect()
    try:
        bot_paper = await _book_stats(bot_db, is_paper=True)
        bot_live = await _book_stats(bot_db, is_paper=False)
    finally:
        await bot_db.close()

    return {
        "agent_paper": agent,
        "bot_paper": bot_paper,
        "bot_live": bot_live,
        "verdict": _verdict(agent, bot_paper),
    }


def render_comparison(data: dict):
    """Render the scorecard as a rich Table + verdict line."""
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text

    a, bp, bl = data["agent_paper"], data["bot_paper"], data["bot_live"]
    v = data["verdict"]

    def _money(x):
        if x is None:
            return Text("—", style="dim")
        return Text(f"${x:+,.2f}", style="green" if x >= 0 else "red")

    def _opt(x, suffix="", pct=False):
        if x is None:
            return Text("—", style="dim")
        return Text(f"{x:.1f}%" if pct else f"{x}{suffix}")

    t = Table(title="Agent vs Auramaur — realized scorecard (pnl_ledger)")
    t.add_column("metric", style="cyan")
    t.add_column("Agent (paper)", justify="right")
    t.add_column("Bot (paper)", justify="right")
    t.add_column("Bot (live)", justify="right", style="dim")

    t.add_row("realized", _money(a["realized"]), _money(bp["realized"]), _money(bl["realized"]))
    t.add_row("events", str(a["events"]), str(bp["events"]), str(bl["events"]))
    t.add_row("win %", _opt(a["win_pct"], pct=True), _opt(bp["win_pct"], pct=True),
              _opt(bl["win_pct"], pct=True))
    t.add_row("$ / event", _money(a["per_event"]), _money(bp["per_event"]), _money(bl["per_event"]))
    t.add_row("fees", _money(-a["fees"]), _money(-bp["fees"]), _money(-bl["fees"]))
    t.add_row("open positions", str(a["open_n"]), str(bp["open_n"]), str(bl["open_n"]))
    t.add_row("open exposure", _money(a["open_usd"]), _money(bp["open_usd"]), _money(bl["open_usd"]))

    if not v["agent_has_history"]:
        line = Text("No agent trades yet — let the agent run a few sessions, then "
                    "compare.", style="yellow")
    else:
        who = {"agent": "Agent leads", "bot": "Bot leads", "tie": "Dead heat"}
        head = who.get(v["realized_leader"], "—")
        line = Text(
            f"{head} on realized by ${abs(v['realized_gap']):,.2f} "
            f"(paper vs paper) · per-event: {v['per_event_leader'] or '—'} · "
            f"win%: {v['win_pct_leader'] or '—'}.  Both books are paper; "
            f"normalize for exposure before drawing conclusions.",
            style="bold",
        )
    return Group(t, Text(""), line)
