"""Strategy-books view — the live terminal's picture of the edge-first system.

After the 2026-06 redesign the bot runs several independent strategy books
(llm, bias_harvest, entailment_arb, resolution_lens, market_maker,
arbitrage, kraken wind-down), each in its own mode, gated by graduation and
the name-the-gap audit. The old terminal narrated only the llm cycle; this
module renders the whole system:

  * ``render_books_panel`` — startup: every book with its TRUE mode (LIVE /
    PAPER / WIND-DOWN / off), the active gates, graduation mode, and the
    pnl_ledger lifetime number.
  * ``gather_books`` / ``render_books_table`` — periodic: per-book open
    exposure, realized P&L (live + paper, from the ledger), and win rate.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _book_modes(settings) -> list[tuple[str, str, str]]:
    """(book, mode, note) for every strategy book, from config truth."""
    rows: list[tuple[str, str, str]] = []
    rows.append(("llm", "LIVE" if settings.is_live else "paper",
                 "gates: divergence + name-the-gap"
                 if settings.risk.mispricing_gate_enabled
                 else "gates: divergence"))

    def paper_mode(cfg, note: str) -> tuple[str, str]:
        if not cfg.enabled:
            return "off", note
        return ("PAPER" if cfg.paper or not settings.is_live else "LIVE"), note

    bh = settings.bias_harvest
    rows.append(("bias_harvest", *paper_mode(
        bh, f"band {bh.band_lo:.2f}-{bh.band_hi:.2f}, ${bh.stake_usd:.0f}/mkt")))
    ea = settings.entailment_arb
    rows.append(("entailment_arb", *paper_mode(
        ea, f"min gap {ea.min_gap*100:.0f}c, both-or-nothing legs")))
    rl = settings.resolution_lens
    rows.append(("resolution_lens", *paper_mode(
        rl, f"gap>={rl.min_gap_score:.1f}, edge>={rl.min_edge*100:.0f}c")))
    at = settings.agent_trader
    rows.append(("agent_trader", *paper_mode(
        at, f"models: {', '.join(m.alias for m in at.models)} — one cell each")))
    ts = settings.term_structure
    rows.append(("term_structure", *paper_mode(
        ts, f">={ts.min_strikes} strikes, curve/24h, edge>={ts.min_edge_pts:.0f}pts")))
    va = settings.vol_anchor
    rows.append(("vol_anchor", *paper_mode(
        va, f"GBM thresholds, tau {va.tau_years:.2f}y, edge>={va.min_edge_pts:.0f}pts")))

    rows.append(("market_maker",
                 "LIVE" if settings.market_maker.enabled and settings.is_live
                 else ("paper" if settings.market_maker.enabled else "off"),
                 "graduation-exempt"))
    ol = settings.oddlot_tender
    rows.append(("oddlot_tender", *paper_mode(
        ol, "EDGAR scan, 99-sh entries, manual tender")))
    etf = settings.ibkr
    rows.append((
        "ibkr_etf_openai", "PAPER" if etf.etf_paper_enabled else "off",
        (f"{len(etf.etf_models)} cells × {len(etf.etf_symbols)} ETFs, "
         f"${etf.etf_paper_budget_usd:,.0f}/cell"),
    ))
    rows.append(("arbitrage",
                 "LIVE" if settings.arbitrage.enabled and settings.is_live
                 else ("paper" if settings.arbitrage.enabled else "off"),
                 "graduation-exempt"))

    k = settings.kraken
    if k.enabled and k.directional_enabled:
        if k.directional_budget_usd <= 0:
            rows.append(("kraken spec", "WIND-DOWN", "exits only, no re-entry"))
        else:
            rows.append(("kraken spec", "LIVE" if settings.is_live else "paper",
                         f"${k.directional_budget_usd:.0f} budget"))
    return rows


_MODE_STYLE = {"LIVE": "bold red", "PAPER": "green", "WIND-DOWN": "yellow",
               "paper": "green", "off": "dim"}


def render_books_panel(settings, ledger_live_total: float | None = None) -> Panel:
    """Startup panel: every book, its true mode, the gates, the ledger truth."""
    t = Table.grid(padding=(0, 2))
    t.add_column(style="cyan", min_width=16)
    t.add_column(min_width=10)
    t.add_column(style="dim")
    for book, mode, note in _book_modes(settings):
        t.add_row(book, Text(mode, style=_MODE_STYLE.get(mode, "")), note)

    g = settings.graduation
    t.add_row("graduation", Text(g.mode, style="yellow" if g.mode == "observe" else "bold"),
              f">= {g.min_events} events / {g.window_days}d to earn live")
    if ledger_live_total is not None:
        t.add_row("ledger", Text(f"${ledger_live_total:+,.2f}",
                                 style="green" if ledger_live_total >= 0 else "red"),
                  "lifetime live realized (auramaur pnl)")
    return Panel(t, title="strategy books", border_style="cyan")


async def gather_books(db) -> list[dict]:
    """Per-book ledger record + open exposure (entry-strategy attributed)."""
    rows = await db.fetchall(
        """SELECT COALESCE(NULLIF(strategy_source, ''), '(none)') AS book,
                  SUM(CASE WHEN is_paper = 0 THEN pnl ELSE 0 END) AS live_pnl,
                  SUM(CASE WHEN is_paper = 0 THEN 1 ELSE 0 END) AS live_n,
                  SUM(CASE WHEN is_paper = 1 THEN pnl ELSE 0 END) AS paper_pnl,
                  SUM(CASE WHEN is_paper = 1 THEN 1 ELSE 0 END) AS paper_n,
                  SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                  COUNT(*) AS n
           FROM pnl_ledger GROUP BY 1""")
    by_book = {r["book"]: dict(r) for r in (rows or [])}

    open_rows = await db.fetchall(
        """SELECT CASE WHEN p.exchange = 'kraken' THEN 'kraken_directional'
                  ELSE COALESCE(
                 (SELECT s.strategy_source FROM signals s
                  WHERE s.market_id = p.market_id
                    AND s.strategy_source IS NOT NULL
                    AND s.strategy_source != 'order_monitor'
                  ORDER BY s.timestamp ASC LIMIT 1), 'llm') END AS book,
                  SUM(CASE WHEN p.is_paper = 0 THEN 1 ELSE 0 END) AS open_n,
                  SUM(CASE WHEN p.is_paper = 0 THEN p.size * p.avg_price ELSE 0 END) AS open_usd,
                  SUM(CASE WHEN p.is_paper = 1 THEN 1 ELSE 0 END) AS open_paper_n
           FROM portfolio p GROUP BY 1""")
    for r in open_rows or []:
        by_book.setdefault(r["book"], {}).update(
            open_n=int(r["open_n"] or 0), open_usd=float(r["open_usd"] or 0.0),
            open_paper_n=int(r["open_paper_n"] or 0))

    etf_rows = await db.fetchall(
        """SELECT l.model_alias,
                  COUNT(*) AS n,
                  SUM(l.pnl) AS pnl,
                  SUM(CASE WHEN l.pnl > 0 THEN 1 ELSE 0 END) AS wins,
                  (SELECT COUNT(*) FROM ibkr_etf_positions p
                    WHERE p.model_alias = l.model_alias) AS open_n
             FROM ibkr_etf_ledger l GROUP BY l.model_alias""")
    for row in etf_rows or []:
        by_book[f"ibkr_etf_{row['model_alias']}"] = {
            "paper_n": int(row["n"] or 0), "paper_pnl": float(row["pnl"] or 0),
            "wins": int(row["wins"] or 0), "n": int(row["n"] or 0),
            "open_paper_n": int(row["open_n"] or 0),
        }

    out = []
    for book, d in sorted(by_book.items()):
        n = int(d.get("n") or 0)
        out.append({
            "book": book,
            "open_n": int(d.get("open_n") or 0),
            "open_usd": float(d.get("open_usd") or 0.0),
            "open_paper_n": int(d.get("open_paper_n") or 0),
            "live_n": int(d.get("live_n") or 0),
            "live_pnl": float(d.get("live_pnl") or 0.0),
            "paper_n": int(d.get("paper_n") or 0),
            "paper_pnl": float(d.get("paper_pnl") or 0.0),
            "win_pct": (int(d.get("wins") or 0) / n * 100.0) if n else None,
        })
    return out


def render_books_table(rows: list[dict]) -> Table:
    t = Table(title="Strategy Books (realized = pnl_ledger)")
    t.add_column("book", style="cyan")
    t.add_column("open (live)", justify="right")
    t.add_column("open (paper)", justify="right")
    t.add_column("live realized", justify="right")
    t.add_column("paper realized", justify="right")
    t.add_column("win%", justify="right")
    for r in rows:
        live = Text(f"{r['live_n']:3d} / ${r['live_pnl']:+,.2f}",
                    style="green" if r["live_pnl"] >= 0 else "red")
        paper = Text(f"{r['paper_n']:3d} / ${r['paper_pnl']:+,.2f}",
                     style="green" if r["paper_pnl"] >= 0 else "red")
        t.add_row(
            r["book"],
            f"{r['open_n']} / ${r['open_usd']:,.0f}" if r["open_n"] else "—",
            str(r["open_paper_n"]) if r.get("open_paper_n") else "—",
            live if r["live_n"] else Text("—", style="dim"),
            paper if r["paper_n"] else Text("—", style="dim"),
            f"{r['win_pct']:.0f}%" if r["win_pct"] is not None else "—",
        )
    return t
