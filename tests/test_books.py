"""Tests for the strategy-books terminal view (monitoring/books.py)."""

from __future__ import annotations

import asyncio

from auramaur.db.database import Database
from auramaur.monitoring.books import (
    _book_modes,
    gather_books,
    render_books_panel,
    render_books_table,
)
from config.settings import Settings


def test_gather_books_attribution_and_modes():
    async def run():
        db = Database(":memory:")
        await db.connect()
        # Ledger: one live llm loss, one paper bias_harvest win.
        await db.execute(
            "INSERT INTO pnl_ledger (market_id, category, strategy_source, kind, "
            "token, qty, pnl, is_paper, source_ref) VALUES "
            "('m1', 'tech', 'llm', 'sell', 'YES', 1, -5.0, 0, 'r1'), "
            "('m2', 'tech', 'bias_harvest', 'settlement', 'YES', 1, 2.0, 1, 'r2')")
        # Portfolio: a live llm position, a paper bias_harvest position, and a
        # Kraken pair (no signals row — must attribute via exchange).
        await db.execute(
            "INSERT INTO signals (market_id, claude_prob, claude_confidence, "
            "market_prob, edge, strategy_source) VALUES "
            "('p1', 0.6, 'MEDIUM', 0.5, 10, 'llm'), "
            "('p2', 0.9, 'MEDIUM', 0.86, 4, 'bias_harvest')")
        await db.execute(
            "INSERT INTO portfolio (market_id, exchange, side, size, avg_price, "
            "current_price, is_paper) VALUES "
            "('p1', 'polymarket', 'BUY', 10, 0.50, 0.55, 0), "
            "('p2', 'polymarket', 'BUY', 10, 0.86, 0.86, 1), "
            "('XBTUSDC', 'kraken', 'BUY', 0.001, 70000, 70000, 0)")
        await db.execute(
            "INSERT INTO ibkr_paper_ledger (book, kind, pnl_usd, source_ref) VALUES "
            "('fx', 'trade', 5, 'ibkr-win'), "
            "('fx', 'trade', -2, 'ibkr-loss'), "
            "('fx', 'commission', -1, 'ibkr-fee')")
        await db.commit()

        books = {b["book"]: b for b in await gather_books(db)}
        assert books["llm"]["open_n"] == 1 and books["llm"]["live_pnl"] == -5.0
        assert books["bias_harvest"]["open_paper_n"] == 1
        assert books["bias_harvest"]["paper_pnl"] == 2.0
        assert books["kraken_directional"]["open_n"] == 1  # via exchange, not signals
        assert books["ibkr_fx"]["paper_n"] == 2
        assert books["ibkr_fx"]["paper_pnl"] == 2
        assert books["ibkr_fx"]["win_pct"] == 50

        # Renders without error.
        render_books_table(list(books.values()))
        await db.close()

    asyncio.run(run())


def test_book_modes_reflect_config_truth():
    s = Settings()
    s.bias_harvest.enabled = True
    s.bias_harvest.paper = True
    s.kraken.enabled = True
    s.kraken.directional_enabled = True
    s.kraken.directional_budget_usd = 0.0
    modes = dict((b, m) for b, m, _ in _book_modes(s))
    assert modes["bias_harvest"] == "PAPER"
    assert modes["kraken spec"] == "WIND-DOWN"
    # llm reflects global mode (paper in tests).
    assert modes["llm"] == ("LIVE" if s.is_live else "paper")
    render_books_panel(s, -96.21)  # renders without error


def test_kraken_llm_paper_override_is_reported_as_paper():
    s = Settings()
    s.kraken.enabled = True
    s.kraken.directional_enabled = True
    s.kraken.directional_budget_usd = 60.0
    s.kraken.directional_llm_paper = True
    modes = dict((book, mode) for book, mode, _ in _book_modes(s))
    assert modes["kraken spec"] == "PAPER"
