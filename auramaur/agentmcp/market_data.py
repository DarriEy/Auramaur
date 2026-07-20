"""Read-only market data for the agent — the SAME universe the live bot sees.

The agent trades paper against *real* prices, so it reads the bot's
``auramaur.db`` (markets / orderbook_snapshots / lens_verdicts / news_items)
rather than duplicating discovery. The connection is opened ``mode=ro`` (URI),
so the kernel — not a code convention — guarantees the agent can never mutate
the bot's database. Writes go only to the agent's own ``agent.db`` (see
``book.py``).
"""

from __future__ import annotations

import aiosqlite


class MarketData:
    """Read-only views over the bot's database for scan/quote/evidence."""

    def __init__(self, auramaur_db_path: str) -> None:
        self._path = auramaur_db_path

    async def _connect(self) -> aiosqlite.Connection:
        # mode=ro: open existing DB read-only, never create, never write.
        conn = await aiosqlite.connect(f"file:{self._path}?mode=ro", uri=True)
        conn.row_factory = aiosqlite.Row
        # WAL readers can hit the bot's checkpoint lock; wait briefly instead
        # of erroring (the bot holds busy_timeout=30000 on its side).
        await conn.execute("PRAGMA busy_timeout=5000")
        return conn

    async def scan_markets(
        self,
        category: str | None = None,
        query: str | None = None,
        min_volume: float = 0.0,
        min_liquidity: float = 0.0,
        limit: int = 25,
    ) -> list[dict]:
        """Active markets, most-traded first, with optional category/text/size
        filters. The agent's discovery surface."""
        clauses = ["active = 1"]
        params: list[object] = []
        if category:
            clauses.append("category = ?")
            params.append(category)
        if query:
            clauses.append("question LIKE ?")
            params.append(f"%{query}%")
        if min_volume:
            clauses.append("volume >= ?")
            params.append(min_volume)
        if min_liquidity:
            clauses.append("liquidity >= ?")
            params.append(min_liquidity)
        sql = (
            "SELECT id, exchange, question, category, outcome_yes_price, "
            "outcome_no_price, volume, liquidity, end_date "
            "FROM markets WHERE " + " AND ".join(clauses) +
            " ORDER BY volume DESC LIMIT ?"
        )
        params.append(max(1, min(limit, 200)))
        conn = await self._connect()
        try:
            rows = await (await conn.execute(sql, tuple(params))).fetchall()
        finally:
            await conn.close()
        return [dict(r) for r in rows]

    async def get_quote(self, market_id: str) -> dict:
        """Latest prices + freshest order-book depth (per token) for a market."""
        conn = await self._connect()
        try:
            mrow = await (await conn.execute(
                "SELECT id, exchange, question, category, outcome_yes_price, "
                "outcome_no_price, volume, liquidity, clob_token_yes, "
                "clob_token_no, last_updated FROM markets WHERE id = ?",
                (market_id,),
            )).fetchone()
            if mrow is None:
                return {"market_id": market_id, "found": False}
            books = await (await conn.execute(
                "SELECT token_id, best_bid, best_ask, bid_size, ask_size, mid, "
                "recorded_at FROM orderbook_snapshots WHERE market_id = ? "
                "AND recorded_at = (SELECT MAX(recorded_at) FROM orderbook_snapshots "
                "ob2 WHERE ob2.market_id = orderbook_snapshots.market_id "
                "AND ob2.token_id = orderbook_snapshots.token_id) "
                "GROUP BY token_id",
                (market_id,),
            )).fetchall()
        finally:
            await conn.close()
        out = dict(mrow)
        out["found"] = True
        out["order_book"] = [dict(b) for b in books]
        return out

    async def get_evidence(self, market_id: str, news_limit: int = 5) -> dict:
        """The bot's evidence surface for a market: the resolution-lens verdict
        (fair prob, gap, mechanism, reasoning) plus any matched news items."""
        conn = await self._connect()
        try:
            mrow = await (await conn.execute(
                "SELECT question, description, category, end_date "
                "FROM markets WHERE id = ?",
                (market_id,),
            )).fetchone()
            verdict = await (await conn.execute(
                "SELECT fair_prob, gap_score, mechanism, reasoning, verified, "
                "grounded_fair, checked_at FROM lens_verdicts WHERE market_id = ?",
                (market_id,),
            )).fetchone()
            news = await (await conn.execute(
                "SELECT source, title, url, published_at, relevance_score "
                "FROM news_items WHERE market_ids LIKE ? "
                "ORDER BY published_at DESC LIMIT ?",
                (f"%{market_id}%", max(1, min(news_limit, 25))),
            )).fetchall()
        finally:
            await conn.close()
        return {
            "market_id": market_id,
            "market": dict(mrow) if mrow else None,
            "lens_verdict": dict(verdict) if verdict else None,
            "news": [dict(n) for n in news],
        }
