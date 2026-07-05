"""``auramaur-trade`` MCP server — the Hermes agent-trader's view of Auramaur.

Tool surface (consumed by Hermes over stdio as ``mcp_auramaur-trade_<tool>``):

  Read (bot's universe, opened read-only):
    * scan_markets   — discover active markets, most-traded first
    * get_quote      — latest prices + order-book depth for a market
    * get_evidence   — resolution-lens verdict + matched news for a market
  Read (agent's own book):
    * get_portfolio  — open paper positions + realized/unrealized P&L
  Write (agent's own book — UNCONSTRAINED, no RiskManager):
    * place_trade    — record a paper BUY/SELL fill
    * close_position — sell the full paper position in a market/token

Two databases, deliberately split:
  * ``AGENT_DB_PATH`` (default ``agent.db``)     — the agent's book; read+write.
  * ``AURAMAUR_DB_PATH`` (default ``auramaur.db``) — the bot's universe; READ-ONLY
    (``mode=ro``). The agent reads the same markets the bot sees (fair A/B) but
    cannot mutate the bot's database.

Paper-only by construction: ``AURAMAUR_LIVE`` is forced off here before
``Settings`` is constructed, so ``is_live`` can never be True in this process —
the live-order gates are unreachable, not merely unused.
"""

from __future__ import annotations

import os

import sys
import logging
import structlog

# Redirect all structlog and stdlib logging to stderr so they don't corrupt JSON-RPC stdout
logging.basicConfig(level=logging.INFO, stream=sys.stderr, force=True)
structlog.configure(
    logger_factory=structlog.WriteLoggerFactory(file=sys.stderr)
)

# Force paper BEFORE Settings is ever constructed. Belt-and-suspenders on top of
# the separate process / no-live-credentials isolation.
os.environ["AURAMAUR_LIVE"] = "false"

from fastmcp import FastMCP  # noqa: E402  (import after the env guard, intentionally)

from auramaur.agentmcp.book import AgentBook  # noqa: E402
from auramaur.agentmcp.market_data import MarketData  # noqa: E402
from auramaur.broker.pnl import PnLTracker  # noqa: E402
from auramaur.db.database import Database  # noqa: E402
from auramaur.risk.portfolio import PortfolioTracker  # noqa: E402
from config.settings import Settings  # noqa: E402

mcp = FastMCP("auramaur-trade")


def _agent_db_path() -> str:
    return os.environ.get("AGENT_DB_PATH", "agent.db")


def _auramaur_db_path() -> str:
    return os.environ.get("AURAMAUR_DB_PATH", "auramaur.db")


def _market_data() -> MarketData:
    return MarketData(_auramaur_db_path())


# ----------------------------------------------------------------------
# Read tools — the bot's universe (read-only)
# ----------------------------------------------------------------------


@mcp.tool()
async def scan_markets(
    category: str | None = None,
    query: str | None = None,
    min_volume: float = 0.0,
    min_liquidity: float = 0.0,
    limit: int = 25,
) -> list[dict]:
    """Discover active markets (most-traded first) from the bot's live universe.

    Optional filters: ``category`` (exact), ``query`` (substring of the
    question), ``min_volume`` / ``min_liquidity``. Returns id, question,
    category, yes/no price, volume, liquidity, end_date.
    """
    return await _market_data().scan_markets(
        category=category, query=query, min_volume=min_volume,
        min_liquidity=min_liquidity, limit=limit,
    )


@mcp.tool()
async def get_quote(market_id: str) -> dict:
    """Latest prices and freshest order-book depth (best bid/ask/size/mid per
    token) for one market."""
    return await _market_data().get_quote(market_id)


@mcp.tool()
async def get_evidence(market_id: str) -> dict:
    """The bot's evidence for a market: the resolution-lens verdict (fair prob,
    gap score, mechanism, reasoning) plus any matched news items."""
    return await _market_data().get_evidence(market_id)


# ----------------------------------------------------------------------
# Read tool — the agent's own book
# ----------------------------------------------------------------------


async def _read_portfolio(db_path: str) -> dict:
    settings = Settings()  # is_live forced False by the env guard above
    db = Database(db_path)
    await db.connect()  # CREATE TABLE IF NOT EXISTS — safe on an empty file
    try:
        positions = await PortfolioTracker(db, settings).get_positions(is_paper=True)
        realized = await PnLTracker(db, settings).get_realized_pnl()
    finally:
        await db.close()

    marked = [
        {
            "market_id": p.market_id,
            "exchange": p.exchange,
            "side": p.side.value,
            "token": p.token.value,
            "size": p.size,
            "avg_price": p.avg_price,
            "current_price": p.current_price,
            "category": p.category,
            "unrealized": round((p.current_price - p.avg_price) * p.size, 4),
        }
        for p in positions
    ]
    return {
        "db_path": db_path,
        "is_paper": True,
        "open_positions": len(marked),
        "realized_pnl": round(realized, 4),
        "unrealized_pnl_marked": round(sum(m["unrealized"] for m in marked), 4),
        "positions": marked,
    }


@mcp.tool()
async def get_portfolio() -> dict:
    """Snapshot the agent's current paper portfolio from its isolated ledger:
    open positions (with marked unrealized), realized P&L, and marked
    unrealized total. Read only."""
    return await _read_portfolio(_agent_db_path())


# ----------------------------------------------------------------------
# Write tools — the agent's own book (UNCONSTRAINED)
# ----------------------------------------------------------------------


async def _with_book(fn):
    """Run ``fn(book)`` against a freshly-opened agent ledger, then close it."""
    settings = Settings()
    db = Database(_agent_db_path())
    await db.connect()
    try:
        return await fn(AgentBook(db, settings))
    finally:
        await db.close()


@mcp.tool()
async def place_trade(
    market_id: str,
    token: str,
    side: str,
    size: float,
    price: float,
) -> dict:
    """Record one paper fill in the agent's book. ``token`` is YES/NO, ``side``
    is BUY/SELL, ``size`` is shares, ``price`` is the per-share fill price.

    BUY opens or adds; SELL trims or closes, realizing P&L at
    ``(price - avg_cost) * size``. Unconstrained: no risk checks, no size cap —
    the agent owns its own risk. Paper only. Category/exchange are enriched from
    the bot's market record when available.
    """
    # Enrich category/exchange from the bot's universe so the position row is
    # well-formed; fall back to defaults if the market isn't catalogued.
    quote = await _market_data().get_quote(market_id)
    category = quote.get("category") or "" if quote.get("found") else ""
    exchange = quote.get("exchange") or "polymarket" if quote.get("found") else "polymarket"

    # Strictly enforce real-market pricing: override the LLM's price with the actual
    # database-quoted price for the token if available to prevent cheating/hallucinated exits.
    real_price = price
    if quote.get("found"):
        token_upper = token.upper()
        if token_upper == "YES" and quote.get("outcome_yes_price") is not None:
            real_price = float(quote["outcome_yes_price"])
        elif token_upper == "NO" and quote.get("outcome_no_price") is not None:
            real_price = float(quote["outcome_no_price"])

    return await _with_book(
        lambda book: book.place_trade(
            market_id, token, side, size, real_price,
            exchange=exchange, category=category,
        )
    )


@mcp.tool()
async def close_position(market_id: str, token: str, price: float) -> dict:
    """Sell the entire current paper position in (market, token) at ``price``,
    realizing its P&L. No-op if nothing is held. Paper only."""
    quote = await _market_data().get_quote(market_id)
    real_price = price
    if quote.get("found"):
        token_upper = token.upper()
        if token_upper == "YES" and quote.get("outcome_yes_price") is not None:
            real_price = float(quote["outcome_yes_price"])
        elif token_upper == "NO" and quote.get("outcome_no_price") is not None:
            real_price = float(quote["outcome_no_price"])

    return await _with_book(
        lambda book: book.close_position(market_id, token, real_price)
    )


# ----------------------------------------------------------------------
# Scoreboard — how the agent is doing vs the bot
# ----------------------------------------------------------------------


@mcp.tool()
async def compare_to_bot() -> dict:
    """Your head-to-head scorecard vs Auramaur: realized P&L, events, win%, and
    $/event for your paper book vs the bot's paper book (the fair A/B) and the
    bot's live book (context). Use it to judge whether your judgment is beating
    the strategy ensemble."""
    from auramaur.agentmcp.compare import build_comparison

    return await build_comparison(_agent_db_path(), _auramaur_db_path())


def main() -> None:
    mcp.run()  # stdio transport (Hermes default)


if __name__ == "__main__":
    main()
