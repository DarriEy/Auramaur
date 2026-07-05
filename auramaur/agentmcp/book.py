"""The agent's paper book — write tools backed by the isolated ``agent.db``.

Unconstrained SIZING by design (the S1 decision): no RiskManager, no edge
floor, no $25 ceiling. The agent sizes and manages its own risk — that *is*
the paradigm under test. Safety comes from structure, not policy: this only
ever runs in the agent process (``AURAMAUR_LIVE=false``, no live credentials)
against a separate sqlite file, so it cannot reach a real venue however it's
called.

PRICES are NOT unconstrained (the S4 decision, 2026-07): every fill executes
at the bot database's recorded quote for the token, never at a caller-supplied
price. The 2026-07 incident showed why — an agent driving this book through
direct scripts "settled" still-active markets at $1.00 (and re-bought and
settled them again), fabricating its entire scorecard. Verification lives HERE
at the book layer, not in the MCP server, precisely because direct imports
bypass the server. Concretely:

  * a fill on a market the bot database doesn't know is REJECTED (the A/B
    premise is "same universe as the bot"; an unknown market has no
    verifiable price);
  * the caller's ``price`` is advisory — the executed price is the bot DB's
    token quote (``requested_price`` is echoed back for transparency);
  * a SELL is clamped to the held size, so a position cannot be realized
    twice.

Resolved markets need no special settlement path: the bot's resolution
tracking pins their recorded prices, so closing at the quote IS settlement.

Accounting is the SAME primitives the live bot uses, so the agent's book is
directly comparable to ``auramaur.db``:
  * ``PnLTracker.record_fill`` — fills + cost_basis + pnl_ledger (realized on
    sells) + daily_stats.
  * a ``portfolio`` upsert — the denormalized position row, same shape the
    strategies write (see ``bias_harvest._record_position``).
"""

from __future__ import annotations

import os
import uuid

from auramaur.agentmcp.market_data import MarketData
from auramaur.broker.pnl import PnLTracker
from auramaur.db.database import Database
from auramaur.exchange.models import Fill, OrderSide, TokenType
from config.settings import Settings


class UnverifiablePrice(ValueError):
    """The requested fill has no verifiable price in the bot's database."""


class AgentBook:
    """Records the agent's paper trades into its isolated ledger."""

    def __init__(
        self,
        db: Database,
        settings: Settings,
        market_data: MarketData | None = None,
    ) -> None:
        self._db = db
        self._pnl = PnLTracker(db, settings)
        # The quote source used to verify every fill price. Defaults to a
        # read-only view over the bot's database so that DIRECT constructions
        # (scripts importing AgentBook, not just the MCP server) get the same
        # enforcement. Injectable for tests.
        self._quotes = market_data or MarketData(
            os.environ.get("AURAMAUR_DB_PATH", "auramaur.db")
        )

    async def _verified_price(self, market_id: str, token: TokenType) -> float:
        """The bot DB's current quote for ``token`` — the only price a fill may
        execute at. Raises :class:`UnverifiablePrice` when the market is
        unknown or carries no usable quote."""
        try:
            quote = await self._quotes.get_quote(market_id)
        except Exception as e:  # ro DB missing/locked — fail CLOSED, not open
            raise UnverifiablePrice(
                f"quote lookup failed for {market_id}: {e}"
            ) from e
        if not quote.get("found"):
            raise UnverifiablePrice(
                f"market {market_id} is not in the bot universe — "
                "no verifiable price"
            )
        yes = float(quote.get("outcome_yes_price") or 0.0)
        no = float(quote.get("outcome_no_price") or 0.0)
        if token == TokenType.YES:
            price = yes
        else:
            # Same fallback the exchange client uses: a stale/zero NO quote
            # is derived from the YES side.
            price = no if no > 0.01 else (1.0 - yes if yes > 0 else 0.0)
        if not (0.0 < price < 1.0 + 1e-9):
            raise UnverifiablePrice(
                f"market {market_id} has no usable {token.value} quote"
            )
        return min(1.0, price)

    async def _token_cost_basis(self, market_id: str, token: str) -> tuple[float, float]:
        """``(avg_cost, size)`` for the EXACT (market, token) paper row — unlike
        ``PnLTracker.get_cost_basis`` which collapses to the dominant token."""
        row = await self._db.fetchone(
            "SELECT avg_cost, size FROM cost_basis "
            "WHERE market_id = ? AND token = ? AND is_paper = 1",
            (market_id, token),
        )
        if row is None:
            return 0.0, 0.0
        return float(row["avg_cost"]), float(row["size"])

    async def _sync_portfolio_row(
        self, market_id: str, token: str, *, exchange: str, category: str
    ) -> dict:
        """Reconcile the portfolio row to the post-fill cost_basis. Fully-closed
        positions (size ≈ 0) are removed so the snapshot stays clean."""
        avg_cost, size = await self._token_cost_basis(market_id, token)
        if size <= 1e-9:
            await self._db.execute(
                "DELETE FROM portfolio WHERE market_id = ? AND token = ? AND is_paper = 1",
                (market_id, token),
            )
            await self._db.commit()
            return {"market_id": market_id, "token": token, "size": 0.0, "avg_price": 0.0}

        await self._db.execute(
            """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
               current_price, unrealized_pnl, category, token, token_id,
               is_paper, updated_at)
               VALUES (?, ?, 'BUY', ?, ?, ?, 0, ?, ?, '', 1, datetime('now'))
               ON CONFLICT(market_id, is_paper, token) DO UPDATE SET
                   size = excluded.size,
                   avg_price = excluded.avg_price,
                   current_price = excluded.current_price,
                   category = excluded.category,
                   updated_at = excluded.updated_at""",
            (market_id, exchange, size, avg_cost, avg_cost, category, token),
        )
        await self._db.commit()
        return {"market_id": market_id, "token": token, "size": size, "avg_price": avg_cost}

    async def place_trade(
        self,
        market_id: str,
        token: str,
        side: str,
        size: float,
        price: float,
        *,
        exchange: str = "polymarket",
        category: str = "",
        fee: float = 0.0,
    ) -> dict:
        """Record one paper fill AT THE BOT DB'S QUOTE for the token (``price``
        is the caller's request, echoed back as ``requested_price``; it never
        sets the fill price). BUY opens/adds; SELL trims/closes (realizing P&L
        at ``(quote - avg_cost) * size``) and is clamped to the held size.
        Returns the resulting position and the realized-P&L delta."""
        token_t = TokenType(token.upper())
        side_t = OrderSide(side.upper())
        if size <= 0:
            raise ValueError("size must be positive")

        exec_price = await self._verified_price(market_id, token_t)

        if side_t == OrderSide.SELL:
            _, held = await self._token_cost_basis(market_id, token_t.value)
            if held <= 1e-9:
                return {
                    "filled": None,
                    "rejected": "sell_without_position",
                    "market_id": market_id,
                    "token": token_t.value,
                }
            # A position can only be realized once.
            size = min(size, held)

        realized_before = await self._pnl.get_realized_pnl()
        fill = Fill(
            order_id=f"agent-{uuid.uuid4().hex[:16]}",
            market_id=market_id,
            token=token_t,
            side=side_t,
            size=size,
            price=exec_price,
            fee=fee,
            is_paper=True,
        )
        await self._pnl.record_fill(fill)
        realized_after = await self._pnl.get_realized_pnl()

        position = await self._sync_portfolio_row(
            market_id, token_t.value, exchange=exchange, category=category
        )
        return {
            "filled": {"side": side_t.value, "size": size, "price": exec_price,
                       "requested_price": price,
                       "token": token_t.value, "market_id": market_id},
            "position": position,
            "realized_delta": round(realized_after - realized_before, 4),
        }

    async def close_position(
        self, market_id: str, token: str, price: float, *, fee: float = 0.0
    ) -> dict:
        """Sell the entire current paper position in (market, token) at the bot
        DB's quote (``price`` is advisory, as in :meth:`place_trade`). No-op
        (with a note) if nothing is held."""
        token_t = TokenType(token.upper())
        _, size = await self._token_cost_basis(market_id, token_t.value)
        if size <= 1e-9:
            return {"closed": False, "reason": "no_position",
                    "market_id": market_id, "token": token_t.value}
        result = await self.place_trade(
            market_id, token_t.value, "SELL", size, price, fee=fee
        )
        return {"closed": True, **result}
