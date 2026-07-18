"""Coinbase shadow fills for the Kraken directional paper book."""

from __future__ import annotations

import structlog

from auramaur.broker.pnl import PnLTracker
from auramaur.exchange.models import Fill, OrderSide, TokenType

log = structlog.get_logger()


class CoinbasePaperBook:
    """Price identical decisions on Coinbase without credentials or execution."""

    _PRODUCTS = {
        "XBTUSDC": "BTC-USDC",
        "ETHUSDC": "ETH-USDC",
        "SOLUSDC": "SOL-USDC",
    }

    def __init__(self, settings, client, db) -> None:
        self._settings = settings
        self._client = client
        self._pnl = PnLTracker(db, settings)

    async def record_shadow_fill(self, pair: str, side: OrderSide, qty: float) -> None:
        product = self._PRODUCTS.get(pair)
        if product is None or qty <= 0:
            return
        try:
            quote = await self._client.get_quote(product)
        except Exception as exc:  # noqa: BLE001 — comparator must not break Kraken
            log.warning("coinbase.paper.quote_error", pair=pair, error=str(exc)[:120])
            return
        if quote is None:
            log.warning("coinbase.paper.quote_unavailable", pair=pair, product=product)
            return
        price = quote.ask if side == OrderSide.BUY else quote.bid
        notional = qty * price
        fee_pct = self._settings.coinbase.paper_fee_pct / 100.0
        fill = Fill(
            order_id=f"coinbase-paper-{pair}",
            market_id=f"coinbase:{product}",
            token_id=product,
            side=side,
            token=TokenType.YES,
            size=qty,
            price=price,
            fee=notional * fee_pct,
            is_paper=True,
        )
        await self._pnl.record_fill(fill)
        log.info("coinbase.paper.fill", pair=pair, product=product,
                 side=side.value, price=price, qty=round(qty, 10), fee=fill.fee)
