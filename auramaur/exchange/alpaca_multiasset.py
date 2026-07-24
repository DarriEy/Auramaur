"""Free Alpaca IEX quote fallback for the multiasset paper books.

The multiasset program refuses paper fills on anything but real-time quotes
(delayed/frozen "cannot create credible fills"), and without IBKR market-data
subscriptions every US stock instrument sits `qualified_no_live_data` — the
global_etf book accrues no evidence at all. Alpaca's free IEX feed is
real-time bid/ask from a real exchange: legitimately credible for PAPER
evidence (provenance-stamped `alpaca_iex`), never execution-ready.

Scope is deliberately narrow: USD STOCK instruments only. FX already has live
IDEALPRO quotes; futures/bonds/options and non-US listings are not covered by
IEX and keep waiting on IBKR entitlements.
"""

from __future__ import annotations

import structlog

from auramaur.exchange.equity_market_data import AlpacaIEXMarketData
from auramaur.exchange.ibkr_instruments import ContractKind, InstrumentSpec
from auramaur.exchange.ibkr_market_data import IBKRReadOnlyMarketData, MarketDataQuote

log = structlog.get_logger()


class AlpacaMultiAssetQuotes:
    """IBKR-first multiasset quotes with Alpaca IEX fallback for US stocks."""

    def __init__(self, ibkr: IBKRReadOnlyMarketData, alpaca: AlpacaIEXMarketData):
        self._ibkr = ibkr
        self._alpaca = alpaca

    def __getattr__(self, name):
        # resolve/is_market_open/get_daily_bars/get_fx_to_usd/... — everything
        # except the quote path stays IBKR's (history needs no subscription).
        return getattr(self._ibkr, name)

    async def get_quote(self, spec: InstrumentSpec) -> MarketDataQuote | None:
        quote = None
        try:
            quote = await self._ibkr.get_quote(spec)
        except Exception as exc:  # noqa: BLE001 — fallback path handles it
            log.debug("alpaca_bridge.ibkr_quote_error", key=spec.key,
                      error=str(exc)[:120])
        if quote is not None and quote.source == "ibkr_live":
            return quote
        if spec.kind is not ContractKind.STOCK or spec.currency != "USD":
            return quote
        eq = await self._alpaca.get_quote(spec.symbol)
        if eq is None:
            return quote
        return MarketDataQuote(
            spec.key, eq.bid, eq.ask, eq.timestamp,
            int(getattr(quote, "con_id", 0) or 0),
            spec.currency, spec.multiplier, source="alpaca_iex")

    async def close(self) -> None:
        await self._alpaca.close()
        await self._ibkr.close()


def build_multiasset_market_data(settings, *, client_id: int | None = None):
    """Shared factory for the bot task and the preflight CLI so both probe
    through the same provenance chain (registry qualification must see the
    same quote sources the trading loop will)."""
    ibkr = IBKRReadOnlyMarketData(settings, client_id=client_id)
    if not settings.ibkr.multiasset_alpaca_quotes:
        return ibkr
    if not (settings.alpaca_api_key and settings.alpaca_api_secret):
        log.warning("alpaca_bridge.disabled_no_keys")
        return ibkr
    return AlpacaMultiAssetQuotes(ibkr, AlpacaIEXMarketData(
        settings.alpaca_api_key, settings.alpaca_api_secret,
        base_url=settings.alpaca_data_url,
        timeout=settings.alpaca_timeout_seconds))
