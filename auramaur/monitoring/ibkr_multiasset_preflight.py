"""Readiness checks for all six IBKR local paper books."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from auramaur.exchange.ibkr_instruments import BY_BOOK, IBKRBook
from auramaur.exchange.ibkr_market_data import IBKRReadOnlyMarketData


@dataclass(frozen=True)
class MultiAssetPreflightResult:
    book: str
    severity: str
    detail: str


@dataclass(frozen=True)
class MultiAssetPreflightReport:
    results: tuple[MultiAssetPreflightResult, ...]

    @property
    def ready(self) -> bool:
        return not any(result.severity == "BLOCK" for result in self.results)


async def preflight(settings, db, *, client=None, timeout_seconds: int = 30):
    cfg = settings.ibkr
    results: list[MultiAssetPreflightResult] = []

    def add(book, severity, detail):
        results.append(MultiAssetPreflightResult(book, severity, detail))

    own_client = client is None
    client = client or IBKRReadOnlyMarketData(settings)
    if not getattr(client, "readonly", False) or hasattr(client, "place_order"):
        add("isolation", "BLOCK", "market-data client is not structurally read-only")
    else:
        add("isolation", "OK", "read-only client exposes no broker order method")

    required = {"ibkr_paper_positions", "ibkr_paper_fills",
                "ibkr_paper_ledger", "ibkr_paper_state"}
    rows = await db.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
    missing = required - {row["name"] for row in rows}
    add("database", "BLOCK" if missing else "OK",
        "missing: " + ", ".join(sorted(missing)) if missing
        else "isolated multi-asset accounting tables present")

    if not cfg.enabled:
        add("feature gate", "BLOCK", "ibkr.enabled must be true")
    else:
        mode = "enabled" if cfg.multiasset_paper_enabled else "staged (activation off)"
        add("feature gate", "OK", mode)

    for book in IBKRBook:
        book_cfg = cfg.multiasset_books[book.value]
        if not book_cfg.enabled:
            add(book.value, "WARN", "book disabled")
            continue
        spec = BY_BOOK[book][0]
        try:
            quote = await asyncio.wait_for(client.get_quote(spec), timeout_seconds)
            bars = await asyncio.wait_for(client.get_daily_bars(spec), timeout_seconds)
            if len(bars) < 21:
                add(book.value, "BLOCK", f"{spec.key}: only {len(bars)} daily bars")
            elif quote is None:
                add(book.value, "WARN",
                    f"{spec.key}: contract/history ready ({len(bars)} bars); no off-session BBO")
            else:
                add(book.value, "OK",
                    f"{spec.key}: conId {quote.con_id}, BBO {quote.bid:g}/{quote.ask:g}, "
                    f"{len(bars)} bars")
        except Exception as exc:  # noqa: BLE001
            add(book.value, "BLOCK", f"{spec.key}: {str(exc)[:220]}")
    if own_client:
        await client.close()
    return MultiAssetPreflightReport(tuple(results))
