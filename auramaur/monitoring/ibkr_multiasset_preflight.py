"""Readiness checks for all six IBKR local paper books."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time

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
        failures = []
        sources = set()
        for spec in BY_BOOK[book]:
            try:
                quote = await asyncio.wait_for(client.get_quote(spec), timeout_seconds)
                bars = await asyncio.wait_for(client.get_daily_bars(spec), timeout_seconds)
                if len(bars) < 21:
                    failures.append(f"{spec.key}: only {len(bars)} daily bars")
                elif quote is None:
                    failures.append(f"{spec.key}: no executable BBO")
                else:
                    age = time.time() - float(quote.timestamp)
                    if age < 0 or age > cfg.multiasset_max_quote_age_seconds:
                        failures.append(f"{spec.key}: stale BBO ({age:.0f}s old)")
                    source = getattr(quote, "source", "ibkr")
                    if source != "ibkr":
                        failures.append(f"{spec.key}: non-executable {source} quote")
                    sources.add(source)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{spec.key}: {str(exc)[:160]}")
        if failures:
            preview = "; ".join(failures[:3])
            if len(failures) > 3:
                preview += f"; +{len(failures) - 3} more"
            add(book.value, "BLOCK", preview)
        else:
            add(book.value, "OK", f"all {len(BY_BOOK[book])} instruments ready; "
                f"sources={','.join(sorted(sources))}")

        edge = await db.fetchone(
            """SELECT COUNT(*) AS n, COALESCE(SUM(pnl_usd), 0) AS pnl,
                      COALESCE(SUM(CASE WHEN pnl_usd > 0 THEN pnl_usd ELSE 0 END), 0) AS gains,
                      ABS(COALESCE(SUM(CASE WHEN pnl_usd < 0 THEN pnl_usd ELSE 0 END), 0)) AS losses
                 FROM ibkr_paper_ledger WHERE book = ? AND kind = 'trade'""",
            (book.value,))
        n, pnl = int(edge["n"] or 0), float(edge["pnl"] or 0)
        gains, losses = float(edge["gains"] or 0), float(edge["losses"] or 0)
        profit_factor = gains / losses if losses else (float("inf") if gains else 0.0)
        proven = n >= 30 and pnl > 0 and profit_factor > 1.1
        add(f"{book.value}:edge", "OK" if proven else "WARN",
            f"forward paper evidence: {n} exits, ${pnl:.2f} trade P&L, "
            f"profit factor {profit_factor:.2f}")
    if own_client:
        await client.close()
    return MultiAssetPreflightReport(tuple(results))
