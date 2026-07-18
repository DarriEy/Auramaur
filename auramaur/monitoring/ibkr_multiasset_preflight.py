"""Readiness checks for all six IBKR local paper books."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time

import structlog

from auramaur.exchange.ibkr_instruments import BY_BOOK, IBKRBook
from auramaur.exchange.ibkr_market_data import IBKRReadOnlyMarketData

log = structlog.get_logger()


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


async def preflight(settings, db, *, client=None, timeout_seconds: int = 30,
                    books: tuple[IBKRBook, ...] | None = None):
    cfg = settings.ibkr
    results: list[MultiAssetPreflightResult] = []

    def add(book, severity, detail):
        results.append(MultiAssetPreflightResult(book, severity, detail))

    own_client = client is None
    client = client or IBKRReadOnlyMarketData(
        settings, client_id=cfg.multiasset_preflight_client_id)
    semaphore = asyncio.Semaphore(cfg.multiasset_preflight_concurrency)

    def pacing_error(exc: Exception) -> bool:
        detail = str(exc).lower()
        return any(marker in detail for marker in ("pacing", "error 162", "error 420"))

    async def probe(spec):
        attempts = cfg.multiasset_preflight_pacing_retries + 1
        for attempt in range(attempts):
            try:
                async with semaphore:
                    market_open = (await client.is_market_open(spec)
                                   if hasattr(client, "is_market_open") else True)
                    quote = await asyncio.wait_for(client.get_quote(spec), timeout_seconds)
                    bars = await asyncio.wait_for(client.get_daily_bars(spec), timeout_seconds)
                if len(bars) < 21:
                    return f"{spec.key}: only {len(bars)} daily bars", None, None
                if not market_open:
                    source = getattr(quote, "source", "none") if quote else "none"
                    return None, source, f"{spec.key}: venue closed; contract/history ready"
                if quote is None:
                    return f"{spec.key}: no executable BBO", None, None
                age = time.time() - float(quote.timestamp)
                if age < 0 or age > cfg.multiasset_max_quote_age_seconds:
                    return f"{spec.key}: stale BBO ({age:.0f}s old)", None, None
                source = getattr(quote, "source", "ibkr")
                if source != "ibkr":
                    return f"{spec.key}: non-executable {source} quote", source, None
                return None, source, None
            except Exception as exc:  # noqa: BLE001
                if pacing_error(exc) and attempt + 1 < attempts:
                    delay = cfg.multiasset_preflight_retry_seconds * (2 ** attempt)
                    log.warning("ibkr_multiasset.preflight_pacing_retry", key=spec.key,
                                attempt=attempt + 1, delay_seconds=delay,
                                error=str(exc)[:160])
                    await asyncio.sleep(delay)
                    continue
                prefix = "pacing exhausted" if pacing_error(exc) else "probe failed"
                return f"{spec.key}: {prefix}: {str(exc)[:140]}", None, None
        return f"{spec.key}: pacing retry exhausted", None, None  # pragma: no cover
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

    for book in (books or tuple(IBKRBook)):
        book_cfg = cfg.multiasset_books[book.value]
        if not book_cfg.enabled:
            add(book.value, "WARN", "book disabled")
            continue
        log.info("ibkr_multiasset.preflight_book_start", book=book.value,
                 instruments=len(BY_BOOK[book]))
        failures = []
        closed = []
        sources = set()
        probes = await asyncio.gather(*(probe(spec) for spec in BY_BOOK[book]))
        for failure, source, closed_detail in probes:
            if failure:
                failures.append(failure)
            if source:
                sources.add(source)
            if closed_detail:
                closed.append(closed_detail)
        if failures:
            preview = "; ".join(failures[:3])
            if len(failures) > 3:
                preview += f"; +{len(failures) - 3} more"
            add(book.value, "BLOCK", preview)
        elif closed:
            add(book.value, "WARN",
                f"all contracts/history ready; {len(closed)}/{len(BY_BOOK[book])} "
                f"venues currently closed")
        else:
            add(book.value, "OK", f"all {len(BY_BOOK[book])} instruments ready; "
                f"sources={','.join(sorted(sources))}")
        log.info("ibkr_multiasset.preflight_book_complete", book=book.value,
                 passed=len(BY_BOOK[book]) - len(failures), failed=len(failures))

        edge = await db.fetchone(
            """SELECT SUM(CASE WHEN kind = 'trade' THEN 1 ELSE 0 END) AS n,
                      COALESCE(SUM(pnl_usd), 0) AS pnl,
                      COALESCE(SUM(CASE WHEN pnl_usd > 0 THEN pnl_usd ELSE 0 END), 0) AS gains,
                      ABS(COALESCE(SUM(CASE WHEN pnl_usd < 0 THEN pnl_usd ELSE 0 END), 0)) AS losses
                 FROM ibkr_paper_ledger WHERE book = ?""",
            (book.value,))
        n, pnl = int(edge["n"] or 0), float(edge["pnl"] or 0)
        gains, losses = float(edge["gains"] or 0), float(edge["losses"] or 0)
        profit_factor = gains / losses if losses else (float("inf") if gains else 0.0)
        proven = n >= 30 and pnl > 0 and profit_factor > 1.1
        add(f"{book.value}:edge", "OK" if proven else "WARN",
            f"forward paper evidence: {n} exits, ${pnl:.2f} net P&L, "
            f"profit factor {profit_factor:.2f}")
    if own_client:
        await client.close()
    return MultiAssetPreflightReport(tuple(results))
