"""Readiness checks for all six IBKR local paper books."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from types import SimpleNamespace

import structlog

from auramaur.exchange.ibkr_instruments import BY_BOOK, IBKRBook
from auramaur.exchange.ibkr_market_data import IBKRReadOnlyMarketData
from auramaur.exchange.ibkr_registry import record_validation
from auramaur.risk.ibkr_evidence import (evaluate_ibkr_daily_evidence,
                                         evaluate_ibkr_evidence)

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


async def preflight(settings, db, *, client=None, timeout_seconds: float | None = None,
                    books: tuple[IBKRBook, ...] | None = None):
    cfg = settings.ibkr
    if timeout_seconds is None:
        timeout_seconds = cfg.multiasset_preflight_timeout_seconds
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
                    contract = (await client.resolve(spec)
                                if hasattr(client, "resolve") else None)
                    market_open = (await client.is_market_open(spec)
                                   if hasattr(client, "is_market_open") else True)
                    quote = await asyncio.wait_for(client.get_quote(spec), timeout_seconds)
                    bars = await asyncio.wait_for(client.get_daily_bars(spec), timeout_seconds)
                if contract is None:
                    contract = SimpleNamespace(
                        conId=int(getattr(quote, "con_id", 0) or 0),
                        exchange=spec.exchange, currency=spec.currency,
                        multiplier=spec.multiplier)
                if len(bars) < 21:
                    await record_validation(db, spec, contract,
                                            quote_source=getattr(quote, "source", "none"),
                                            has_history=False, error="insufficient history")
                    return f"{spec.key}: only {len(bars)} daily bars", None, None
                if not market_open:
                    source = getattr(quote, "source", "none") if quote else "none"
                    await record_validation(db, spec, contract, quote_source=source,
                                            has_history=True)
                    return None, source, f"{spec.key}: venue closed; contract/history ready"
                if quote is None:
                    await record_validation(db, spec, contract, quote_source="none",
                                            has_history=True, error="no executable BBO")
                    return f"{spec.key}: no executable BBO", None, None
                age = time.time() - float(quote.timestamp)
                if age < 0 or age > cfg.multiasset_max_quote_age_seconds:
                    await record_validation(db, spec, contract,
                                            quote_source=getattr(quote, "source", "none"),
                                            has_history=True, error="stale BBO")
                    return f"{spec.key}: stale BBO ({age:.0f}s old)", None, None
                source = getattr(quote, "source", "ibkr_unknown")
                if source != "ibkr_live":
                    await record_validation(db, spec, contract, quote_source=source,
                                            has_history=True)
                    return f"{spec.key}: non-executable {source} quote", source, None
                await record_validation(db, spec, contract, quote_source=source,
                                        has_history=True)
                return None, source, None
            except Exception as exc:  # noqa: BLE001
                # A wait_for deadline cancels the pending IB request, which the
                # gateway logs as "Error 162 ... query cancelled" — same
                # transient class as pacing, so it gets the same backoff.
                timed_out = isinstance(exc, TimeoutError)
                if (pacing_error(exc) or timed_out) and attempt + 1 < attempts:
                    delay = cfg.multiasset_preflight_retry_seconds * (2 ** attempt)
                    log.warning("ibkr_multiasset.preflight_transient_retry", key=spec.key,
                                attempt=attempt + 1, delay_seconds=delay,
                                timed_out=timed_out, error=str(exc)[:160])
                    await asyncio.sleep(delay)
                    continue
                if timed_out:
                    prefix = f"no gateway response within {timeout_seconds:.0f}s"
                elif pacing_error(exc):
                    prefix = "pacing exhausted"
                else:
                    prefix = "probe failed"
                detail = str(exc)[:140] or exc.__class__.__name__
                await record_validation(
                    db, spec, SimpleNamespace(conId=0, exchange=spec.exchange,
                                              currency=spec.currency,
                                              multiplier=spec.multiplier),
                    quote_source="none", has_history=False, error=f"{prefix}: {detail}")
                return f"{spec.key}: {prefix}: {detail}", None, None
        return f"{spec.key}: pacing retry exhausted", None, None  # pragma: no cover
    if not getattr(client, "readonly", False) or hasattr(client, "place_order"):
        add("isolation", "BLOCK", "market-data client is not structurally read-only")
    else:
        add("isolation", "OK", "read-only client exposes no broker order method")

    required = {"ibkr_paper_positions", "ibkr_paper_fills",
                "ibkr_paper_ledger", "ibkr_paper_round_trips",
                "ibkr_paper_state", "ibkr_contract_registry"}
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
        disabled = set(cfg.multiasset_disabled_instruments)
        universe = tuple(spec for spec in BY_BOOK[book] if spec.key not in disabled)
        skipped = [spec.key for spec in BY_BOOK[book] if spec.key in disabled]
        if skipped:
            add(f"{book.value}:coverage", "WARN",
                "entitlement-gated: " + ", ".join(skipped))
        if not universe:
            add(book.value, "BLOCK", "no enabled instruments")
            continue
        log.info("ibkr_multiasset.preflight_book_start", book=book.value,
                 instruments=len(universe))
        failures = []
        closed = []
        sources = set()
        probes = await asyncio.gather(*(probe(spec) for spec in universe))
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
                f"all contracts/history ready; {len(closed)}/{len(universe)} "
                f"venues currently closed")
        else:
            add(book.value, "OK", f"all {len(universe)} enabled instruments ready; "
                f"sources={','.join(sorted(sources))}")
        log.info("ibkr_multiasset.preflight_book_complete", book=book.value,
                 passed=len(universe) - len(failures), failed=len(failures))
        registry = await db.fetchall(
            "SELECT status, COUNT(*) AS n FROM ibkr_contract_registry "
            "WHERE book=? GROUP BY status ORDER BY status", (book.value,))
        counts = {row["status"]: int(row["n"]) for row in registry}
        add(f"{book.value}:registry",
            "OK" if set(counts) <= {"eligible"} else "WARN",
            ", ".join(f"{status}={count}" for status, count in counts.items()))

        # PRIMARY contract (pre-registered 2026-07-20): daily marked-to-market
        # returns. The round-trip count is a SECONDARY cost-realism check —
        # slow-turnover books physically cannot produce 200 trips in 180 days,
        # and holding brackets must never be shortened to manufacture them.
        marks = await db.fetchall(
            "SELECT equity_usd FROM ibkr_paper_daily_marks "
            "WHERE book = ? ORDER BY mark_date", (book.value,))
        equities = [float(row["equity_usd"]) for row in marks]
        daily = [b - a for a, b in zip(equities, equities[1:])]
        mark_age = await db.fetchone(
            "SELECT CAST(julianday('now') - julianday(MIN(mark_date)) AS INTEGER) "
            "AS elapsed FROM ibkr_paper_daily_marks WHERE book = ?", (book.value,))
        elapsed_days = max(0, int(mark_age["elapsed"] or 0))
        evidence = evaluate_ibkr_daily_evidence(
            daily, elapsed_days=elapsed_days, budget_usd=book_cfg.budget_usd)

        observations = await db.fetchall(
            "SELECT net_pnl_usd FROM ibkr_paper_round_trips "
            "WHERE book = ? ORDER BY closed_at, id", (book.value,))
        pnls = [float(row["net_pnl_usd"]) for row in observations]
        trips = evaluate_ibkr_evidence(
            pnls, elapsed_days=elapsed_days, budget_usd=book_cfg.budget_usd,
            min_observations=30)
        gains = sum(pnl for pnl in pnls if pnl > 0)
        losses = abs(sum(pnl for pnl in pnls if pnl < 0))
        profit_factor = gains / losses if losses else (float("inf") if gains else 0.0)
        ready = evidence.ready and trips.ready
        status = "graduated" if ready else "not graduated"
        reasons = "; ".join(evidence.reasons + trips.reasons) or "contract passed"
        add(f"{book.value}:edge", "OK" if ready else "WARN",
            f"{status}: {evidence.observations} daily marks over "
            f"{evidence.elapsed_days} days (mean lower bound "
            f"${evidence.mean_lower_95_usd:.2f}/day), {len(pnls)} cost-adjusted "
            f"round trips, ${trips.net_pnl_usd:.2f} trip P&L, "
            f"profit factor {profit_factor:.2f}; {reasons}")
    if own_client:
        await client.close()
    return MultiAssetPreflightReport(tuple(results))
