"""Odd-lot tender harvester — the first equity pillar.

The edge: issuer tender offers often include ODD-LOT PRIORITY — holders of
fewer than 100 shares are accepted in full, exempt from proration. Buy 99
shares after the announcement, tender at the offer price, collect the
premium. The edge persists BECAUSE it cannot scale (funds can't split into
odd lots), which makes it a small-account specialty — and the work is
reading SEC filing fine print, exactly the asymmetry this bot has.

Pipeline per cycle (default every 6h):
  1. EDGAR full-text search for recent SC TO-I (+amendment) filings
     mentioning "odd lot" — cheap, free, no LLM.
  2. For each NEW accession: fetch the filing document, lexical pre-check,
     then the LLM extracts the structured terms ADVERSARIALLY (default:
     no odd-lot priority): priority y/n, fixed price or Dutch range,
     expiration, conditions. Verdict cached permanently in oddlot_filings.
  3. Confirmed opportunities ALERT the operator (Telegram/Discord) and, when
     IBKR is enabled, place a 99-share limit BUY via place_share_order —
     PAPER-FORCED by default, behind the three live gates like everything.
  4. TENDERING IS MANUAL (corporate-action submission isn't wired): the
     alert includes the expiration so the operator can submit the tender in
     TWS before the deadline. The settlement P&L lands via position sync.

Detection runs even with IBKR disabled — the opportunity record builds
while the account finishes onboarding. Standard rails on entry: markets
row (exchange='ibkr', category='ibkr_equity'), signals/trades
(strategy_source='oddlot_tender'), fills -> pnl_ledger, portfolio row.
"""

from __future__ import annotations

from auramaur.strategy.protocols import ExecutionMode

import json

import structlog

from auramaur.exchange.models import Fill, OrderSide

log = structlog.get_logger()


def _safe_float(value, default: float = 0.0) -> float:
    """Coerce an LLM-returned value to float, tolerating non-numeric strings
    (e.g. 'NAV', 'N/A', 'TBD' for NAV-linked or undetermined tenders). Returns
    *default* rather than raising, so one unparseable field can't discard a
    whole filing audit."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


ODDLOT_PROMPT = """You are auditing an SEC issuer tender-offer filing for the odd-lot arbitrage trade. Be adversarial: the default answer is that there is NO usable odd-lot priority.

Company: {company} ({ticker})
Form: {form}, filed {filed_at}

Filing text (truncated):
{text}

Extract the terms that matter for buying 99 shares and tendering them:
1. Does the offer give ODD-LOT HOLDERS (fewer than 100 shares) priority / exemption from proration? Quote-check: many filings mention "odd lots" only to say odd-lot tenders are NOT preferred, or require holding BEFORE a record date (which kills the trade for a new buyer — answer false in that case).
2. The price: fixed cash price, or a Dutch-auction range (give low and high).
3. Expiration date of the offer.
4. Conditions that could kill it (financing, minimum tender, withdrawal).

Respond with ONLY this JSON:
{{"odd_lot_priority": true|false, "requires_record_date_holding": true|false, "tender_price": <fixed price or Dutch LOW, USD>, "tender_price_high": <Dutch HIGH or same as tender_price>, "expiration": "YYYY-MM-DD or empty", "conditions": "<one sentence>", "confidence": 0.0-1.0}}"""


class OddLotTenderPillar:

    # Uniform Strategy contract (see strategy/protocols.py).
    name = "oddlot_tender"
    execution_mode = ExecutionMode.DIRECT_EQUITY
    def __init__(self, db, settings, edgar, analyzer, alerts=None,
                 equity_client=None, pnl_tracker=None) -> None:
        self._db = db
        self._settings = settings
        self._edgar = edgar
        self._analyzer = analyzer
        self._alerts = alerts
        self._equity = equity_client
        self._pnl = pnl_tracker

    # ------------------------------------------------------------------

    async def run_once(self) -> int:
        cfg = self._settings.oddlot_tender
        if not cfg.enabled:
            return 0
        filings = await self._edgar.recent_tender_filings(days=cfg.lookback_days)
        found = 0
        analyzed = 0
        for f in filings:
            row = await self._db.fetchone(
                "SELECT 1 FROM oddlot_filings WHERE accession = ?", (f.accession,))
            if row is not None:
                continue  # already audited — verdicts are permanent
            if analyzed >= cfg.max_filings_per_cycle:
                break
            analyzed += 1
            verdict = await self._audit_filing(f)
            if verdict is None:
                continue
            if (verdict["odd_lot_priority"]
                    and not verdict["requires_record_date_holding"]
                    and verdict["confidence"] >= cfg.llm_min_confidence):
                found += 1
                await self._on_opportunity(f, verdict)
        if analyzed:
            log.info("oddlot.cycle_done", analyzed=analyzed, opportunities=found)
        return found

    async def _audit_filing(self, f) -> dict | None:
        text = await self._edgar.fetch_document(f)
        verdict = {
            "odd_lot_priority": False, "requires_record_date_holding": False,
            "tender_price": 0.0, "tender_price_high": 0.0,
            "expiration": "", "conditions": "", "confidence": 0.0,
        }
        if text and "odd lot" in text.lower() and self._analyzer is not None:
            try:
                raw = await self._analyzer._call_llm(ODDLOT_PROMPT.format(
                    company=f.company, ticker=f.ticker or "?", form=f.form,
                    filed_at=f.filed_at, text=text[:40000],
                ))
                parsed = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
                # Coerce per-field with a tolerant float: the LLM sometimes
                # returns a non-numeric price for NAV-linked tenders (a closed-
                # end fund tendering "at NAV" has no fixed price). float('NAV')
                # used to raise inside verdict.update(), which evaluates all args
                # before applying — so ONE bad field discarded the entire read,
                # including a perfectly-parsed odd_lot_priority. A non-numeric
                # price -> 0.0, which the downstream logic rejects as "no usable
                # fixed premium" — the correct outcome, reached gracefully.
                verdict.update(
                    odd_lot_priority=bool(parsed.get("odd_lot_priority", False)),
                    requires_record_date_holding=bool(
                        parsed.get("requires_record_date_holding", False)),
                    tender_price=_safe_float(parsed.get("tender_price")),
                    tender_price_high=_safe_float(parsed.get("tender_price_high")),
                    expiration=str(parsed.get("expiration", ""))[:10],
                    conditions=str(parsed.get("conditions", ""))[:300],
                    confidence=_safe_float(parsed.get("confidence")),
                )
            except Exception as e:
                log.warning("oddlot.llm_parse_error", accession=f.accession,
                            error=str(e)[:120])
        await self._db.execute(
            """INSERT OR IGNORE INTO oddlot_filings
               (accession, cik, ticker, company, form, filed_at,
                odd_lot_priority, tender_price, tender_price_high, expiration,
                conditions, confidence, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'detected')""",
            (f.accession, f.cik, f.ticker or "", f.company, f.form, f.filed_at,
             1 if verdict["odd_lot_priority"] else 0, verdict["tender_price"],
             verdict["tender_price_high"], verdict["expiration"],
             verdict["conditions"], verdict["confidence"]),
        )
        await self._db.commit()
        return verdict

    # ------------------------------------------------------------------

    async def _on_opportunity(self, f, verdict: dict) -> None:
        cfg = self._settings.oddlot_tender
        ticker = f.ticker or "?"
        msg = (f"ODD-LOT TENDER: {f.company} [{ticker}] {f.form} filed {f.filed_at} — "
               f"price ${verdict['tender_price']:.2f}"
               + (f"-${verdict['tender_price_high']:.2f}"
                  if verdict["tender_price_high"] > verdict["tender_price"] else "")
               + f", expires {verdict['expiration'] or '?'}. "
               f"Conditions: {verdict['conditions'] or 'none noted'}. "
               f"TENDERING IS MANUAL — submit in TWS before expiration.")
        log.info("oddlot.opportunity", accession=f.accession, ticker=ticker,
                 price=verdict["tender_price"], expiration=verdict["expiration"])
        if self._alerts is not None:
            try:
                await self._alerts.send(msg, level="warning")
            except Exception as e:
                log.debug("oddlot.alert_error", error=str(e))

        if self._equity is None or not self._settings.ibkr.enabled or not ticker or ticker == "?":
            await self._set_status(f.accession, "alerted")
            return

        # Entry: 99 shares (or fewer if the per-position cap binds) at a limit
        # that preserves the minimum premium vs the LOW tender price (Dutch
        # offers fill at >= low, so low is the conservative payout).
        price = await self._equity.get_price(ticker)
        if not price or price <= 0:
            await self._set_status(f.accession, "alerted_no_price")
            return
        payout = verdict["tender_price"]
        premium_pct = (payout - price) / price * 100.0 if price else 0.0
        if premium_pct < cfg.min_premium_pct:
            log.info("oddlot.premium_too_thin", ticker=ticker,
                     price=price, payout=payout, premium_pct=round(premium_pct, 2))
            await self._set_status(f.accession, "premium_too_thin")
            return
        qty = min(99, int(cfg.max_position_usd // price))
        if qty < 1:
            await self._set_status(f.accession, "too_expensive")
            return
        dry_run = cfg.paper or not self._settings.is_live
        result = await self._equity.place_share_order(
            ticker, OrderSide.BUY, qty, limit_price=price, dry_run=dry_run)
        if result.status not in ("filled", "paper", "partial", "pending"):
            log.warning("oddlot.order_rejected", ticker=ticker,
                        status=result.status, error=result.error_message)
            await self._set_status(f.accession, "order_rejected")
            return
        await self._record_entry(ticker, f, qty, price, result)
        await self._set_status(f.accession, "entered")

    async def _set_status(self, accession: str, status: str) -> None:
        await self._db.execute(
            "UPDATE oddlot_filings SET status = ? WHERE accession = ?",
            (status, accession))
        await self._db.commit()

    # ------------------------------------------------------------------
    # Standard rails (ledger/attribution/graduation all read these)
    # ------------------------------------------------------------------

    async def _record_entry(self, ticker: str, f, qty: int, price: float,
                            result) -> None:
        fill_size = result.filled_size if result.filled_size > 0 else float(qty)
        fill_price = result.filled_price if result.filled_price > 0 else price
        is_paper = bool(result.is_paper)
        await self._db.execute(
            """INSERT OR IGNORE INTO markets (id, exchange, question, category,
               active, outcome_yes_price, outcome_no_price, last_updated)
               VALUES (?, 'ibkr', ?, 'ibkr_equity', 1, 0.5, 0.5, datetime('now'))""",
            (ticker, f"Odd-lot tender: {f.company} ({f.form} {f.filed_at})"),
        )
        await self._db.execute(
            """INSERT INTO signals (market_id, claude_prob, claude_confidence,
               market_prob, edge, evidence_summary, action, strategy_source)
               VALUES (?, 0.5, 'HIGH', 0.5, 0, ?, 'BUY', 'oddlot_tender')""",
            (ticker, f"odd-lot tender {f.accession}"),
        )
        await self._db.execute(
            """INSERT INTO trades (market_id, timestamp, side, size, price,
               is_paper, order_id, status, strategy_source, exchange)
               VALUES (?, datetime('now'), 'BUY', ?, ?, ?, ?, ?,
                       'oddlot_tender', 'ibkr')""",
            (ticker, fill_size, fill_price, 1 if is_paper else 0,
             result.order_id,
             "filled" if result.status in ("filled", "paper") else result.status),
        )
        if self._pnl is not None and result.status in ("filled", "paper", "partial"):
            await self._pnl.record_fill(Fill(
                order_id=result.order_id, market_id=ticker, side=OrderSide.BUY,
                size=fill_size, price=fill_price, is_paper=is_paper,
            ))
        await self._db.execute(
            """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
               current_price, unrealized_pnl, category, token, is_paper, updated_at)
               VALUES (?, 'ibkr', 'BUY', ?, ?, ?, 0, 'ibkr_equity', 'YES', ?,
                       datetime('now'))
               ON CONFLICT(market_id, is_paper, token) DO UPDATE SET
                   size = excluded.size, avg_price = excluded.avg_price,
                   current_price = excluded.current_price,
                   updated_at = excluded.updated_at""",
            (ticker, fill_size, fill_price, fill_price, 1 if is_paper else 0),
        )
        await self._db.commit()
        log.info("oddlot.entered", ticker=ticker, qty=qty, price=fill_price,
                 paper=is_paper)
