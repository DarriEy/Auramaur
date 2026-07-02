"""Cross-venue semantic-equivalence arbitrage — Polymarket × Kalshi.

The arb scanner already pairs IDENTICAL questions across venues. This pillar
targets the harder, untapped case: markets that are *logically equivalent but
worded differently* across Poly and Kalshi (e.g. "Fed cuts rates in July?" vs
"Fed funds target below X after the July meeting?"). When two such markets price
the same real claim differently, the gap is a structural arb — no forecast.

The KILLER risk is resolution-criteria mismatch: two markets that look equivalent
but resolve on different sources / dates / edge-cases turn a "free" arb into a
paired loss. So equivalence is verified ADVERSARIALLY (default: not equivalent)
with the same skeptical machinery the resolution lens uses, and only a
high-confidence verdict trades. PAPER-FORCED and NOT graduation-exempt (unlike
same-question arb, a mismatch here is a real directional loss): the cell must
earn live like any other.

Orientation:
  * "same"     — A_YES resolves iff B_YES. Arb when the YES prices differ: buy
                 YES on the cheaper-YES venue + NO on the dearer; payout is always
                 $1, cost 1-|pA-pB|, profit |pA-pB|.
  * "inverted" — A_YES resolves iff B_NO. A and B are complementary, so fair
                 pA+pB == 1; arb when pA+pB < 1 (buy both YES) — profit 1-pA-pB.

Rides the standard rails as strategy_source='cross_venue_arb'.
"""

from __future__ import annotations

from auramaur.strategy.protocols import ExecutionMode

from datetime import datetime, timezone

import structlog

from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.exchange.models import Confidence, Market, OrderSide, Signal
from auramaur.strategy.arbitrage_scanner import _word_overlap_score

log = structlog.get_logger()


EQUIVALENCE_PROMPT = """You are auditing whether two prediction markets on DIFFERENT venues resolve the SAME underlying claim. Be adversarial: the default answer is "none". A false match turns a "risk-free" arbitrage into a paired loss, so only assert equivalence you are sure of.

Polymarket market: "{question_a}"
Polymarket resolution details: {description_a}

Kalshi market: "{question_b}"
Kalshi resolution details: {description_b}

Two markets are EQUIVALENT only if, in EVERY possible world, they resolve to a determined relationship under their OWN written criteria — accounting for thresholds, timing/settlement windows, qualifying language, the exact resolution source, and edge cases. Decide the orientation:
- "same": Polymarket-YES resolves YES if and only if Kalshi-YES resolves YES.
- "inverted": Polymarket-YES resolves YES if and only if Kalshi-YES resolves NO (they are complementary).
If you can construct ANY plausible scenario where the claimed relationship breaks (different dates, sources, thresholds, rounding, "official" vs actual, partial events), answer "none".

Respond with ONLY this JSON:
{{"orientation": "same" | "inverted" | "none", "confidence": 0.0-1.0, "counterexample": "<the scenario that best breaks equivalence, or 'none found'>"}}"""


class CrossVenueArbPillar:

    # Uniform Strategy contract (see strategy/protocols.py).
    name = "cross_venue_arb"
    execution_mode = ExecutionMode.GATEWAY_PAIRED
    def __init__(self, db, settings, discovery, exchange, risk_manager,
                 pnl_tracker, analyzer=None, kalshi_discovery=None,
                 exchanges=None) -> None:
        self._db = db
        self._settings = settings
        self._discovery = discovery            # Polymarket discovery
        self._exchange = exchange
        self._exchanges = dict(exchanges or {})
        if exchange is not None:
            self._exchanges.setdefault("polymarket", exchange)
        self._risk = risk_manager
        self._pnl = pnl_tracker
        self._analyzer = analyzer
        self._kalshi_discovery = kalshi_discovery
        # Shared execution tail; submit_paired takes the per-leg exchange, so the
        # gateway's own exchange is only a default (legs are cross-venue).
        self._gateway = ExecutionGateway(
            router=None, exchange=exchange, exchange_name="polymarket",
            settings=settings, db=db, pnl_tracker=pnl_tracker,
        )

    # -- schema ----------------------------------------------------------

    async def _ensure_schema(self) -> None:
        await self._db.execute(
            """CREATE TABLE IF NOT EXISTS cross_venue_verdicts (
                   poly_id TEXT NOT NULL,
                   kalshi_id TEXT NOT NULL,
                   orientation TEXT NOT NULL,
                   confidence REAL NOT NULL DEFAULT 0,
                   reasoning TEXT NOT NULL DEFAULT '',
                   traded_at TEXT,
                   partial_at TEXT,
                   last_error TEXT,
                   checked_at TEXT NOT NULL DEFAULT (datetime('now')),
                   PRIMARY KEY (poly_id, kalshi_id))""")
        for ddl in (
            "ALTER TABLE cross_venue_verdicts ADD COLUMN partial_at TEXT",
            "ALTER TABLE cross_venue_verdicts ADD COLUMN last_error TEXT",
        ):
            try:
                await self._db.execute(ddl)
                await self._db.commit()
            except Exception:
                pass
        await self._db.commit()

    # -- market quality --------------------------------------------------

    def _real_book(self, m: Market) -> bool:
        """Reject dead/placeholder books before believing any price."""
        cfg = self._settings.cross_venue_arb
        if not m.active or not (0.0 < m.outcome_yes_price < 1.0):
            return False
        venue = m.exchange or "polymarket"
        min_liq = cfg.kalshi_min_liquidity if venue == "kalshi" else cfg.min_liquidity
        if m.liquidity < min_liq:
            return False
        if m.spread and m.spread * 100.0 > cfg.max_spread_pct:
            return False
        if m.end_date is None:
            return False
        end = m.end_date if m.end_date.tzinfo else m.end_date.replace(tzinfo=timezone.utc)
        hours = (end - datetime.now(timezone.utc)).total_seconds() / 3600.0
        return hours >= cfg.min_hours_to_resolution

    # -- candidate pairing -----------------------------------------------

    async def _candidate_pairs(self) -> list[tuple[Market, Market]]:
        """Poly × Kalshi pairs above a word-overlap floor (cheap pre-filter
        before the LLM equivalence check). Both legs must be real books."""
        cfg = self._settings.cross_venue_arb
        if self._discovery is None or self._kalshi_discovery is None:
            return []
        try:
            poly = await self._discovery.get_markets(limit=cfg.scan_limit)
            kalshi = await self._kalshi_discovery.get_markets(limit=cfg.scan_limit)
        except Exception as e:
            log.debug("cross_venue.scan_error", error=str(e))
            return []
        poly = [m for m in poly if self._real_book(m)]
        kalshi = [m for m in kalshi if self._real_book(m)]
        pairs: list[tuple[Market, Market, float]] = []
        for a in poly:
            for b in kalshi:
                score = _word_overlap_score(a.question, b.question)
                if score >= cfg.min_word_overlap:
                    pairs.append((a, b, score))
        # Strongest lexical matches first; cap the LLM fan-out per cycle.
        pairs.sort(key=lambda t: t[2], reverse=True)
        return [(a, b) for a, b, _ in pairs[:cfg.max_llm_calls_per_cycle]]

    async def _verify_equivalence(self, a: Market, b: Market):
        """Cached adversarial equivalence verdict -> (orientation, confidence)."""
        row = await self._db.fetchone(
            "SELECT orientation, confidence FROM cross_venue_verdicts "
            "WHERE poly_id = ? AND kalshi_id = ?", (a.id, b.id))
        if row is not None:
            return row["orientation"], float(row["confidence"])
        if self._analyzer is None:
            return "none", 0.0
        orientation, confidence, reasoning = "none", 0.0, ""
        try:
            raw = await self._analyzer._call_llm(EQUIVALENCE_PROMPT.format(
                question_a=a.question, description_a=(a.description or "")[:600],
                question_b=b.question, description_b=(b.description or "")[:600]))
            # Shared robust parser (fences, prose tails, braces inside strings)
            # — one implementation for every LLM-JSON call site, not a clone.
            from auramaur.nlp.analyzer import _parse_claude_json
            parsed = _parse_claude_json(raw)
            orientation = str(parsed.get("orientation", "none"))
            confidence = float(parsed.get("confidence", 0.0))
            reasoning = str(parsed.get("counterexample", ""))[:400]
            if orientation not in ("same", "inverted"):
                orientation = "none"
        except Exception as e:
            log.warning("cross_venue.llm_parse_error", a=a.id, b=b.id, error=str(e))
        await self._db.execute(
            """INSERT OR REPLACE INTO cross_venue_verdicts
               (poly_id, kalshi_id, orientation, confidence, reasoning)
               VALUES (?, ?, ?, ?, ?)""",
            (a.id, b.id, orientation, confidence, reasoning))
        await self._db.commit()
        return orientation, confidence

    # -- arb math --------------------------------------------------------

    def _required_gap(self, a: Market, b: Market) -> float:
        """Min edge to clear BOTH legs' taker fees + a buffer (Poly maker ~0,
        Kalshi charges per leg) — a gap that doesn't clear fees is a loss."""
        from auramaur.strategy.signals import taker_fee_rate
        cfg = self._settings.cross_venue_arb
        fees = self._settings.arbitrage.exchange_fees
        pa, pb = a.outcome_yes_price, b.outcome_yes_price
        fee_a = taker_fee_rate(a.exchange or "polymarket", a.category or "", fees) * pa * (1.0 - pa)
        fee_b = taker_fee_rate(b.exchange or "kalshi", b.category or "", fees) * pb * (1.0 - pb)
        return fee_a + fee_b + cfg.gap_buffer

    def _arb(self, a: Market, b: Market, orientation: str):
        """Return (edge, side_a, side_b) for an executable arb, else None.

        'same': buy YES where YES is cheaper, NO where dearer (payout always $1).
        'inverted': A,B complementary — if pA+pB<1 buy both YES; if >1 buy both NO.
        """
        pa, pb = a.outcome_yes_price, b.outcome_yes_price
        if orientation == "same":
            edge = abs(pa - pb)
            if pa < pb:   # A_YES cheap -> buy A_YES, buy B_NO (sell dear B_YES)
                return edge, OrderSide.BUY, OrderSide.SELL
            return edge, OrderSide.SELL, OrderSide.BUY
        if orientation == "inverted":
            s = pa + pb
            if s < 1.0:   # both underpriced -> buy both YES
                return 1.0 - s, OrderSide.BUY, OrderSide.BUY
            return s - 1.0, OrderSide.SELL, OrderSide.SELL  # both overpriced -> both NO
        return None

    # -- execution -------------------------------------------------------

    def _leg_signal(self, m: Market, side: OrderSide, fair: float,
                    edge: float, why: str) -> Signal:
        return Signal(
            market_id=m.id, market_question=m.question,
            claude_prob=max(0.01, min(0.99, fair)),
            claude_confidence=Confidence.HIGH,   # verified structural arb, not a forecast
            market_prob=m.outcome_yes_price,
            edge=edge * 100.0,
            evidence_summary=f"Cross-venue arb ({why}).",
            recommended_side=side,
            strategy_source="cross_venue_arb",
        )

    async def _already_traded(self, a_id: str, b_id: str) -> bool:
        row = await self._db.fetchone(
            "SELECT traded_at, partial_at FROM cross_venue_verdicts "
            "WHERE poly_id = ? AND kalshi_id = ?",
            (a_id, b_id))
        return bool(row and (row["traded_at"] or row["partial_at"]))

    def _exchange_for(self, market: Market):
        return self._exchanges.get(market.exchange or "polymarket")

    async def _mark_partial(self, a_id: str, b_id: str, error: str) -> None:
        await self._db.execute(
            """UPDATE cross_venue_verdicts
               SET partial_at = datetime('now'), last_error = ?
               WHERE poly_id = ? AND kalshi_id = ?""",
            (error[:400], a_id, b_id))
        await self._db.commit()

    async def _enter_pair(self, a: Market, b: Market, edge: float,
                          side_a: OrderSide, side_b: OrderSide,
                          orientation: str, conf: float) -> bool:
        cfg = self._settings.cross_venue_arb
        exchange_a = self._exchange_for(a)
        exchange_b = self._exchange_for(b)
        if exchange_a is None or exchange_b is None:
            log.info(
                "cross_venue.missing_exchange",
                a=a.id,
                b=b.id,
                exchange_a=a.exchange,
                exchange_b=b.exchange,
            )
            return False

        why = f"{orientation}; gap {edge:.3f}"
        sig_a = self._leg_signal(a, side_a, fair=b.outcome_yes_price, edge=edge, why=why)
        sig_b = self._leg_signal(b, side_b, fair=a.outcome_yes_price, edge=edge, why=why)

        dec_a = await self._risk.evaluate(sig_a, a)
        dec_b = await self._risk.evaluate(sig_b, b)
        if not (dec_a.approved and dec_b.approved):
            log.info("cross_venue.risk_rejected", a=a.id, b=b.id,
                     reason_a=dec_a.reason, reason_b=dec_b.reason)
            return False
        size = min(dec_a.position_size, dec_b.position_size, cfg.stake_usd)
        if size <= 0:
            return False
        # Paper-force the pair together (cfg.paper or either leg's graduation
        # downgrade). The gateway owns both-or-nothing placement, fills, the
        # trades-mirror, and the live-pending leg-A unwind; the pillar keeps its
        # signals-per-leg + verdict bookkeeping below.
        force_paper = (cfg.paper or getattr(dec_a, "force_paper", False)
                       or getattr(dec_b, "force_paper", False))
        res_a, res_b = await self._gateway.submit_paired(
            TradeIntent(signal=sig_a, market=a, size_dollars=size, force_paper=force_paper),
            TradeIntent(signal=sig_b, market=b, size_dollars=size, force_paper=force_paper),
            exchange_a=exchange_a, exchange_name_a=(a.exchange or "polymarket"),
            exchange_b=exchange_b, exchange_name_b=(b.exchange or "kalshi"))

        ok_statuses = ("filled", "paper", "partial", "pending")
        if res_a.status not in ok_statuses:
            log.warning("cross_venue.leg_a_rejected", a=a.id, status=res_a.status)
            return False
        if res_b.status not in ok_statuses:
            log.error("cross_venue.leg_b_failed_single_leg", a=a.id, b=b.id,
                      status=res_b.status)
            await self._record_leg(a, res_a.order, res_a.result, why)
            await self._mark_partial(
                a.id, b.id, res_b.reason or f"leg_b_{res_b.status}")
            return False

        for market, res in ((a, res_a), (b, res_b)):
            await self._record_leg(market, res.order, res.result, why)
        await self._db.execute(
            "UPDATE cross_venue_verdicts SET traded_at = datetime('now') "
            "WHERE poly_id = ? AND kalshi_id = ?", (a.id, b.id))
        await self._db.commit()
        log.info("cross_venue.entered", a=a.id, b=b.id, orientation=orientation,
                 edge=round(edge, 3), confidence=conf, paper=res_a.result.is_paper)
        return True

    async def _record_leg(self, market: Market, order, result, why: str) -> None:
        # The fill (-> cost_basis -> pnl_ledger) and the trades-mirror are owned
        # by the ExecutionGateway; the pillar keeps the per-leg signals row (the
        # arb-leg convention stores the market price as claude_prob — there is no
        # LLM estimate for an arb leg).
        await self._db.execute(
            """INSERT INTO signals (market_id, claude_prob, claude_confidence,
               market_prob, edge, evidence_summary, action, strategy_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'cross_venue_arb')""",
            (market.id, market.outcome_yes_price, "HIGH", market.outcome_yes_price,
             0.0, f"Cross-venue arb leg ({why})",
             "BUY" if order.side == OrderSide.BUY else "SELL"))
        await self._db.commit()

    # -- main cycle ------------------------------------------------------

    async def run_once(self) -> int:
        cfg = self._settings.cross_venue_arb
        if not cfg.enabled:
            return 0
        await self._ensure_schema()
        pairs = await self._candidate_pairs()
        if not pairs:
            return 0
        entered = 0
        for a, b in pairs:
            if entered >= cfg.max_pairs_per_cycle:
                break
            if await self._already_traded(a.id, b.id):
                continue
            orientation, conf = await self._verify_equivalence(a, b)
            if orientation == "none" or conf < cfg.llm_min_confidence:
                continue
            res = self._arb(a, b, orientation)
            if res is None:
                continue
            edge, side_a, side_b = res
            if edge < self._required_gap(a, b):
                continue
            try:
                if await self._enter_pair(a, b, edge, side_a, side_b, orientation, conf):
                    entered += 1
            except Exception as e:
                log.error("cross_venue.entry_error", a=a.id, b=b.id, error=str(e))
        if entered:
            log.info("cross_venue.cycle_done", entered=entered)
        return entered
