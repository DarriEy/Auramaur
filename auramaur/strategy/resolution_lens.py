"""Resolution-language lens — trade the gap between headline and fine print.

The bot's best documented wins share one shape: the crowd prices the
HEADLINE event, but the resolution criteria resolve something subtly
different — "Will X ANNOUNCE Y?" (statement risk), "PERMANENT peace deal"
(permanence bars), literal model counting (LMArena ranks), compound
conditions and deadlines. An LLM reading 400 words of fine print
adversarially is exactly the asymmetry this bot has over casual flow.

Pipeline per cycle:
  1. Lexical triggers select candidate markets (announce/officially/
     permanent/exact-count/compound phrasing) — cheap, no LLM.
  2. Real-book guards (liquidity, spread, resolution window, blocked
     categories) — no price is believed on a dead book.
  3. The LENS PROMPT reads the criteria adversarially BOTH ways ("how does
     YES resolve without the headline event — and how does the event happen
     without YES resolving?") and returns a strict-criteria fair price, a
     gap_score, and the named mechanism. Verdicts cache permanently
     (criteria are static).
  4. Trade only when gap_score and |fair - market| clear config floors.
     Lens signals are BORN with mispricing_reason set ("behavioral: ..."),
     so the name-the-gap risk gate passes without a second audit call.

PAPER-FORCED by default; one shot per market; rides the standard rails as
strategy_source='resolution_lens', scored by the graduation ladder.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import structlog

from auramaur.strategy.classifier import ensure_category
from auramaur.exchange.models import (
    Confidence,
    Fill,
    Market,
    OrderSide,
    Signal,
)

log = structlog.get_logger()

# Lexical triggers — phrasings where fine print historically diverges from
# the headline. Deliberately broad; the LLM lens is the precision stage.
TRIGGER_PATTERNS = [
    r"\bannounce[sd]?\b", r"\bofficially\b", r"\bconfirm(s|ed)?\b",
    r"\bsay(s)?\b", r"\bstate(s|d)?\b", r"\bdeclar(e|es|ed)\b",
    r"\bpermanent(ly)?\b", r"\blasting\b", r"\bceasefire\b",
    r"\bexactly\b", r"\bbetween\s+\d", r"(?:^|\s)#\d+\b", r"\brank(ed)?\b",
    r"\band\b.*\bby\b.*\?\s*$",  # compound condition + deadline
]
_TRIGGERS = [re.compile(p, re.IGNORECASE) for p in TRIGGER_PATTERNS]


def has_lens_trigger(question: str) -> bool:
    q = question or ""
    return any(p.search(q) for p in _TRIGGERS)


LENS_PROMPT = """You are a resolution-criteria auditor for a prediction market. The crowd usually prices the HEADLINE; you price the FINE PRINT. Read adversarially.

Question: "{question}"
Resolution criteria: {description}
Current market price (YES): {market_prob:.2f}

Answer BOTH directions:
1. How can YES resolve WITHOUT the headline event most people imagine actually happening?
2. How can the headline event happen WITHOUT YES resolving (qualifying language, deadlines, required sources, permanence/officiality bars, literal counting rules, compound conditions)?

Then estimate the fair P(YES) under a STRICT reading of the written criteria — not the vibe. If the criteria and the headline reading genuinely coincide, say so with gap_score 0.

Respond with ONLY this JSON:
{{"fair_prob": <float 0-1>, "gap_score": <float 0-1, how much the strict reading diverges from the headline reading>, "mechanism": "<one concrete sentence naming the specific fine-print mechanism, or 'none'>"}}"""


class ResolutionLensPillar:
    def __init__(self, db, settings, discovery, exchange, risk_manager,
                 pnl_tracker, calibration, analyzer) -> None:
        self._db = db
        self._settings = settings
        self._discovery = discovery
        self._exchange = exchange
        self._risk = risk_manager
        self._pnl = pnl_tracker
        self._calibration = calibration
        self._analyzer = analyzer

    # -- candidate filtering ---------------------------------------------

    def _eligible(self, m: Market) -> bool:
        cfg = self._settings.resolution_lens
        # While paper-forced the strategy can't reach the venue, so the
        # live-trading eligibility guards (hold horizon, fillable liquidity)
        # don't apply and the strict values starve accrual. Use the loosened
        # paper thresholds; they revert to strict automatically if paper is
        # flipped to graduate the cell to live.
        min_liq = cfg.paper_min_liquidity if cfg.paper else cfg.min_liquidity
        min_hours = (cfg.paper_min_hours_to_resolution if cfg.paper
                     else cfg.min_hours_to_resolution)
        if not m.active or not (0.0 < m.outcome_yes_price < 1.0):
            return False
        if (m.exchange or "polymarket") != "polymarket":
            return False
        if m.liquidity < min_liq:
            return False
        if m.spread and m.spread * 100.0 > cfg.max_spread_pct:
            return False
        if (m.category or "") in set(self._settings.risk.blocked_categories):
            return False
        if len(m.description or "") < cfg.min_description_chars:
            return False  # no fine print to read
        if m.end_date is None:
            return False
        end = m.end_date if m.end_date.tzinfo else m.end_date.replace(tzinfo=timezone.utc)
        hours = (end - datetime.now(timezone.utc)).total_seconds() / 3600.0
        return min_hours <= hours <= cfg.max_days_to_resolution * 24.0

    async def _verdict(self, m: Market):
        """Cached lens verdict (criteria are static — cache forever)."""
        row = await self._db.fetchone(
            "SELECT fair_prob, gap_score, mechanism FROM lens_verdicts WHERE market_id = ?",
            (m.id,),
        )
        if row is not None:
            return float(row["fair_prob"]), float(row["gap_score"]), row["mechanism"]
        if self._analyzer is None:
            return None
        fair, score, mech = m.outcome_yes_price, 0.0, "none"
        try:
            raw = await self._analyzer._call_llm(LENS_PROMPT.format(
                question=m.question,
                description=(m.description or "")[:800],
                market_prob=m.outcome_yes_price,
            ))
            parsed = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
            fair = max(0.01, min(0.99, float(parsed.get("fair_prob", fair))))
            score = max(0.0, min(1.0, float(parsed.get("gap_score", 0.0))))
            mech = str(parsed.get("mechanism", "none"))[:400] or "none"
        except Exception as e:
            log.warning("lens.llm_parse_error", market_id=m.id, error=str(e))
        await self._db.execute(
            """INSERT OR REPLACE INTO lens_verdicts
               (market_id, fair_prob, gap_score, mechanism)
               VALUES (?, ?, ?, ?)""",
            (m.id, fair, score, mech),
        )
        await self._db.commit()
        return fair, score, mech

    async def _already_entered_or_held(self, market_id: str) -> bool:
        row = await self._db.fetchone(
            "SELECT 1 FROM signals WHERE market_id = ? AND strategy_source = 'resolution_lens' LIMIT 1",
            (market_id,),
        )
        if row is not None:
            return True
        row = await self._db.fetchone(
            "SELECT 1 FROM portfolio WHERE market_id = ? LIMIT 1", (market_id,),
        )
        return row is not None

    # -- candidate selection ----------------------------------------------

    async def _lexical_candidates(self) -> list[Market]:
        """Select candidates by fine-print LEXICAL shape across the FULL markets
        table, not the top-N-by-volume scan.

        resolution_lens mines the long tail — niche markets with tricky
        resolution criteria — but the per-cycle scan is the top markets by
        VOLUME, the mainstream where fine print rarely diverges, re-scanned
        every cycle. So it never reached its own opportunity set and emitted
        zero signals. The markets table caches question + description (the fine
        print) + price + liquidity + end_date for ~all markets — everything
        ``_eligible`` and the LLM lens need — so we filter the whole universe
        cheaply by trigger keywords here and let the precise ``has_lens_trigger``
        regex + ``_eligible`` refine. Per-cycle LLM work stays bounded by
        ``max_llm_calls_per_cycle``; the lens_verdicts cache advances coverage
        across cycles.
        """
        rows = await self._db.fetchall(
            """SELECT id, question, description, category, end_date,
                      outcome_yes_price, outcome_no_price, liquidity,
                      clob_token_yes, clob_token_no
               FROM markets
               WHERE active = 1 AND exchange = 'polymarket'
                 AND (question LIKE '%announce%' OR question LIKE '%official%'
                      OR question LIKE '%confirm%' OR question LIKE '%permanent%'
                      OR question LIKE '%exactly%' OR question LIKE '%ceasefire%'
                      OR question LIKE '%declare%' OR question LIKE '%ranked%'
                      OR question LIKE '%lasting%' OR question LIKE '%between %'
                      OR question LIKE '%state%' OR question LIKE '% #%')""",
        )
        out: list[Market] = []
        for r in rows or []:
            if not has_lens_trigger(r["question"]):
                continue
            end = None
            if r["end_date"]:
                try:
                    end = datetime.fromisoformat(str(r["end_date"]).replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass
            out.append(Market(
                id=r["id"], exchange="polymarket",
                question=r["question"] or "", description=r["description"] or "",
                category=r["category"] or "",
                outcome_yes_price=r["outcome_yes_price"] or 0.0,
                outcome_no_price=r["outcome_no_price"] or 0.0,
                liquidity=r["liquidity"] or 0.0, volume=r["liquidity"] or 0.0,
                clob_token_yes=r["clob_token_yes"] or "",
                clob_token_no=r["clob_token_no"] or "",
                end_date=end, active=True,
            ))
        return out

    # -- main cycle --------------------------------------------------------

    async def run_once(self) -> int:
        cfg = self._settings.resolution_lens
        if not cfg.enabled:
            return 0
        # Lexical pre-filter over the whole universe; fall back to the volume
        # scan only if the markets table is empty (cold start).
        markets = await self._lexical_candidates()
        if markets:
            log.info("lens.candidates", lexical=len(markets))
        else:
            markets = await self._discovery.get_markets(limit=cfg.scan_limit)
        calls = 0
        entered = 0
        for m in markets:
            if entered >= cfg.max_entries_per_cycle:
                break
            if not (self._eligible(m) and has_lens_trigger(m.question)):
                continue
            if await self._already_entered_or_held(m.id):
                continue
            cached = await self._db.fetchone(
                "SELECT 1 FROM lens_verdicts WHERE market_id = ?", (m.id,))
            if cached is None:
                if calls >= cfg.max_llm_calls_per_cycle:
                    continue
                calls += 1
            v = await self._verdict(m)
            if v is None:
                continue
            fair, score, mech = v
            edge = fair - m.outcome_yes_price
            if (score < cfg.min_gap_score or abs(edge) < cfg.min_edge
                    or mech.strip().lower() == "none"):
                continue
            try:
                if await self._enter(m, fair, score, mech, edge):
                    entered += 1
            except Exception as e:
                log.error("lens.entry_error", market_id=m.id, error=str(e))
        if entered:
            log.info("lens.cycle_done", entered=entered, llm_calls=calls)
        return entered

    # -- execution ----------------------------------------------------------

    async def _enter(self, m: Market, fair: float, score: float,
                     mech: str, edge: float) -> bool:
        cfg = self._settings.resolution_lens
        signal = Signal(
            market_id=m.id,
            market_question=m.question,
            claude_prob=fair,
            # A concrete, named fine-print mechanism with a high gap_score is
            # the one class of LLM divergence with a documented win record —
            # HIGH here both reflects that and clears the divergence filter
            # (which exists to block UNexplained mid-band disagreement).
            claude_confidence=Confidence.HIGH if score >= cfg.high_conf_gap_score
            else Confidence.MEDIUM,
            market_prob=m.outcome_yes_price,
            edge=abs(edge) * 100.0,
            evidence_summary=f"Resolution lens (gap {score:.2f}): {mech}",
            recommended_side=OrderSide.BUY if edge > 0 else OrderSide.SELL,
            strategy_source="resolution_lens",
            mispricing_reason=f"behavioral: {mech}",
        )
        decision = await self._risk.evaluate(signal, m)
        if not decision.approved or decision.position_size <= 0:
            log.info("lens.risk_rejected", market_id=m.id, reason=decision.reason)
            return False
        size = min(decision.position_size, cfg.stake_usd)
        is_live = (self._settings.is_live and not cfg.paper
                   and not getattr(decision, "force_paper", False))
        order = self._exchange.prepare_order(signal, m, size, is_live)
        if order is None:
            return False
        result = await self._exchange.place_order(order)
        if result.status not in ("filled", "paper", "partial", "pending"):
            log.warning("lens.order_rejected", market_id=m.id, status=result.status)
            return False
        await self._record_entry(signal, m, order, result)
        log.info("lens.entered", market_id=m.id, fair=fair, market=m.outcome_yes_price,
                 gap_score=score, paper=result.is_paper)
        return True

    async def _record_entry(self, signal: Signal, market: Market,
                            order, result) -> None:
        fill_size = result.filled_size if result.filled_size > 0 else order.size
        fill_price = result.filled_price if result.filled_price > 0 else order.price
        is_paper = bool(result.is_paper)
        if result.status in ("filled", "paper", "partial") and fill_size > 0:
            await self._pnl.record_fill(Fill(
                order_id=result.order_id, market_id=order.market_id,
                token_id=order.token_id, side=order.side, token=order.token,
                size=fill_size, price=fill_price, is_paper=is_paper,
            ))
        await self._db.execute(
            """INSERT OR IGNORE INTO markets (id, exchange, question, description,
               category, active, outcome_yes_price, outcome_no_price, volume,
               liquidity, last_updated)
               VALUES (?, 'polymarket', ?, ?, ?, 1, ?, ?, ?, ?, datetime('now'))""",
            (market.id, market.question, (market.description or "")[:500],
             ensure_category(market.question, market.description, market.category),
             market.outcome_yes_price,
             market.outcome_no_price, market.volume, market.liquidity),
        )
        await self._db.execute(
            """INSERT INTO signals (market_id, claude_prob, claude_confidence,
               market_prob, edge, evidence_summary, action, strategy_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'resolution_lens')""",
            (signal.market_id, signal.claude_prob, signal.claude_confidence.value,
             signal.market_prob, signal.edge, signal.evidence_summary,
             signal.recommended_side.value),
        )
        await self._db.execute(
            """INSERT INTO trades (market_id, timestamp, side, size, price,
               is_paper, order_id, status, strategy_source, exchange)
               VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, 'resolution_lens',
                       'polymarket')""",
            (order.market_id, order.side.value, fill_size, fill_price,
             1 if is_paper else 0, result.order_id,
             "filled" if result.status in ("filled", "paper") else result.status),
        )
        await self._db.execute(
            """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
               current_price, unrealized_pnl, category, token, token_id,
               is_paper, updated_at)
               VALUES (?, 'polymarket', 'BUY', ?, ?, ?, 0, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(market_id, is_paper, token) DO UPDATE SET
                   size = excluded.size, avg_price = excluded.avg_price,
                   current_price = excluded.current_price,
                   updated_at = excluded.updated_at""",
            (order.market_id, fill_size, fill_price, fill_price,
             market.category or "", order.token.value, order.token_id,
             1 if is_paper else 0),
        )
        await self._db.commit()
        try:
            await self._calibration.record_prediction(
                signal.market_id, signal.claude_prob, market.category or "")
        except Exception as e:
            log.debug("lens.calibration_error", error=str(e))
