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

from auramaur.strategy.protocols import ExecutionMode

import json
import re
from datetime import datetime, timezone

import structlog

from auramaur.strategy.classifier import blocked_category_hit, ensure_category
from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.exchange.models import (
    Confidence,
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
Today's date is {today}. Interpret EVERY date in the criteria relative to today, including 2-digit years (e.g. '26 = 2026). A resolution date that is today, imminent, or already past resolves on NEAR-TERM reality (current forecasts/observations) — it is NOT a distant-future climatological base-rate bet. Never invent a "distant year, revert to base rate" mechanism for a near-term or already-passed market; that is a misread of the current year, not a fine-print edge.

Answer BOTH directions:
1. How can YES resolve WITHOUT the headline event most people imagine actually happening?
2. How can the headline event happen WITHOUT YES resolving (qualifying language, deadlines, required sources, permanence/officiality bars, literal counting rules, compound conditions)?

Then estimate the fair P(YES) under a STRICT reading of the written criteria — not the vibe. If the criteria and the headline reading genuinely coincide, say so with gap_score 0.

Respond with ONLY this JSON:
{{"fair_prob": <float 0-1>, "gap_score": <float 0-1, how much the strict reading diverges from the headline reading>, "mechanism": "<one concrete sentence naming the specific fine-print mechanism, or 'none'>"}}"""


# Phase 2: adversarial verification — a SECOND, skeptical pass that tries to
# REFUTE the claimed mechanism, defaulting to refuted. Kills hallucinated /
# over-read fine print before any (paper or live) capital is committed.
VERIFY_PROMPT = """A resolution-criteria auditor claims a prediction market is mispriced because of a fine-print mechanism. Your job is to ADVERSARIALLY CHECK that claim — default to REFUTED unless the mechanism is clearly real and decisive under the LITERAL written criteria.

Question: "{question}"
Resolution criteria: {description}
Market price (YES): {market_prob:.2f}
Claimed strict-criteria fair P(YES): {fair:.2f}
Claimed mechanism: "{mechanism}"
Today's date is {today}. Interpret dates (incl. 2-digit years like '26 = 2026) relative to today.

Check: does the cited clause ACTUALLY qualify/disqualify as claimed, or is the auditor over-reading or inventing a rule the criteria don't state? If you can't point to specific criteria text that supports the mechanism, it is REFUTED. In particular, REFUTE any mechanism that treats a today/imminent/already-past resolution date as a "distant future" base-rate bet — that is a current-year misread, not a fine-print mechanism.

Respond with ONLY this JSON:
{{"verdict": "confirmed" | "refuted", "confidence": <float 0-1>, "why": "<one sentence citing the criteria>"}}"""


# Phase 3: evidence-grounded comprehension — the synergy stage. The model reads
# CURRENT evidence AGAINST the strict criteria. This is deliberately NOT a
# forecast ("will the event happen?", where the crowd beats the LLM); it is a
# reading task ("do the literal criteria resolve YES given THIS evidence by the
# deadline?", where reading-a-rule-against-facts is the LLM's edge). Catches the
# case where a fine-print mechanism is real but current reality has already moved
# the true probability — so the lens doesn't trade a stale mechanic against a
# market that is correctly priced for the present.
GROUND_PROMPT = """You are grounding a prediction market's STRICT resolution reading in CURRENT EVIDENCE. Do NOT forecast whether the headline event will happen — assess whether the LITERAL written criteria resolve YES given the evidence and the deadline.

Question: "{question}"
Resolution criteria: {description}
Deadline / resolution window: {deadline}
Today's date is {today} (interpret 2-digit years like '26 = 2026 relative to today).
Identified fine-print mechanism: "{mechanism}"
Strict-criteria fair P(YES) before evidence: {fair:.2f}

Current evidence (most recent / relevant first):
{evidence}

Read the evidence against the CRITERIA, not the headline. Where does current reality stand relative to the specific written bar (officiality, permanence, exact source, deadline, compound conditions)? Has the bar already been met, is it clearly on track, or is it unmet/blocked? If the evidence shows the strict bar is unlikely to be met as written, P(YES) should be LOW even if the headline event seems likely — and vice versa.

Respond with ONLY this JSON:
{{"grounded_prob": <float 0-1, P(YES) under the strict criteria given the evidence>, "confidence": <float 0-1, how much the evidence actually informs this>, "why": "<one sentence on what the evidence shows about the strict bar>"}}"""


class ResolutionLensPillar:

    # Uniform Strategy contract (see strategy/protocols.py).
    name = "resolution_lens"
    execution_mode = ExecutionMode.GATEWAY_SINGLE
    def __init__(self, db, settings, discovery, exchange, risk_manager,
                 pnl_tracker, calibration, analyzer, aggregator=None,
                 exchange_name: str = "polymarket",
                 source_tag: str = "resolution_lens") -> None:
        self._db = db
        self._settings = settings
        self._discovery = discovery
        self._exchange = exchange
        self._risk = risk_manager
        self._pnl = pnl_tracker
        self._calibration = calibration
        # The venue this instance scans. Default Polymarket (the proven lens);
        # a second instance can bind Kalshi for the paper measurement spike.
        self._exchange_name = exchange_name
        # Attribution tag — the Kalshi spike uses a DISTINCT source so it gets
        # its own graduation cells and can't dilute the proven Poly lens record.
        self._source_tag = source_tag
        # Shared execution tail; no router (passive entry at the observed price).
        self._gateway = ExecutionGateway(
            router=None, exchange=exchange, exchange_name=exchange_name,
            settings=settings, db=db, pnl_tracker=pnl_tracker,
        )
        self._analyzer = analyzer
        # Phase 3: the evidence aggregator (same one the ensemble uses). Optional
        # so the pillar still constructs without it (grounding then no-ops and the
        # lens falls back to the criteria-strict fair).
        self._aggregator = aggregator
        # Per-market verdict-call failure counts (in-memory; resets on restart).
        # Errors are retried, not cached — but after 3 strikes the market gets
        # the permanent neutral verdict so it can't eat the budget forever.
        self._verdict_failures: dict[str, int] = {}

    # -- candidate filtering ---------------------------------------------

    def _eligible(self, m: Market) -> bool:
        cfg = self._settings.resolution_lens
        # While paper-forced the strategy can't reach the venue, so the
        # live-trading eligibility guards (hold horizon, fillable liquidity)
        # don't apply and the strict values starve accrual. Use the loosened
        # paper thresholds; they revert to strict automatically if paper is
        # flipped to graduate the cell to live.
        min_liq = cfg.paper_min_liquidity if cfg.paper else cfg.min_liquidity
        # Kalshi's book is thinner than Polymarket's — its own (lower) floor so
        # the spike isn't starved of candidates by the Poly-tuned threshold.
        if self._exchange_name == "kalshi":
            min_liq = min(min_liq, cfg.kalshi_min_liquidity)
        min_hours = (cfg.paper_min_hours_to_resolution if cfg.paper
                     else cfg.min_hours_to_resolution)
        if not m.active or not (0.0 < m.outcome_yes_price < 1.0):
            return False
        if (m.exchange or self._exchange_name) != self._exchange_name:
            return False
        if m.liquidity < min_liq:
            return False
        if m.spread and m.spread * 100.0 > cfg.max_spread_pct:
            return False
        # Classify-before-block like the gateway: trust the stored label OR a
        # fresh classification, so a mislabeled blocked market is filtered here
        # instead of only at the gateway (saves an LLM read).
        if blocked_category_hit(self._settings.risk.blocked_categories,
                                m.question, m.description, m.category):
            return False
        if len(m.description or "") < cfg.min_description_chars:
            return False  # no fine print to read
        if m.end_date is None:
            return False
        end = m.end_date if m.end_date.tzinfo else m.end_date.replace(tzinfo=timezone.utc)
        hours = (end - datetime.now(timezone.utc)).total_seconds() / 3600.0
        return min_hours <= hours <= cfg.max_days_to_resolution * 24.0

    def _criteria_text(self, desc: str) -> str:
        """Full resolution criteria, capped — keeping the TAIL where the
        decisive qualifier/source/edge-case clauses live (the old [:800] cut
        them off). Head+tail when over the cap, so context survives too."""
        cap = self._settings.resolution_lens.criteria_char_cap
        desc = desc or ""
        if len(desc) <= cap:
            return desc
        head = cap // 3
        return desc[:head] + "\n…[criteria trimmed]…\n" + desc[-(cap - head):]

    async def _ensure_schema(self) -> None:
        """Idempotent: add columns on pre-existing DBs (verified for Phase 2;
        grounded_fair/grounded_at for Phase 3 — the latter is TTL'd, not a
        permanent cache, since evidence is fresh while criteria are static)."""
        for ddl in (
            "ALTER TABLE lens_verdicts ADD COLUMN verified INTEGER NOT NULL DEFAULT -1",
            "ALTER TABLE lens_verdicts ADD COLUMN grounded_fair REAL",
            "ALTER TABLE lens_verdicts ADD COLUMN grounded_at TEXT",
        ):
            try:
                await self._db.execute(ddl)
                await self._db.commit()
            except Exception:
                pass  # column already exists

    async def _verdict(self, m: Market):
        """Cached lens verdict -> (fair, gap_score, mechanism, verified). The
        criteria reading is static (cache it); `verified` is -1 until the
        adversarial check runs at trade-decision time."""
        row = await self._db.fetchone(
            "SELECT fair_prob, gap_score, mechanism, verified FROM lens_verdicts WHERE market_id = ?",
            (m.id,),
        )
        if row is not None:
            return (float(row["fair_prob"]), float(row["gap_score"]),
                    row["mechanism"], int(row["verified"]))
        if self._analyzer is None:
            return None
        # pin_claude: the fine-print read IS the edge — never let budget
        # routing silently hand it to a different model (that's what went
        # dark 2026-06-25 → 07-02: Gemini verdicts never cleared the floors).
        try:
            raw = await self._analyzer._call_llm(LENS_PROMPT.format(
                question=m.question,
                description=self._criteria_text(m.description),  # Phase 1: full criteria
                market_prob=m.outcome_yes_price,
                today=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            ), pin_claude=True)
            parsed = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
            fair = max(0.01, min(0.99, float(parsed.get("fair_prob", m.outcome_yes_price))))
            score = max(0.0, min(1.0, float(parsed.get("gap_score", 0.0))))
            mech = str(parsed.get("mechanism", "none"))[:400] or "none"
        except Exception as e:
            # Budget exhaustion is GLOBAL, not this market's fault — no strike,
            # no cache write, and no Gemini fallback either: the lens's calls
            # are pinned because Gemini verdicts measurably never clear the
            # entry floors, and a cached Gemini verdict would permanently
            # blind Claude to the market (the 06-25 lobotomy, with memory).
            # Retry when the budget window resets.
            from auramaur.nlp.errors import BudgetExhausted
            if isinstance(e, BudgetExhausted):
                log.debug("lens.budget_exhausted", market_id=m.id)
                return None
            # Do NOT cache other failures either: the old neutral-row write
            # made every LLM/parse error a PERMANENT no-gap verdict (the cache
            # never expires). Retry next cycle, but give up after a few
            # attempts so one confusing market can't eat the per-cycle budget
            # forever.
            self._verdict_failures[m.id] = self._verdict_failures.get(m.id, 0) + 1
            if self._verdict_failures[m.id] < 3:
                log.warning("lens.llm_parse_error", market_id=m.id, error=str(e))
                return None
            log.warning("lens.verdict_gave_up", market_id=m.id, error=str(e))
            fair, score, mech = m.outcome_yes_price, 0.0, "none"
        await self._db.execute(
            """INSERT OR REPLACE INTO lens_verdicts
               (market_id, fair_prob, gap_score, mechanism, verified)
               VALUES (?, ?, ?, ?, -1)""",
            (m.id, fair, score, mech),
        )
        await self._db.commit()
        return fair, score, mech, -1

    async def _verify_mechanism(self, m: Market, fair: float, mechanism: str) -> bool:
        """Phase 2: adversarial check of the named mechanism. Returns True only
        when the second pass CONFIRMS it at >= verify_min_confidence. Persists
        the verdict (0/1) so it isn't re-checked. A parsed refutation fails
        closed and persists — a hallucinated mechanism must not slip through.
        An infra/parse ERROR is not a refutation: skip this cycle without
        persisting, so a budget blip can't permanently kill a real gap."""
        cfg = self._settings.resolution_lens
        try:
            raw = await self._analyzer._call_llm(VERIFY_PROMPT.format(
                question=m.question,
                description=self._criteria_text(m.description),
                market_prob=m.outcome_yes_price, fair=fair, mechanism=mechanism,
                today=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            ), pin_claude=True)
            parsed = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
            confirmed = (str(parsed.get("verdict", "refuted")).lower() == "confirmed"
                         and float(parsed.get("confidence", 0.0)) >= cfg.verify_min_confidence)
        except Exception as e:
            log.warning("lens.verify_parse_error", market_id=m.id, error=str(e))
            return False  # verified stays -1; retried next cycle
        await self._db.execute(
            "UPDATE lens_verdicts SET verified = ? WHERE market_id = ?",
            (1 if confirmed else 0, m.id))
        await self._db.commit()
        return confirmed

    async def _gather_evidence(self, m: Market) -> list:
        """Fetch + rank current evidence for one market via the shared aggregator
        (the same pipeline the ensemble uses). Returns [] on any failure so
        grounding degrades gracefully to the criteria-strict fair."""
        if self._aggregator is None:
            return []
        cfg = self._settings.resolution_lens
        try:
            from auramaur.nlp.query_decomposer import extract_search_queries
            from auramaur.nlp.evidence_ranker import rank_evidence
            queries = extract_search_queries(
                m.question, m.description or "", m.category or "")
            seen: set[str] = set()
            items: list = []
            per_q = max(1, self._settings.nlp.evidence_per_source)
            for q in (queries or [m.question])[:3]:
                for it in await self._aggregator.gather(
                        q, limit_per_source=per_q, category=m.category or None):
                    if it.id not in seen:
                        seen.add(it.id)
                        items.append(it)
            nlp = self._settings.nlp
            ranked = rank_evidence(
                m.question, items, top_n=cfg.phase3_max_evidence,
                backend=nlp.relevance_backend, model_name=nlp.embedding_model)
            return ranked or items[:cfg.phase3_max_evidence]
        except Exception as e:
            log.warning("lens.evidence_error", market_id=m.id, error=str(e))
            return []

    async def _ground(self, m: Market, fair: float, mech: str) -> tuple[float, float] | None:
        """Phase 3: read CURRENT evidence against the strict criteria → a grounded
        fair P(YES) + confidence. Persists (grounded_fair, grounded_at) with a TTL
        re-check. Returns (grounded_fair, confidence), or None when evidence/LLM
        is unavailable (caller falls back to the criteria-strict fair)."""
        cfg = self._settings.resolution_lens
        evidence = await self._gather_evidence(m)
        if not evidence:
            return None
        ev_lines = []
        for it in evidence[:cfg.phase3_max_evidence]:
            when = it.published_at.strftime("%Y-%m-%d") if it.published_at else "?"
            snippet = (it.content or it.title or "")[:200]
            ev_lines.append(f"- [{when}] {it.source}: {it.title} — {snippet}")
        deadline = (m.end_date.strftime("%Y-%m-%d") if m.end_date else "unspecified")
        try:
            raw = await self._analyzer._call_llm(GROUND_PROMPT.format(
                question=m.question,
                description=self._criteria_text(m.description),
                deadline=deadline, mechanism=mech, fair=fair,
                evidence="\n".join(ev_lines),
                today=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            ), pin_claude=True)
            parsed = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
            g = max(0.01, min(0.99, float(parsed.get("grounded_prob", fair))))
            conf = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
        except Exception as e:
            log.warning("lens.ground_parse_error", market_id=m.id, error=str(e))
            return None
        await self._db.execute(
            "UPDATE lens_verdicts SET grounded_fair = ?, grounded_at = datetime('now') WHERE market_id = ?",
            (g, m.id))
        await self._db.commit()
        log.info("lens.grounded", market_id=m.id, criteria_fair=round(fair, 3),
                 grounded_fair=round(g, 3), confidence=round(conf, 2),
                 evidence_n=len(evidence))
        return g, conf

    async def _grounding_fresh(self, market_id: str) -> float | None:
        """Return a still-fresh grounded_fair (within the TTL), else None."""
        cfg = self._settings.resolution_lens
        row = await self._db.fetchone(
            """SELECT grounded_fair FROM lens_verdicts
               WHERE market_id = ? AND grounded_fair IS NOT NULL
                 AND grounded_at IS NOT NULL
                 AND (julianday('now') - julianday(grounded_at)) * 24.0 < ?""",
            (market_id, cfg.phase3_ttl_hours))
        return float(row["grounded_fair"]) if row else None

    async def _already_entered_or_held(self, market_id: str) -> bool:
        # Per-instance tag: the Kalshi spike (resolution_lens_kalshi) must
        # check ITS OWN signals, not the Poly lens's.
        row = await self._db.fetchone(
            "SELECT 1 FROM signals WHERE market_id = ? AND strategy_source = ? LIMIT 1",
            (market_id, self._source_tag),
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
               WHERE active = 1 AND exchange = ?
                 -- Only future-dated markets: the table is ~87% stale rows
                 -- (resolved markets keep active=1 with a past end_date), and the
                 -- lens can't trade a resolved market. Without this the scan loads
                 -- tens of thousands of dead rows every cycle just to drop them in
                 -- _eligible. NULL end_date is kept (undated markets still eligible).
                 AND (end_date IS NULL
                      OR end_date >= strftime('%Y-%m-%dT%H:%M:%SZ','now'))
                 AND (question LIKE '%announce%' OR question LIKE '%official%'
                      OR question LIKE '%confirm%' OR question LIKE '%permanent%'
                      OR question LIKE '%exactly%' OR question LIKE '%ceasefire%'
                      OR question LIKE '%declare%' OR question LIKE '%ranked%'
                      OR question LIKE '%lasting%' OR question LIKE '%between %'
                      OR question LIKE '%state%' OR question LIKE '% #%')""",
            (self._exchange_name,),
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
                id=r["id"], exchange=self._exchange_name,
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

    async def _prioritize_known_gaps(self, markets: list[Market], cfg) -> list[Market]:
        """Order the scan so the small per-cycle budget earns the most. Two tiers
        of priority, highest first:

        1. Candidates that already carry a qualifying cached verdict, whose cell
           trades LIVE (real money) — the proven edge.
        2. Same, but on a PAPER-forced cell — unproven exploration.
        Then fresh discovery with whatever budget remains.

        Entering a gap costs up to three LLM calls (verdict -> verify -> ground)
        against a small budget, so two things starve real money:
          - Spending the budget computing NEW verdicts every cycle, so already-
            found gaps never advance to verify/enter (the cell went silent
            2026-06-24 while live SELL gaps sat unacted on the book).
          - Letting a high-gap-but-PAPER cell (unproven/negative — e.g. politics
            gaps scoring ~0.9) crowd out the proven LIVE weather entries (often
            scoring lower) that actually earn. gap_score is NOT edge quality.
        Graduation cells are cached, so the per-category live check is cheap.
        Stable sort preserves order within each tier.
        """
        if not markets:
            return markets
        rows = await self._db.fetchall(
            "SELECT market_id, gap_score FROM lens_verdicts WHERE gap_score >= ?",
            (float(cfg.min_gap_score),))
        known = {r["market_id"]: (r["gap_score"] or 0.0) for r in (rows or [])}
        if not known:
            return markets
        # Which categories among the known gaps currently trade live for this
        # cell? Only those put real money to work; everything else is paper
        # exploration and must not pre-empt it.
        live_cats: set[str] = set()
        grad = getattr(self._risk, "graduation", None)
        if grad is not None and not cfg.paper:
            for cat in {m.category for m in markets if m.id in known}:
                try:
                    cell = await grad.decide(self._source_tag, cat or "")
                    if not getattr(cell, "force_paper", True):
                        live_cats.add(cat)
                except Exception:
                    pass

        def _rank(m: Market):
            g = known.get(m.id)
            if g is None:
                return (0, 0.0)                       # fresh discovery -> last
            tier = 2 if m.category in live_cats else 1  # live gaps over paper gaps
            return (tier, g)

        markets.sort(key=_rank, reverse=True)
        log.info("lens.prioritized_known_gaps", known_gaps=len(known),
                 live_gap_cells=len(live_cats), scanned=len(markets))
        return markets

    async def run_once(self) -> int:
        cfg = self._settings.resolution_lens
        if not cfg.enabled:
            return 0
        await self._ensure_schema()
        # Lexical pre-filter over the whole universe; fall back to the volume
        # scan only if the markets table is empty (cold start).
        markets = await self._lexical_candidates()
        if markets:
            log.info("lens.candidates", lexical=len(markets))
        else:
            markets = await self._discovery.get_markets(limit=cfg.scan_limit)
        markets = await self._prioritize_known_gaps(markets, cfg)
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
            fair, score, mech, verified = v
            edge = fair - m.outcome_yes_price
            if (score < cfg.min_gap_score or abs(edge) < cfg.min_edge
                    or mech.strip().lower() == "none"):
                continue
            # Phase 2: a real gap is on the table — adversarially verify the
            # mechanism before trading (skip hallucinated fine print). Bounded
            # by the same per-cycle LLM budget; result cached in `verified`.
            if cfg.verify_enabled and verified == -1:
                if calls >= cfg.max_llm_calls_per_cycle:
                    continue  # out of budget; verify next cycle
                calls += 1
                verified = 1 if await self._verify_mechanism(m, fair, mech) else 0
            if cfg.verify_enabled and verified != 1:
                log.info("lens.mechanism_refuted", market_id=m.id, mechanism=mech[:80])
                continue
            # Phase 3: evidence-grounded comprehension on a VERIFIED gap. Read
            # current evidence against the strict criteria (not a re-forecast). If
            # grounding has edge, trade the grounded fair; if it shows the gap is
            # already priced for current reality, the recomputed edge falls below
            # the floor and we skip — the synergy gate. Falls back to the
            # criteria-strict fair when evidence/LLM is unavailable or low-
            # confidence, so a fetch miss never blocks accrual.
            trade_fair = fair
            if cfg.phase3_grounding_enabled and self._aggregator is not None:
                g = await self._grounding_fresh(m.id)
                if g is None and calls < cfg.max_llm_calls_per_cycle:
                    calls += 1
                    res = await self._ground(m, fair, mech)
                    if res is not None and res[1] >= cfg.phase3_min_confidence:
                        g = res[0]
                if g is not None:
                    trade_fair = g
            edge = trade_fair - m.outcome_yes_price
            if abs(edge) < cfg.min_edge:
                log.info("lens.grounded_no_edge", market_id=m.id,
                         grounded_fair=round(trade_fair, 3), market=m.outcome_yes_price)
                continue
            try:
                if await self._enter(m, trade_fair, score, mech, edge):
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
        # Favorite-discipline floor (BUY side only). Edge audit (2026-06-24):
        # every lens×weather loss was a BUY of a narrow temperature bin entered
        # in the near-coin-flip band — favorite-longshot variance, not a real
        # fine-print read. Buying only YES sides already priced as favorites
        # cut the cell to 100% win in-sample. SELL longshot plays (overpriced
        # 'permanent'/'announce' YES) sit below this floor by construction and
        # are deliberately exempt. edge > 0 ⇒ BUY YES at ~outcome_yes_price.
        if (edge > 0 and cfg.min_entry_price > 0.0
                and m.outcome_yes_price < cfg.min_entry_price):
            log.info("lens.below_entry_floor", market_id=m.id,
                     yes_price=round(m.outcome_yes_price, 3),
                     floor=cfg.min_entry_price)
            return False
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
            strategy_source=self._source_tag,
            mispricing_reason=f"behavioral: {mech}",
        )
        decision = await self._risk.evaluate(signal, m)
        if not decision.approved or decision.position_size <= 0:
            log.info("lens.risk_rejected", market_id=m.id, reason=decision.reason)
            return False
        size = min(decision.position_size, cfg.stake_usd)
        force_paper = cfg.paper or getattr(decision, "force_paper", False)
        res = await self._gateway.submit(TradeIntent(
            signal=signal, market=m, size_dollars=size, force_paper=force_paper))
        if res.status not in ("filled", "paper", "partial", "pending"):
            log.warning("lens.order_rejected", market_id=m.id, status=res.status)
            return False
        await self._record_position(signal, m, res.order, res.result)
        log.info("lens.entered", market_id=m.id, fair=fair, market=m.outcome_yes_price,
                 gap_score=score, paper=res.result.is_paper)
        return True

    async def _record_position(self, signal: Signal, market: Market,
                               order, result) -> None:
        # Fill + trades-mirror owned by the ExecutionGateway; this keeps the
        # markets/signals persist, the portfolio row (resolution tracker settles
        # it), and the calibration prediction.
        fill_size = result.filled_size if result.filled_size > 0 else order.size
        fill_price = result.filled_price if result.filled_price > 0 else order.price
        is_paper = bool(result.is_paper)
        await self._db.execute(
            """INSERT OR IGNORE INTO markets (id, exchange, question, description,
               category, active, outcome_yes_price, outcome_no_price, volume,
               liquidity, last_updated)
               VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, datetime('now'))""",
            (market.id, self._exchange_name, market.question, (market.description or "")[:500],
             ensure_category(market.question, market.description, market.category),
             market.outcome_yes_price,
             market.outcome_no_price, market.volume, market.liquidity),
        )
        await self._db.execute(
            """INSERT INTO signals (market_id, claude_prob, claude_confidence,
               market_prob, edge, evidence_summary, action, strategy_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (signal.market_id, signal.claude_prob, signal.claude_confidence.value,
             signal.market_prob, signal.edge, signal.evidence_summary,
             signal.recommended_side.value, self._source_tag),
        )
        await self._db.execute(
            """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
               current_price, unrealized_pnl, category, token, token_id,
               is_paper, updated_at)
               VALUES (?, ?, 'BUY', ?, ?, ?, 0, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(market_id, is_paper, token) DO UPDATE SET
                   size = excluded.size, avg_price = excluded.avg_price,
                   current_price = excluded.current_price,
                   updated_at = excluded.updated_at""",
            (order.market_id, self._exchange_name, fill_size, fill_price, fill_price,
             market.category or "", order.token.value, order.token_id,
             1 if is_paper else 0),
        )
        await self._db.commit()
        try:
            await self._calibration.record_prediction(
                signal.market_id, signal.claude_prob, market.category or "")
        except Exception as e:
            log.debug("lens.calibration_error", error=str(e))
