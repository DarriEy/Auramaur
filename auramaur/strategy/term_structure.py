"""Term-structure pillar — price a deadline ladder off ONE reading of the event.

Deadline families ("X by July 10?", "X by July 17?", … — 70+ active families,
some with 13 strikes spanning 1c-90c) are priced strike-by-strike by the
crowd, independently. But the strikes share one underlying question: WHEN, if
ever, does the event happen. This pillar reads the family once — rules plus
current evidence, one LLM call — into an event-time curve (P(event by each
listed deadline)), then prices EVERY strike off that curve and trades the
largest gaps. One call amortizes across the whole ladder, which is the direct
answer to the budget-throughput constraint that starves per-market readers
(the lens); and a curve fit is structurally immune to the "right story,
wrong strike" failure observed in the per-market reads.

The reading-edge thesis this operationalizes: an LLM can out-READ the crowd —
resolution rules, registers, deadlines, term structure — not out-forecast it.
The curve is a timeline judgment anchored on the family's own resolution
criteria, not a world-model prophecy.

Deterministic sanity comes free: within a family, P(by T1) <= P(by T2) for
T1 < T2. Violations are logged (entailment_arb owns trading them); the model
curve itself is isotonic-clamped so a noisy read can't emit an impossible
curve.

Data hygiene (why candidates come from live discovery, not the markets
table): stored end_date is NULL or wrong for many ladder strikes and
resolved markets freeze active=1, so families are built from the dated
discovery scan and the deadline is parsed from the QUESTION TEXT.

Rides the standard rails: signals + trades attribution, full RiskManager
gate, ExecutionGateway placement, resolution-tracker settlement. PAPER-FORCED
(new directional cell under the enforced graduation ladder). One position per
market bot-wide (settlement attribution is market-scoped for same-token
stacks; see agent_trader for the same rule and rationale).
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
from datetime import datetime, timedelta, timezone

import structlog

from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.exchange.models import Confidence, Market, OrderSide, Signal
from auramaur.strategy.classifier import blocked_category_hit, ensure_category

log = structlog.get_logger()

_MONTHS = {m.lower(): i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}

# "... by July 10, 2026?" / "... by July 10?" / "... by end of 2026?"
_BY_DATE_RE = re.compile(
    r"\bby\s+([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?\s*\??\s*$",
    re.IGNORECASE,
)

CURVE_PROMPT = """\
You are pricing a prediction-market DEADLINE LADDER: the same event with \
multiple "by <date>" strikes. Read the resolution criteria and research the \
current state of the event (you may use WebSearch/WebFetch), then give YOUR \
probability that the event happens by EACH deadline.

Anchor on the RESOLUTION CRITERIA, not the headline: what exactly must occur, \
per the rules text, for YES. Probabilities must be non-decreasing with later \
deadlines. Base your read ONLY on the material below plus your own research.

Respond with STRICT JSON only, no prose, no code fences:
{{"thesis": "one sentence: the timeline mechanism the crowd misprices", \
"curve": [{{"market_id": "...", "prob": 0.0}}]}}
Include every strike listed below in "curve".

EVENT FAMILY: {family}

RESOLUTION CRITERIA (from the longest-dated strike):
{rules}

STRIKES (deadline | current market price = crowd's probability):
{strikes}
"""

_CURVES_TABLE = """
CREATE TABLE IF NOT EXISTS term_structure_curves (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    family TEXT NOT NULL,
    market_id TEXT NOT NULL,
    deadline TEXT DEFAULT '',
    model_prob REAL,
    market_prob REAL,
    thesis TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
)
"""


def parse_deadline(question: str) -> datetime | None:
    """Deadline from the question tail ('… by July 10, 2026?'). The stored
    end_date is unreliable for ladder strikes (NULL/wrong), so the question
    text is the source of truth."""
    m = _BY_DATE_RE.search(question or "")
    if not m:
        return None
    month = _MONTHS.get(m.group(1).lower())
    if month is None:
        return None
    day = int(m.group(2))
    year = int(m.group(3)) if m.group(3) else datetime.now(timezone.utc).year
    try:
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None


def family_key(question: str) -> str | None:
    """Normalized family identity: the question up to its ' by <date>' tail."""
    m = _BY_DATE_RE.search(question or "")
    if not m:
        return None
    return question[: m.start()].strip().lower()


def parse_curve(raw: str, strikes: list[Market]) -> tuple[str, dict[str, float]]:
    """(thesis, {market_id: prob}) from the model reply — tolerant of fences
    and prose, isotonic-clamped in deadline order so an impossible curve
    (P(by T1) > P(by T2)) can never be emitted. Returns ({}, {}) shape on
    garbage; a bad reply must never crash a cycle."""
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return "", {}
    try:
        payload = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return "", {}
    entries = payload.get("curve")
    if not isinstance(entries, list):
        return "", {}
    probs: dict[str, float] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        mid = str(e.get("market_id", "")).strip()
        try:
            p = float(e.get("prob"))
        except (TypeError, ValueError):
            continue
        if mid and 0.0 <= p <= 1.0:
            probs[mid] = p
    # Isotonic clamp in deadline order: running max.
    running = 0.0
    for mkt in strikes:  # strikes arrive deadline-sorted
        if mkt.id in probs:
            running = max(running, probs[mkt.id])
            probs[mkt.id] = running
    thesis = str(payload.get("thesis", "")).strip()
    return thesis, probs


class TermStructurePillar:
    """Deadline-ladder curve reader over Polymarket."""

    name = "term_structure"

    def __init__(self, db, settings, discovery, exchange, risk_manager,
                 pnl_tracker, calibration) -> None:
        self._db = db
        self._settings = settings
        self._discovery = discovery
        self._exchange = exchange
        self._risk = risk_manager
        self._pnl = pnl_tracker
        self._calibration = calibration
        self._gateway = ExecutionGateway(
            router=None, exchange=exchange, exchange_name="polymarket",
            settings=settings, db=db, pnl_tracker=pnl_tracker,
        )
        self._schema_ready = False

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        await self._db.execute(_CURVES_TABLE)
        await self._db.commit()
        self._schema_ready = True

    # ------------------------------------------------------------------
    # Cycle
    # ------------------------------------------------------------------

    async def run_once(self) -> int:
        cfg = self._settings.term_structure
        if not cfg.enabled:
            return 0
        await self._ensure_schema()
        families = await self._families(cfg)
        if not families:
            log.info("term_structure.no_families")
            return 0
        entered = 0
        calls = 0
        for fam, strikes in families:
            try:
                curve = await self._cached_curve(fam, cfg)
                if curve is None:
                    if calls >= cfg.families_per_cycle:
                        continue  # fresh reads capped; cached fams still trade
                    calls += 1
                    curve = await self._read_family(fam, strikes, cfg)
                if not curve:
                    continue
                thesis, probs = curve
                entered += await self._trade_curve(fam, strikes, thesis,
                                                   probs, cfg)
            except Exception as e:
                log.warning("term_structure.family_error", family=fam,
                            error=str(e))
        log.info("term_structure.cycle", families=len(families), reads=calls,
                 entered=entered)
        return entered

    # ------------------------------------------------------------------
    # Families — live discovery, question-text deadlines
    # ------------------------------------------------------------------

    async def _families(self, cfg) -> list[tuple[str, list[Market]]]:
        """Seed-and-search family assembly. A volume-ranked dated scan cannot
        see ladders — deep strikes are low-volume by construction, so the
        top-of-volume slice yields lone members (observed: 73 eligible, 10
        ladder members, ZERO complete families; the same wall long_horizon hit
        with its first scan). Any parseable 'by <date>' hit is treated as a
        SEED, and the family's siblings are fetched live via text search."""
        now = datetime.now(timezone.utc)
        emin = (now + timedelta(days=cfg.min_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        emax = (now + timedelta(days=cfg.max_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        raw: list[Market] = []
        try:
            for off in range(0, max(int(cfg.scan_limit), 1), 100):
                page = await self._discovery.get_markets(
                    limit=100, offset=off, order="volume",
                    end_date_min=emin, end_date_max=emax)
                if not page:
                    break
                raw.extend(page)
        except TypeError:
            raw = await self._discovery.get_markets(limit=cfg.scan_limit)

        groups: dict[str, dict[str, Market]] = {}
        seed_volume: dict[str, float] = {}
        for m in raw:
            if not self._eligible(m, cfg):
                continue
            fam = family_key(m.question)
            if fam is None or parse_deadline(m.question) is None:
                continue
            groups.setdefault(fam, {})[m.id] = m
            seed_volume[fam] = seed_volume.get(fam, 0.0) + m.volume

        # Complete incomplete families by live sibling search, highest-volume
        # seeds first, bounded to max_families searches per cycle.
        searcher = getattr(self._discovery, "search_markets", None)
        if searcher is not None:
            searched = 0
            for fam in sorted(groups, key=lambda f: -seed_volume[f]):
                if len(groups[fam]) >= cfg.min_strikes:
                    continue  # scan already delivered the ladder
                if searched >= cfg.max_families:
                    break
                searched += 1
                try:
                    siblings = await searcher(fam, limit=20)
                    for s in siblings or []:
                        if family_key(s.question) != fam:
                            continue
                        if parse_deadline(s.question) is None:
                            continue
                        if not self._eligible(s, cfg):
                            continue
                        groups[fam][s.id] = s
                except Exception as e:
                    log.debug("term_structure.sibling_search_failed",
                              family=fam, error=str(e))
                    continue

        out: list[tuple[str, list[Market]]] = []
        for fam, by_id in groups.items():
            strikes = list(by_id.values())
            if len(strikes) < cfg.min_strikes:
                continue
            strikes.sort(key=lambda m: parse_deadline(m.question))
            self._log_monotonicity(fam, strikes)
            out.append((fam, strikes))
        # Widest ladders first — the most opinion per read.
        out.sort(key=lambda fs: (len(fs[1]), sum(m.volume for m in fs[1])),
                 reverse=True)
        return out[: cfg.max_families]

    def _eligible(self, market: Market, cfg) -> bool:
        if not market.active:
            return False
        if (market.exchange or "polymarket") != "polymarket":
            return False
        if market.liquidity < cfg.min_liquidity:
            return False
        if not (0.02 <= market.outcome_yes_price <= 0.98):
            return False
        excluded = set(self._settings.risk.blocked_categories) | set(cfg.exclude_categories)
        if blocked_category_hit(excluded, market.question, market.description,
                                market.category):
            return False
        return True

    @staticmethod
    def _log_monotonicity(fam: str, strikes: list[Market]) -> None:
        """P(by T1) <= P(by T2) must hold; a violation is model-free signal
        for entailment_arb — here it is only surfaced, not traded."""
        prev = 0.0
        for m in strikes:
            if m.outcome_yes_price < prev - 0.03:  # tolerance for spread noise
                log.info("term_structure.monotonicity_violation",
                         family=fam, market_id=m.id,
                         price=m.outcome_yes_price, earlier_strike=prev)
            prev = max(prev, m.outcome_yes_price)

    # ------------------------------------------------------------------
    # Curve read — one LLM call per family, cached
    # ------------------------------------------------------------------

    async def _cached_curve(self, fam: str, cfg) -> tuple[str, dict[str, float]] | None:
        rows = await self._db.fetchall(
            """SELECT market_id, model_prob, thesis FROM term_structure_curves
               WHERE family = ? AND created_at > datetime('now', ?)""",
            (fam, f"-{float(cfg.curve_ttl_hours)} hours"),
        )
        if not rows:
            return None
        probs = {r["market_id"]: float(r["model_prob"]) for r in rows}
        return (rows[0]["thesis"] or "", probs)

    async def _read_family(self, fam: str, strikes: list[Market],
                           cfg) -> tuple[str, dict[str, float]] | None:
        rules = (strikes[-1].description or strikes[-1].question)[:1500]
        lines = []
        for m in strikes:
            d = parse_deadline(m.question)
            lines.append(f"- market_id={m.id} | by {d.strftime('%Y-%m-%d')} | "
                         f"price {m.outcome_yes_price:.2f}")
        prompt = CURVE_PROMPT.format(
            family=strikes[-1].question, rules=rules, strikes="\n".join(lines))
        raw = await self._call_model(prompt, cfg)
        thesis, probs = parse_curve(raw, strikes)
        if not probs:
            log.info("term_structure.unparseable_curve", family=fam)
            return None
        for m in strikes:
            if m.id in probs:
                d = parse_deadline(m.question)
                await self._db.execute(
                    """INSERT INTO term_structure_curves
                       (family, market_id, deadline, model_prob, market_prob, thesis)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (fam, m.id, d.strftime("%Y-%m-%d"), probs[m.id],
                     m.outcome_yes_price, thesis[:400]),
                )
        await self._db.commit()
        log.info("term_structure.curve_read", family=fam,
                 strikes=len(probs))
        return thesis, probs

    async def _call_model(self, prompt: str, cfg) -> str:
        from auramaur.nlp import call_budget
        from auramaur.nlp.errors import BudgetExhausted
        from auramaur.subprocess_security import analysis_subprocess_env

        budget = self._settings.nlp.daily_claude_call_budget
        if budget > 0:
            limit = call_budget.non_reserved_limit(self._settings)
            if call_budget.calls_today() >= limit:
                raise BudgetExhausted(
                    f"non-reserved Claude budget ({limit}/{budget}, paced) exhausted")
        # Neutral cwd: `claude -p` loads CLAUDE.md + project memory from its
        # working directory (see agent_trader / the context-leak note).
        proc = await asyncio.create_subprocess_exec(
            "claude", "-p", prompt,
            "--output-format", "text",
            "--model", cfg.model,
            "--effort", cfg.effort,
            "--allowedTools", "WebSearch,WebFetch",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=tempfile.gettempdir(),
            env=analysis_subprocess_env(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=cfg.llm_timeout_seconds)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError("curve read timed out")
        call_budget.record_call()
        if proc.returncode != 0:
            raise RuntimeError(f"curve read failed: {stderr.decode()[:200]}")
        return stdout.decode()

    # ------------------------------------------------------------------
    # Trading the curve — standard rails per strike
    # ------------------------------------------------------------------

    async def _trade_curve(self, fam: str, strikes: list[Market], thesis: str,
                           probs: dict[str, float], cfg) -> int:
        entered = 0
        gaps = []
        for m in strikes:
            if m.id not in probs:
                continue
            gap = abs(probs[m.id] - m.outcome_yes_price) * 100.0
            if gap >= cfg.min_edge_pts:
                gaps.append((gap, m))
        gaps.sort(reverse=True, key=lambda g: g[0])
        for gap, market in gaps:
            if entered >= cfg.max_entries_per_family:
                break
            # Claimed markets don't consume an entry slot — the cap applies
            # to entries actually made, largest gaps first.
            if await self._market_claimed(market.id):
                continue
            if await self._try_enter(market, probs[market.id], thesis, cfg):
                entered += 1
        return entered

    async def _market_claimed(self, market_id: str) -> bool:
        row = await self._db.fetchone(
            "SELECT 1 FROM trades WHERE market_id = ? LIMIT 1", (market_id,))
        if row is not None:
            return True
        row = await self._db.fetchone(
            "SELECT 1 FROM portfolio WHERE market_id = ? LIMIT 1", (market_id,))
        return row is not None

    async def _try_enter(self, market: Market, prob_yes: float, thesis: str,
                         cfg) -> bool:
        market_yes = market.outcome_yes_price
        side = OrderSide.BUY if prob_yes > market_yes else OrderSide.SELL
        signal = Signal(
            market_id=market.id,
            market_question=market.question,
            claude_prob=prob_yes,
            claude_confidence=Confidence.MEDIUM,
            market_prob=market_yes,
            edge=abs(prob_yes - market_yes) * 100.0,
            evidence_summary=thesis[:500],
            recommended_side=side,
            strategy_source="term_structure",
            mispricing_reason=(
                f"term-structure: {thesis[:250]}" if thesis else
                "term-structure: strike mispriced vs family event-time curve"),
        )
        await self._persist_signal(signal, market)

        decision = await self._risk.evaluate(signal, market)
        if not decision.approved or decision.position_size <= 0:
            log.info("term_structure.risk_rejected", market_id=market.id,
                     reason=decision.reason)
            return False
        size = min(decision.position_size, cfg.stake_usd)
        force_paper = cfg.paper or getattr(decision, "force_paper", False)
        res = await self._gateway.submit(TradeIntent(
            signal=signal, market=market, size_dollars=size,
            force_paper=force_paper))
        if res.status not in ("filled", "paper", "partial", "pending"):
            log.info("term_structure.order_rejected", market_id=market.id,
                     status=res.status, error=res.reason)
            return False
        await self._record_position(signal, market, res.order, res.result)
        log.info("term_structure.entered", market_id=market.id,
                 token=res.order.token.value, price=res.order.price,
                 size=res.order.size, model_prob=round(prob_yes, 2),
                 paper=res.result.is_paper)
        return True

    # ------------------------------------------------------------------
    # Bookkeeping — same rails as the other pillars
    # ------------------------------------------------------------------

    async def _persist_signal(self, signal: Signal, market: Market) -> None:
        await self._db.execute(
            """INSERT OR IGNORE INTO markets (id, exchange, condition_id, question,
               description, category, active, outcome_yes_price, outcome_no_price,
               volume, liquidity, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, datetime('now'))""",
            (market.id, market.exchange or "polymarket", market.condition_id,
             market.question, (market.description or "")[:500],
             ensure_category(market.question, market.description, market.category),
             market.outcome_yes_price, market.outcome_no_price,
             market.volume, market.liquidity),
        )
        await self._db.execute(
            """INSERT INTO signals (market_id, claude_prob, claude_confidence,
               market_prob, edge, evidence_summary, action, strategy_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'term_structure')""",
            (signal.market_id, signal.claude_prob, signal.claude_confidence.value,
             signal.market_prob, signal.edge, signal.evidence_summary,
             signal.recommended_side.value),
        )
        await self._db.commit()

    async def _record_position(self, signal: Signal, market: Market,
                               order, result) -> None:
        fill_size = result.filled_size if result.filled_size > 0 else order.size
        fill_price = result.filled_price if result.filled_price > 0 else order.price
        await self._db.execute(
            """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
               current_price, unrealized_pnl, category, token, token_id,
               is_paper, updated_at)
               VALUES (?, 'polymarket', 'BUY', ?, ?, ?, 0, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(market_id, is_paper, token) DO UPDATE SET
                   size = excluded.size,
                   avg_price = excluded.avg_price,
                   current_price = excluded.current_price,
                   updated_at = excluded.updated_at""",
            (order.market_id, fill_size, fill_price, fill_price,
             market.category or "", order.token.value, order.token_id,
             1 if result.is_paper else 0),
        )
        await self._db.commit()
        try:
            await self._calibration.record_prediction(
                order.market_id, signal.claude_prob, market.category or "")
        except Exception as e:
            log.debug("term_structure.calibration_error", error=str(e))
