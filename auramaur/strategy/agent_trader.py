"""Agent day-trader — the Hermes paradigm rebuilt as a first-class pillar.

The original experiment (agentmcp bridge, PR #209) ran a persistent reasoning
agent as an EXTERNAL process with unconstrained writes to its own ledger. It
answered a different question than intended: given write access to its own
scorecard, the agent fabricated it (see agentmcp/book.py S4). This pillar is
the honest rebuild — the same paradigm under test (does a memory-carrying
judgment agent beat the stateless strategy ensemble?) but riding the bot's own
rails: signals + trades attribution, the FULL RiskManager gate (all 15 checks,
never bypassed), ExecutionGateway placement, and settlement by the resolution
tracker. The agent cannot touch its own ledger; it can only emit opinions.

THE INTELLIGENCE-CAP A/B: the pillar runs the SAME mandate, prompt, candidate
set, and memory format across MULTIPLE models (``agent_trader.models``). Each
model is its own attribution cell — ``strategy_source = agent_trader_<alias>``
— so ``auramaur pnl --paper`` and the graduation ladder segment them
independently. If judgment quality is the binding constraint on directional
edge, the cells should separate by model tier; if they don't, the constraint
is elsewhere (information, market selection, execution) and more intelligence
won't buy edge.

Persistent judgment: every decision is stored in ``agent_trader_theses`` with
its one-sentence thesis; at prompt time the model sees its own alias's recent
CLOSED trades joined to realized P&L (the feedback loop) plus its open book.
Memory is per-alias — models never see each other's reasoning, or the cells
would correlate.

Paper-forced (config ``paper: true`` AND every cell is a new directional cell
under the enforced graduation ladder). Entries only in v1 — near-dated markets
(default <= 30d) keep the book turning over; exits are engine-level and
settlement-based, like bias_harvest/long_horizon.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone

import structlog

from auramaur.strategy.protocols import ExecutionMode

from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.exchange.models import Confidence, Market, OrderSide, Signal
from auramaur.experiments.strategies.agent_trader import (
    AgentTraderInputs,
    AgentTraderRules,
    assess_agent_trader,
    parse_decisions,
)
from auramaur.strategy.classifier import blocked_category_hit, ensure_category

log = structlog.get_logger()

# The arms may research the open web (same precedent as agent_analyzer) —
# real evidence-gathering, identical for every arm, no bot-opinion leakage.
_ALLOWED_TOOLS = "WebSearch,WebFetch"

MANDATE = """\
You are a prediction-market day trader running a small paper book. You are \
judged purely on realized P&L per trade. You see your own recent record below \
— learn from it: if a class of thesis keeps losing, stop making it.

Rules:
- You may use WebSearch/WebFetch to check current facts (prices, dates, \
announcements, standings) before deciding. Base decisions ONLY on the market \
data below plus your own research — you have no other context.
- Only propose a trade when you believe the market price is wrong by at least \
{min_edge_pts:.0f} points and you can say WHY in one concrete sentence (the \
mechanism: what the crowd is mispricing and what you know that it doesn't).
- prob_yes is YOUR probability that the market resolves YES, as a decimal.
- Proposing nothing is always acceptable and often correct.
- At most {max_entries} proposals.

Respond with STRICT JSON only, no prose, no code fences:
{{"decisions": [{{"market_id": "...", "prob_yes": 0.0, "thesis": "..."}}]}}

YOUR RECENT RECORD (closed trades, realized P&L):
{memory}

YOUR OPEN BOOK:
{open_book}

CANDIDATE MARKETS (current YES price is the crowd's probability):
{candidates}
"""

_DECLINES_TABLE = """
CREATE TABLE IF NOT EXISTS agent_trader_declines (
    model_alias TEXT NOT NULL,
    market_id TEXT NOT NULL,
    declined_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (model_alias, market_id)
)
"""

_COSTS_TABLE = """
CREATE TABLE IF NOT EXISTS agent_trader_costs (
    day TEXT NOT NULL,
    model_alias TEXT NOT NULL,
    calls INTEGER NOT NULL DEFAULT 0,
    usd REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (day, model_alias)
)
"""

_THESES_TABLE = """
CREATE TABLE IF NOT EXISTS agent_trader_theses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_alias TEXT NOT NULL,
    -- Venue-scoped strategy cell (agent_trader_<alias>[_<venue>]): each venue
    -- instance keeps its own book, so one cannot fill the other's slots.
    cell TEXT NOT NULL DEFAULT '',
    market_id TEXT NOT NULL,
    question TEXT DEFAULT '',
    token TEXT DEFAULT '',
    prob REAL,
    market_prob REAL,
    thesis TEXT DEFAULT '',
    stake REAL DEFAULT 0,
    entered INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now'))
)
"""


class AgentTraderPillar:
    """Multi-model LLM day-trader over near-dated Polymarket markets."""

    execution_mode = ExecutionMode.GATEWAY_SINGLE

    def __init__(self, db, settings, discovery, exchange, risk_manager,
                 pnl_tracker, calibration, *,
                 exchange_name: str = "polymarket",
                 cell_suffix: str = "") -> None:
        self._db = db
        self._settings = settings
        self._discovery = discovery
        self._exchange = exchange
        self._risk = risk_manager
        self._pnl = pnl_tracker
        self._calibration = calibration
        self._exchange_name = exchange_name
        # Venue-scoped attribution: a Kalshi instance earns its OWN graduation
        # cells (agent_trader_<alias>_kalshi) so its record can neither dilute
        # nor free-ride on the proven Polymarket cells — same rule as the
        # resolution_lens Kalshi spike.
        self._cell_suffix = cell_suffix
        self._gateway = ExecutionGateway(
            router=None, exchange=exchange, exchange_name=exchange_name,
            settings=settings, db=db, pnl_tracker=pnl_tracker,
        )
        self._schema_ready = False
        self._cycle_n = 0

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"agent_trader{self._cell_suffix}"

    def cell(self, alias: str) -> str:
        return f"agent_trader_{alias}{self._cell_suffix}"

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        await self._db.execute(_THESES_TABLE)
        await self._db.execute(_DECLINES_TABLE)
        await self._db.execute(_COSTS_TABLE)
        try:
            # Upgrade a pre-`entered` table in place (rows so far all entered).
            await self._db.execute(
                "ALTER TABLE agent_trader_theses ADD COLUMN entered INTEGER DEFAULT 1")
        except Exception:
            pass  # column already exists
        try:
            # Venue scope. The table was keyed by model_alias alone, so a second
            # venue instance counted the FIRST venue's open book and skipped
            # every arm as book_full — the Kalshi lane could never trade.
            await self._db.execute(
                "ALTER TABLE agent_trader_theses ADD COLUMN cell TEXT "
                "NOT NULL DEFAULT ''")
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise
        # 2026-08-04 (#353 phase 3): the backfill runs UNCONDITIONALLY, outside
        # the ALTER's except. When it shared the ALTER's try, a crash between
        # the two statements (each individually durable under autocommit) left
        # the column added but empty, and every LATER run's ALTER raised
        # duplicate-column into the except — so the backfill was skipped
        # forever and those rows stayed at cell=''. The UPDATE is idempotent
        # (WHERE cell = ''); existing pre-`cell` rows predate any second
        # venue, so they are Polymarket.
        await self._db.execute(
            "UPDATE agent_trader_theses SET cell = 'agent_trader_' || "
            "model_alias WHERE cell = ''")
        # 2026-08-04 (#353 phase 3): trailing commit() removed — dead under
        # the autocommit connection (no-op since eed51b8).
        self._schema_ready = True

    # ------------------------------------------------------------------
    # Cycle
    # ------------------------------------------------------------------

    async def run_once(self) -> int:
        cfg = self._settings.agent_trader
        if not cfg.enabled or not cfg.models:
            return 0
        await self._ensure_schema()
        candidates = await self._candidates(cfg)
        from auramaur.data_edge import record_market_snapshot
        await record_market_snapshot(self._db, self.name, candidates)
        if not candidates:
            log.info("agent_trader.no_candidates")
            return 0
        entered_total = 0
        # Rotate which arm runs first each cycle. Arms run sequentially and a
        # market is claimed by the first entrant, so a fixed order hands every
        # contested market to the same arm — the later arms' cells fill up
        # with leftovers (observed: one arm claim-blocked 3x in a night).
        start = self._cycle_n % len(cfg.models)
        self._cycle_n += 1
        rotated = list(cfg.models[start:]) + list(cfg.models[:start])
        for spec in rotated:
            try:
                entered_total += await self._run_model(spec, candidates, cfg)
            except Exception as e:
                # One model's failure (budget, timeout, bad output) must not
                # stop the other cells — they are independent experiment arms.
                log.warning("agent_trader.model_cycle_error",
                            alias=spec.alias, error=str(e))
        log.info("agent_trader.cycle", arms=len(cfg.models), entered=entered_total)
        return entered_total

    async def _run_model(self, spec, candidates: list[Market], cfg) -> int:
        alias = spec.alias
        open_rows = await self._open_theses(alias)
        if len(open_rows) >= cfg.max_open_per_model:
            log.info("agent_trader.book_full", alias=alias,
                      open=len(open_rows))
            return 0
        seen_rows = await self._db.fetchall(
            "SELECT DISTINCT market_id FROM agent_trader_theses WHERE model_alias = ?",
            (alias,),
        )
        declined_rows = await self._db.fetchall(
            """SELECT market_id FROM agent_trader_declines
               WHERE model_alias = ?
                 AND declined_at > datetime('now', ?)""",
            (alias, f"-{float(cfg.decline_ttl_hours)} hours"),
        )
        # Skip anything this alias already holds, has already opined on (a
        # prediction-only thesis is recorded once, not re-elicited), or
        # DECLINED within the TTL — without the decline memory, the same
        # unseen top-of-volume markets were re-offered every cycle and each
        # arm burned a call re-rejecting them. After the TTL the market is
        # offered again (prices move; a stale no is not a permanent no).
        seen = (
            {r["market_id"] for r in seen_rows}
            | {r["market_id"] for r in open_rows}
            | {r["market_id"] for r in declined_rows}
        )
        offered = [m for m in candidates if m.id not in seen]
        offered = offered[: cfg.markets_per_cycle]
        if not offered:
            return 0

        prompt = MANDATE.format(
            min_edge_pts=cfg.min_edge_pts,
            max_entries=cfg.max_entries_per_cycle,
            memory=await self._memory_block(alias, cfg.memory_events),
            open_book=self._open_block(open_rows),
            candidates=self._candidates_block(offered),
        )
        if getattr(spec, "provider", "claude") == "gemini":
            raw = await self._call_gemini(prompt, spec, cfg)
        else:
            raw = await self._call_model(prompt, spec.model, spec.effort, cfg)
        decisions = parse_decisions(raw, cfg.max_entries_per_cycle)
        from auramaur.data_edge import DataDelivery, record_delivery
        await record_delivery(self._db, DataDelivery(
            strategy=self.name, component="model_response", status="ok",
            provider=str(getattr(spec, "provider", "claude")),
            market_id=alias, source_at=datetime.now(timezone.utc),
            item_count=1, detail={"model": spec.model, "decisions": len(decisions)},
        ))

        # The model actually saw and passed on these — remember the pass for
        # decline_ttl_hours. Only after a SUCCESSFUL call: a budget/timeout
        # failure means the markets were never evaluated.
        decided_ids = {d["market_id"] for d in decisions}
        passed = [m.id for m in offered if m.id not in decided_ids]
        if passed:
            await self._db.executemany(
                """INSERT INTO agent_trader_declines (model_alias, market_id, declined_at)
                   VALUES (?, ?, datetime('now'))
                   ON CONFLICT(model_alias, market_id) DO UPDATE SET
                       declined_at = excluded.declined_at""",
                [(alias, mid) for mid in passed],
            )
            await self._db.commit()

        if not decisions:
            log.info("agent_trader.no_decisions", alias=alias,
                     model=spec.model)
            return 0

        by_id = {m.id: m for m in offered}
        entered = 0
        for d in decisions:
            market = by_id.get(d["market_id"])
            if market is None:
                continue  # hallucinated id — only offered markets count
            if await self._try_enter(alias, market, d, cfg):
                entered += 1
        return entered

    # ------------------------------------------------------------------
    # Candidates — near-dated, liquid, tradeable-priced
    # ------------------------------------------------------------------

    async def _candidates(self, cfg) -> list[Market]:
        now = datetime.now(timezone.utc)
        emin = (now + timedelta(days=cfg.min_days_to_resolution)).strftime("%Y-%m-%dT%H:%M:%SZ")
        emax = (now + timedelta(days=cfg.max_days_to_resolution)).strftime("%Y-%m-%dT%H:%M:%SZ")
        out: list[Market] = []
        try:
            for off in range(0, max(int(cfg.scan_limit), 1), 100):
                page = await self._discovery.get_markets(
                    limit=100, offset=off, order="volume",
                    end_date_min=emin, end_date_max=emax)
                if not page:
                    break
                out.extend(page)
        except TypeError:
            # Discovery without date-window kwargs (older client / test stub).
            out = await self._discovery.get_markets(limit=cfg.scan_limit)
        eligible = [m for m in out if self._eligible(m, cfg)]
        eligible.sort(key=lambda m: m.volume, reverse=True)
        return eligible

    def _eligible(self, market: Market, cfg) -> bool:
        if not market.active:
            return False
        if (market.exchange or self._exchange_name) != self._exchange_name:
            return False
        # Kalshi's bulk liquidity field underreports; the Poly-tuned floor
        # rejects nearly the whole venue (same wall the lens and long_horizon
        # hit). Use the venue's own, lower floor there.
        min_liq = cfg.min_liquidity
        if self._exchange_name == "kalshi":
            min_liq = min(min_liq, cfg.kalshi_min_liquidity)
        if market.liquidity < min_liq:
            return False
        p = market.outcome_yes_price
        if not (0.03 <= p <= 0.97):
            return False  # near-resolved either way; nothing to day-trade
        excluded = set(self._settings.risk.blocked_categories) | set(cfg.exclude_categories)
        if blocked_category_hit(excluded, market.question, market.description,
                                market.category):
            return False
        return True

    # ------------------------------------------------------------------
    # Prompt blocks — memory, open book, candidates
    # ------------------------------------------------------------------

    async def _memory_block(self, alias: str, limit: int) -> str:
        rows = await self._db.fetchall(
            """SELECT t.question, t.token, t.prob, t.market_prob, t.thesis,
                      SUM(l.pnl) AS pnl
               FROM agent_trader_theses t
               JOIN pnl_ledger l ON l.market_id = t.market_id
                    AND l.strategy_source = ? AND l.is_paper = 1
               WHERE t.model_alias = ? AND t.entered = 1
               GROUP BY t.id ORDER BY t.created_at DESC LIMIT ?""",
            (self.cell(alias), alias, limit),
        )
        if not rows:
            return "(no closed trades yet)"
        lines = []
        for r in rows:
            outcome = float(r["pnl"] or 0.0)
            lines.append(
                f"- [{'WIN' if outcome > 0 else 'LOSS'} ${outcome:+.2f}] "
                f"{r['token']} @ crowd {float(r['market_prob'] or 0):.2f} / "
                f"you {float(r['prob'] or 0):.2f} — {r['question']} — "
                f"thesis: {r['thesis']}"
            )
        return "\n".join(lines)

    async def _market_claimed(self, market_id: str) -> bool:
        """Any existing trade or position on this market — by ANY strategy or
        arm — claims it for that entrant's cell (market-level settlement
        attribution), so a new entry here would mis-attribute."""
        row = await self._db.fetchone(
            "SELECT 1 FROM trades WHERE market_id = ? LIMIT 1", (market_id,))
        if row is not None:
            return True
        row = await self._db.fetchone(
            "SELECT 1 FROM portfolio WHERE market_id = ? LIMIT 1", (market_id,))
        return row is not None

    async def _open_theses(self, alias: str) -> list:
        """This alias's ENTERED theses with no realized event yet — its open
        book. Stake-0 prediction-only rows (entered=0) are analysis data, not
        positions."""
        # A thesis is open iff the POSITION IS STILL HELD. Do not infer it from
        # the absence of a closing pnl_ledger row: that inference stuck open in
        # three separate ways and wedged this arm's book at the cap while it
        # held a single real position (2026-07-25 — 9 of 10 "open" were
        # phantoms, entries blocked for days, looking exactly like ordinary
        # selectivity).
        #   * maintenance purges of paper rows delete the position without ever
        #     writing a closure, so the thesis stays open forever;
        #   * when several arms enter the same market the closure is attributed
        #     to whichever cell owned the position, so the others never clear;
        #   * exits book under strategy_source='exit', not the entering cell.
        # cost_basis is the authoritative holding record, so asking it directly
        # is immune to all three. It is not strategy-scoped, so two arms in the
        # same market AND side each see the position — an over-count, but a
        # self-correcting one, unlike a permanent phantom.
        return await self._db.fetchall(
            """SELECT t.market_id, t.question, t.token, t.prob, t.market_prob,
                      t.thesis, t.created_at
               FROM agent_trader_theses t
               WHERE t.model_alias = ? AND t.entered = 1 AND t.cell = ?
                 AND EXISTS (SELECT 1 FROM cost_basis cb
                             WHERE cb.market_id = t.market_id
                               AND UPPER(cb.token) = UPPER(t.token)
                               AND cb.size > 0)""",
            (alias, self.cell(alias)),
        )

    @staticmethod
    def _open_block(rows: list) -> str:
        if not rows:
            return "(empty)"
        return "\n".join(
            f"- {r['token']} since {str(r['created_at'])[:10]} @ crowd "
            f"{float(r['market_prob'] or 0):.2f} — {r['question']}"
            for r in rows
        )

    @staticmethod
    def _candidates_block(markets: list[Market]) -> str:
        lines = []
        for m in markets:
            end = ""
            if m.end_date is not None:
                end_dt = m.end_date if m.end_date.tzinfo else m.end_date.replace(tzinfo=timezone.utc)
                end = f", ends {end_dt.strftime('%Y-%m-%d')}"
            desc = (m.description or "").replace("\n", " ")[:200]
            lines.append(
                f"- id={m.id} | YES={m.outcome_yes_price:.2f} | "
                f"vol=${m.volume:,.0f}{end} | {m.question} | {desc}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM call — the model IS the experimental variable
    # ------------------------------------------------------------------

    async def _call_model(self, prompt: str, model: str, effort: str, cfg) -> str:
        from auramaur.nlp import call_budget
        from auramaur.nlp.errors import BudgetExhausted
        from auramaur.subprocess_security import analysis_subprocess_env

        budget = self._settings.nlp.daily_claude_call_budget
        if budget > 0:
            limit = call_budget.non_reserved_limit(self._settings)
            if call_budget.calls_today() >= limit:
                raise BudgetExhausted(
                    f"non-reserved Claude budget ({limit}/{budget}, paced) exhausted")
        # cwd MUST be neutral: `claude -p` loads CLAUDE.md and the project
        # memory from its working directory, and run from the repo root the
        # arms were caught citing the operator's own market analyses ("your
        # prior CPI work") — contaminating the A/B and correlating the arms.
        proc = await asyncio.create_subprocess_exec(
            "claude", "-p", prompt,
            "--output-format", "text",
            "--model", model,
            "--effort", effort,
            "--allowedTools", _ALLOWED_TOOLS,
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
            raise RuntimeError(f"model call timed out ({model})")
        call_budget.record_call()
        if proc.returncode != 0:
            raise RuntimeError(
                f"model call failed ({model}): {stderr.decode()[:200]}")
        return stdout.decode()

    # ------------------------------------------------------------------
    # Gemini arms — paid REST API, own daily ceiling, metered cost
    # ------------------------------------------------------------------

    async def _gemini_calls_today(self) -> int:
        row = await self._db.fetchone(
            """SELECT COALESCE(SUM(calls), 0) AS n FROM agent_trader_costs
               WHERE day = date('now')""")
        return int(row["n"]) if row else 0

    async def _record_gemini_cost(self, alias: str, usd: float) -> None:
        await self._db.execute(
            """INSERT INTO agent_trader_costs (day, model_alias, calls, usd)
               VALUES (date('now'), ?, 1, ?)
               ON CONFLICT(day, model_alias) DO UPDATE SET
                   calls = calls + 1, usd = usd + excluded.usd""",
            (alias, usd))
        await self._db.commit()

    async def _gemini_request(self, url: str, body: dict, timeout_s: int) -> dict:
        """One POST to the Gemini REST API (separate method so tests stub it)."""
        import aiohttp
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json=body,
                              timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
                return await r.json()

    async def _call_gemini(self, prompt: str, spec, cfg) -> str:
        """A Gemini arm's read. PAID per token (unlike the Claude arms on the
        Max+ plan), so: (a) its own daily ceiling, independent of the Claude
        paced pool; (b) every call's token usage is metered into
        agent_trader_costs at configured $/Mtok — the arm's record is judged
        net of its own intelligence bill (operator's cost-inclusive rule).
        Google-Search grounding is enabled for research parity with the
        Claude arms' WebSearch; no JSON response mime (grounding conflicts),
        the tolerant parse_decisions handles prose-wrapped JSON."""
        key = self._settings.gemini_api_key
        if not key:
            raise RuntimeError("gemini arm: no GEMINI_API_KEY configured")
        limit = int(getattr(cfg, "gemini_daily_call_limit", 0))
        if limit > 0 and await self._gemini_calls_today() >= limit:
            raise RuntimeError(
                f"gemini daily call limit ({limit}) exhausted")
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{spec.model}:generateContent?key={key}")
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048},
        }
        data = await self._gemini_request(url, body, int(cfg.llm_timeout_seconds))
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(
                f"gemini reply unparseable ({spec.model}): "
                f"{str(data)[:160]}")
        usage = data.get("usageMetadata", {}) or {}
        prices = (getattr(cfg, "gemini_price_per_mtok", {}) or {}).get(
            spec.model, [0.0, 0.0])
        usd = (float(usage.get("promptTokenCount", 0)) * float(prices[0])
               + float(usage.get("candidatesTokenCount", 0)) * float(prices[1])
               ) / 1e6
        await self._record_gemini_cost(spec.alias, usd)
        log.info("agent_trader.gemini_call", alias=spec.alias,
                 model=spec.model, usd=round(usd, 5),
                 in_tok=usage.get("promptTokenCount", 0),
                 out_tok=usage.get("candidatesTokenCount", 0))
        return text

    # ------------------------------------------------------------------
    # Entry — full risk gate, gateway placement, same rails as every pillar
    # ------------------------------------------------------------------

    async def _try_enter(self, alias: str, market: Market, decision: dict,
                         cfg) -> bool:
        assessment = assess_agent_trader(AgentTraderInputs(
            market_id=market.id,
            question=market.question,
            market_yes=market.outcome_yes_price,
            prob_yes=decision["prob_yes"],
            thesis=decision["thesis"],
        ), AgentTraderRules(
            min_edge_pts=cfg.min_edge_pts,
            stake_usd=cfg.stake_usd,
            source_tag=self.cell(alias),
        ))
        if assessment.proposal is None:
            log.debug("agent_trader.edge_below_floor", alias=alias,
                      market_id=market.id, rejection=assessment.rejection)
            return False
        proposal = assessment.proposal
        prob_yes = proposal.probability_yes
        market_yes = proposal.market_probability
        edge_pts = proposal.edge_percent
        side = OrderSide.BUY if proposal.buy_yes else OrderSide.SELL

        # ONE POSITION PER MARKET, across the whole bot. Settlement attribution
        # is market-level earliest-entrant-wins (ledger._ENTRY_STRATEGY_SQL), so
        # a second arm — even on the OPPOSITE token — would have its P&L
        # credited to the first arm's cell and corrupt the A/B. The blocked
        # arm's PREDICTION is still recorded (stake 0, entered=0): the
        # head-to-head disagreement stays measurable via calibration, just not
        # via the ledger.
        if await self._market_claimed(market.id):
            await self._db.execute(
                """INSERT INTO agent_trader_theses
                   (model_alias, cell, market_id, question, token, prob,
                    market_prob, thesis, stake, entered)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0)""",
                (alias, self.cell(alias), market.id, market.question,
                 "YES" if side == OrderSide.BUY else "NO",
                 prob_yes, market_yes, proposal.thesis),
            )
            # 2026-08-05 (#353 phase 5): trailing commit() removed — dead
            # under the autocommit connection (no-op since eed51b8); a
            # single INSERT is already atomic (classification B).
            log.info("agent_trader.market_claimed", alias=alias,
                     market_id=market.id, prob=round(prob_yes, 2))
            return False

        signal = Signal(
            market_id=market.id,
            market_question=market.question,
            claude_prob=prob_yes,
            claude_confidence=Confidence.MEDIUM,
            market_prob=market_yes,
            edge=edge_pts,
            evidence_summary=proposal.thesis,
            recommended_side=side,
            strategy_source=proposal.source_tag,
            mispricing_reason=proposal.thesis[:300],
        )
        await self._persist_signal(signal, market)

        # Full risk gate — all checks apply; never bypassed.
        risk_decision = await self._risk.evaluate(signal, market)
        if not risk_decision.approved or risk_decision.position_size <= 0:
            log.info("agent_trader.risk_rejected", alias=alias,
                     market_id=market.id, reason=risk_decision.reason)
            return False
        size = min(risk_decision.position_size, cfg.stake_usd)

        force_paper = cfg.paper or getattr(risk_decision, "force_paper", False)
        res = await self._gateway.submit(TradeIntent(
            signal=signal, market=market, size_dollars=size,
            force_paper=force_paper))
        if res.status not in ("filled", "paper", "partial", "pending"):
            log.info("agent_trader.order_rejected", alias=alias,
                     market_id=market.id, status=res.status, error=res.reason)
            return False

        # 2026-08-05 (#353 phase 5): the portfolio upsert and this thesis
        # INSERT land in ONE span, opened only after gateway.submit has
        # returned (never wrap a gateway call — re-entrancy would JOIN the
        # span across network I/O). Under autocommit each statement was
        # individually durable, so a crash between them left a position row
        # without its thesis row — a held position no arm settles in its own
        # book (_open_theses joins the two). The trailing commit() this
        # replaced was dead (no-op since eed51b8). Calibration stays outside
        # the span, below.
        async with self._db.transaction(owner="agent_trader.entry"):
            await self._record_position(signal, market, res.order, res.result)
            await self._db.execute(
                """INSERT INTO agent_trader_theses
                   (model_alias, cell, market_id, question, token, prob,
                    market_prob, thesis, stake)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (alias, self.cell(alias), market.id, market.question,
                 res.order.token.value,
                 prob_yes, market_yes, proposal.thesis, size),
            )
        try:
            await self._calibration.record_prediction(
                res.order.market_id, signal.claude_prob, market.category or "")
        except Exception as e:
            log.debug("agent_trader.calibration_error", error=str(e))
        log.info("agent_trader.entered", alias=alias, market_id=market.id,
                 token=res.order.token.value, price=res.order.price,
                 size=res.order.size, edge=round(edge_pts, 1),
                 paper=res.result.is_paper)
        return True

    # ------------------------------------------------------------------
    # Bookkeeping — mirrors long_horizon/bias_harvest
    # ------------------------------------------------------------------

    async def _persist_signal(self, signal: Signal, market: Market) -> None:
        # 2026-08-04 (#353 phase 3): markets stub + signals row land in one
        # span. The trailing commit() this replaced was dead under the
        # autocommit connection (no-op since eed51b8) — the span provides
        # the atomicity it implied.
        async with self._db.transaction(owner="agent_trader.signal"):
            await self._db.execute(
                """INSERT OR IGNORE INTO markets (id, exchange, condition_id, question,
                   description, category, active, outcome_yes_price, outcome_no_price,
                   volume, liquidity, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, datetime('now'))""",
                (market.id, market.exchange or self._exchange_name, market.condition_id,
                 market.question, (market.description or "")[:500],
                 ensure_category(market.question, market.description, market.category),
                 market.outcome_yes_price, market.outcome_no_price,
                 market.volume, market.liquidity),
            )
            await self._db.execute(
                """INSERT INTO signals (market_id, claude_prob, claude_confidence,
                   market_prob, edge, evidence_summary, action, strategy_source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (signal.market_id, signal.claude_prob, signal.claude_confidence.value,
                 signal.market_prob, signal.edge, signal.evidence_summary,
                 signal.recommended_side.value, signal.strategy_source),
            )

    async def _record_position(self, signal: Signal, market: Market,
                               order, result) -> None:
        """Portfolio row (resolution tracker settles it). The fill
        (-> cost_basis -> pnl_ledger) and the trades-mirror are owned by
        ExecutionGateway.

        2026-08-05 (#353 phase 5): runs INSIDE the caller's
        agent_trader.entry span (sole caller: _try_enter), so this upsert
        and the thesis INSERT commit together. The trailing commit() this
        carried was dead (no-op since eed51b8); the calibration call moved
        to the caller, after the span, per the never-wrap-calibration rule.
        """
        fill_size = result.filled_size if result.filled_size > 0 else order.size
        fill_price = result.filled_price if result.filled_price > 0 else order.price
        is_paper = bool(result.is_paper)
        await self._db.execute(
            """INSERT INTO portfolio (market_id, exchange, side, size, avg_price,
               current_price, unrealized_pnl, category, token, token_id,
               is_paper, updated_at)
               VALUES (?, ?, 'BUY', ?, ?, ?, 0, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(market_id, is_paper, token) DO UPDATE SET
                   size = excluded.size,
                   avg_price = excluded.avg_price,
                   current_price = excluded.current_price,
                   updated_at = excluded.updated_at""",
            (order.market_id, market.exchange or self._exchange_name,
             fill_size, fill_price, fill_price,
             market.category or "", order.token.value, order.token_id,
             1 if is_paper else 0),
        )
