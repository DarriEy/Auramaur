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
import json
import tempfile
from datetime import datetime, timedelta, timezone

import structlog

from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.exchange.models import Confidence, Market, OrderSide, Signal
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

_THESES_TABLE = """
CREATE TABLE IF NOT EXISTS agent_trader_theses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_alias TEXT NOT NULL,
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


def parse_decisions(raw: str, max_entries: int) -> list[dict]:
    """Tolerant parse of the model's JSON: accepts fenced/prefixed output,
    drops malformed entries, clamps probabilities, caps the count. Returns
    ``[]`` on anything unparseable — a bad LLM reply must never crash a cycle."""
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        payload = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        return []
    out: list[dict] = []
    for d in decisions:
        if not isinstance(d, dict):
            continue
        market_id = str(d.get("market_id", "")).strip()
        thesis = str(d.get("thesis", "")).strip()
        try:
            prob = float(d.get("prob_yes"))
        except (TypeError, ValueError):
            continue
        if not market_id or not thesis:
            continue
        if not (0.0 < prob < 1.0):
            continue
        out.append({"market_id": market_id, "prob_yes": prob, "thesis": thesis})
        if len(out) >= max_entries:
            break
    return out


class AgentTraderPillar:
    """Multi-model LLM day-trader over near-dated Polymarket markets."""

    name = "agent_trader"

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
        self._cycle_n = 0

    @staticmethod
    def cell(alias: str) -> str:
        return f"agent_trader_{alias}"

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        await self._db.execute(_THESES_TABLE)
        await self._db.execute(_DECLINES_TABLE)
        try:
            # Upgrade a pre-`entered` table in place (rows so far all entered).
            await self._db.execute(
                "ALTER TABLE agent_trader_theses ADD COLUMN entered INTEGER DEFAULT 1")
        except Exception:
            pass  # column already exists
        await self._db.commit()
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
            log.debug("agent_trader.book_full", alias=alias,
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
        raw = await self._call_model(prompt, spec.model, spec.effort, cfg)
        decisions = parse_decisions(raw, cfg.max_entries_per_cycle)

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
        if (market.exchange or "polymarket") != "polymarket":
            return False
        if market.liquidity < cfg.min_liquidity:
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
        return await self._db.fetchall(
            """SELECT t.market_id, t.question, t.token, t.prob, t.market_prob,
                      t.thesis, t.created_at
               FROM agent_trader_theses t
               WHERE t.model_alias = ? AND t.entered = 1
                 AND NOT EXISTS (SELECT 1 FROM pnl_ledger l
                                 WHERE l.market_id = t.market_id
                                   AND l.strategy_source = ?
                                   AND l.is_paper = 1)""",
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
    # Entry — full risk gate, gateway placement, same rails as every pillar
    # ------------------------------------------------------------------

    async def _try_enter(self, alias: str, market: Market, decision: dict,
                         cfg) -> bool:
        prob_yes = decision["prob_yes"]
        market_yes = market.outcome_yes_price
        edge_pts = abs(prob_yes - market_yes) * 100.0
        if edge_pts < cfg.min_edge_pts:
            log.debug("agent_trader.edge_below_floor", alias=alias,
                      market_id=market.id, edge=round(edge_pts, 1))
            return False
        side = OrderSide.BUY if prob_yes > market_yes else OrderSide.SELL

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
                   (model_alias, market_id, question, token, prob, market_prob,
                    thesis, stake, entered)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)""",
                (alias, market.id, market.question,
                 "YES" if side == OrderSide.BUY else "NO",
                 prob_yes, market_yes, decision["thesis"][:500]),
            )
            await self._db.commit()
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
            evidence_summary=decision["thesis"][:500],
            recommended_side=side,
            strategy_source=self.cell(alias),
            mispricing_reason=decision["thesis"][:300],
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

        await self._record_position(signal, market, res.order, res.result)
        await self._db.execute(
            """INSERT INTO agent_trader_theses
               (model_alias, market_id, question, token, prob, market_prob,
                thesis, stake)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (alias, market.id, market.question, res.order.token.value,
             prob_yes, market_yes, decision["thesis"][:500], size),
        )
        await self._db.commit()
        log.info("agent_trader.entered", alias=alias, market_id=market.id,
                 token=res.order.token.value, price=res.order.price,
                 size=res.order.size, edge=round(edge_pts, 1),
                 paper=res.result.is_paper)
        return True

    # ------------------------------------------------------------------
    # Bookkeeping — mirrors long_horizon/bias_harvest
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
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (signal.market_id, signal.claude_prob, signal.claude_confidence.value,
             signal.market_prob, signal.edge, signal.evidence_summary,
             signal.recommended_side.value, signal.strategy_source),
        )
        await self._db.commit()

    async def _record_position(self, signal: Signal, market: Market,
                               order, result) -> None:
        """Portfolio row (resolution tracker settles it) + calibration
        prediction. The fill (-> cost_basis -> pnl_ledger) and the
        trades-mirror are owned by ExecutionGateway."""
        fill_size = result.filled_size if result.filled_size > 0 else order.size
        fill_price = result.filled_price if result.filled_price > 0 else order.price
        is_paper = bool(result.is_paper)
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
             1 if is_paper else 0),
        )
        await self._db.commit()
        try:
            await self._calibration.record_prediction(
                order.market_id, signal.claude_prob, market.category or "")
        except Exception as e:
            log.debug("agent_trader.calibration_error", error=str(e))
