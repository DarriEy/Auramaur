"""Agent-based market analyzer — relational ontology approach.

Instead of analyzing markets as isolated objects with independent
probabilities, the agent reasons about markets as nodes in a web of
relationships.  Entities (people, institutions, events, forces) exist
only through their relations to other entities.  A market's probability
is not an intrinsic property — it emerges from the confluence of
relationships that bear on it.

This replaces: data_sources/*, nlp/analyzer.py, nlp/strategic.py,
nlp/prompts.py, and strategy/signals.py — all collapsed into the
agent's relational reasoning.

The agent does NOT have access to risk checks, position sizing, or
order placement.  Those remain external and non-negotiable.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import date
from pathlib import Path

import structlog

from auramaur.db.database import Database
from auramaur.exchange.models import Confidence, Market, OrderSide, Signal
from auramaur.strategy.protocols import TradeCandidate

log = structlog.get_logger()

_MAX_MARKETS_PER_CALL = 15
_WORLD_MODEL_PATH = Path("world_model.json")

# Only allow web search + fetch — no file system, no code execution
_ALLOWED_TOOLS = "WebSearch,WebFetch"

_EMPTY_WORLD_MODEL: dict = {
    "macro_outlook": "",
    "key_beliefs": [],
    "cross_market_patterns": [],
    "active_themes": [],
}


AGENT_SYSTEM_PROMPT = """\
You are a relational analyst for prediction markets.  You do not treat \
markets as isolated questions with independent probabilities.  Instead, \
you see markets as NODES in a web of relationships between entities — \
people, institutions, events, forces, narratives.

=== YOUR ONTOLOGICAL FRAMEWORK ===

**Nothing exists in isolation.**  Every market question involves entities \
that stand in relation to other entities.  "Will X happen?" is never just \
about X — it's about the web of forces, actors, incentives, constraints, \
and path dependencies that make X more or less likely.

Your analysis proceeds in THREE PHASES:

**PHASE 1 — RELATIONAL MAPPING**
Before estimating any probability, map the relational structure:
- What ENTITIES are involved? (people, institutions, markets, forces)
- What RELATIONS connect them? (causal, constraining, enabling, opposing)
- What CLUSTERS emerge? (groups of markets linked by shared entities)
- What TENSIONS exist? (contradictions between related claims)

For example: "Will Trump impose tariffs on China?" and "Will S&P 500 \
exceed 5500?" share entities (Trump, US economy, trade policy) and \
the first CONSTRAINS the second.  Estimating them independently is \
an error.

**PHASE 2 — RELATIONAL EVIDENCE GATHERING**
Research not just individual markets but the RELATIONSHIPS between them:
- Search for the entities that connect multiple markets
- Look for events that update multiple probabilities simultaneously
- Find structural constraints (if A then B becomes less likely)
- Identify narrative threads (stories that the market is pricing in)

One piece of evidence about a shared entity (e.g., "Trump signals \
Iran de-escalation") updates EVERY market connected to that entity.  \
This is where your edge comes from — markets price each question \
independently, but reality is relational.

**PHASE 3 — COHERENT ESTIMATION**
Your probability estimates must be JOINTLY COHERENT:
- If P(A) = 70% and A makes B very unlikely, P(B) can't be 60%
- If markets share a root cause, their probabilities co-move
- If the market prices two related events as independent when they're not, \
  that's your edge

=== YOUR PERSISTENT WORLD MODEL ===
{world_model}

=== CALIBRATION HISTORY ===
{calibration_feedback}

=== RELATIONAL REASONING PRINCIPLES ===

1. **Entities are constituted by their relations.**  "Trump" is not a \
   fixed object but a nexus of relations: to Congress, to Iran, to \
   markets, to voters, to his legal cases.  A change in ANY relation \
   changes the entity itself.

2. **Probabilities are relational, not intrinsic.**  P(Iran war ends) \
   is not a property of "the Iran war" — it's a function of the \
   relations between Trump, Khamenei, Congress, oil markets, military \
   capacity, domestic politics, and regional allies.

3. **Markets are coupled, not independent.**  The prediction market \
   prices each question as if it's independent.  When you find COUPLED \
   markets priced as if independent, that's mispricing.

4. **Evidence propagates through relations.**  A single news item about \
   entity X updates every market that X is connected to.  Don't research \
   each market separately — research the ENTITIES and let the updates \
   propagate.

5. **Narrative coherence reveals edge.**  If the market tells an \
   incoherent story (e.g., prices in Iran de-escalation for oil but \
   prices in escalation for defense stocks), one side is wrong.

=== IMPORTANT CONSTRAINTS ===
- Accuracy over opinions.  If you can't find a relational edge, say so.
- Base rates still matter.  Relations inform the UPDATE from base rate.
- Evidence quality hierarchy: Reuters/AP/official > analysis > opinion > social
- Your estimates must be between 5% and 95% unless extraordinary evidence

=== YOUR TASK ===
You will receive a batch of markets.  Do NOT research each market \
independently.  Instead:

1. Identify the KEY ENTITIES across all markets
2. Research those entities (web search for the 5-8 most important ones)
3. Map how new information propagates across markets
4. Estimate probabilities that are JOINTLY COHERENT
5. Flag markets where relational reasoning reveals mispricing

=== OUTPUT FORMAT ===
Respond with a JSON block containing ALL markets where you found an \
edge (>3% divergence from market price).  Include your relational \
reasoning — what entities connect this market to others and how that \
informs your estimate.

```json
{{
  "entity_map": {{
    "<entity_name>": {{
      "connected_markets": ["<market_id>", ...],
      "current_state": "<what you learned about this entity>",
      "update_direction": "<how this updates connected markets>"
    }}
  }},
  "candidates": [
    {{
      "market_id": "<id>",
      "probability": <float 0-1>,
      "confidence": "<LOW|MEDIUM|HIGH>",
      "reasoning": "<relational analysis: which entities and relations inform this estimate>",
      "related_markets": ["<ids of markets constrained by this estimate>"],
      "recommended_side": "<BUY|SELL>"
    }}
  ]
}}
```

Include the entity_map even if you find no edges — it updates the world model.
Omit markets where you agree with the market price (no relational edge).
"""


def _format_markets_for_agent(markets: list[Market]) -> str:
    """Format markets into a concise block for the agent prompt."""
    lines = []
    for i, m in enumerate(markets, 1):
        lines.append(
            f"--- MARKET {i} (id: {m.id}) ---\n"
            f"Question: {m.question}\n"
            f"Description: {m.description[:300]}\n"
            f"Category: {m.category}\n"
            f"Current YES price: {m.outcome_yes_price:.1%}\n"
            f"End date: {m.end_date.isoformat() if m.end_date else 'Unknown'}\n"
            f"Liquidity: ${m.liquidity:,.0f}\n"
        )
    return "\n".join(lines)


def _ensure_world_model() -> dict:
    """Load the world model, seeding an empty one on first run."""
    if _WORLD_MODEL_PATH.exists():
        try:
            return json.loads(_WORLD_MODEL_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("agent.world_model_corrupt, reseeding")

    data = dict(_EMPTY_WORLD_MODEL)
    _WORLD_MODEL_PATH.write_text(json.dumps(data, indent=2))
    log.info("agent.world_model_seeded", path=str(_WORLD_MODEL_PATH))
    return data


def _format_world_model(data: dict) -> str:
    """Format world model dict into a context string for the prompt."""
    parts = []
    if data.get("macro_outlook"):
        parts.append(f"MACRO STATE:\n{data['macro_outlook'][:2000]}")
    if data.get("key_beliefs"):
        beliefs = data["key_beliefs"][-10:]
        parts.append("KEY BELIEFS:\n" + "\n".join(f"- {b}" for b in beliefs))
    if data.get("cross_market_patterns"):
        patterns = data["cross_market_patterns"][-8:]
        parts.append("CROSS-MARKET PATTERNS:\n" + "\n".join(f"- {p}" for p in patterns))
    if data.get("active_themes"):
        parts.append("ACTIVE THEMES: " + ", ".join(data["active_themes"][-8:]))
    return "\n\n".join(parts) if parts else "(No world model yet — first run.)"


class AgentAnalyzer:
    """Implements MarketAnalyzer using a relational-ontology Claude agent.

    The agent reasons about markets as interconnected nodes in a web
    of entities and relations, not as independent questions.  It
    researches ENTITIES (shared across markets) rather than individual
    markets, letting insights propagate through the relational graph.

    Uses the normal triple-gate for live/paper trading.
    """

    def __init__(self, settings, db: Database, calibration=None):
        self.settings = settings
        self.db = db
        self.calibration = calibration
        self._model = settings.nlp.model
        self._max_turns: int = 15  # Enough room for deep research
        self._timeout_seconds: int = 600  # 10 minutes for thorough analysis
        # Daily budget tracking (resets each calendar day)
        self._daily_calls: int = 0
        self._daily_calls_date: str = ""

    def _check_budget(self) -> None:
        """Enforce daily Claude call budget.  Raises RuntimeError if exhausted."""
        today = date.today().isoformat()
        if self._daily_calls_date != today:
            self._daily_calls = 0
            self._daily_calls_date = today
        budget = self.settings.nlp.daily_claude_call_budget
        if budget > 0 and self._daily_calls >= budget:
            raise RuntimeError(
                f"Daily agent call budget ({budget}) exhausted"
            )

    async def analyze_markets(
        self,
        markets: list[Market],
        price_history: dict[str, list[float]] | None = None,
    ) -> list[TradeCandidate]:
        """Launch the relational agent to analyze markets."""
        if not markets:
            return []

        all_candidates: list[TradeCandidate] = []
        for i in range(0, len(markets), _MAX_MARKETS_PER_CALL):
            self._check_budget()
            batch = markets[i:i + _MAX_MARKETS_PER_CALL]
            try:
                candidates, entity_map = await self._run_agent(batch)
                self._daily_calls += 1
                all_candidates.extend(candidates)
                self._update_world_model_from_entities(entity_map)
            except RuntimeError:
                raise  # budget errors propagate
            except Exception as e:
                log.error("agent.batch_failed", batch_start=i, error=str(e))

        return all_candidates

    async def _run_agent(
        self, markets: list[Market],
    ) -> tuple[list[TradeCandidate], dict]:
        """Run the relational agent on a batch of markets."""
        calibration = await self._get_calibration_feedback()
        world_data = _ensure_world_model()
        world_model = _format_world_model(world_data)
        markets_block = _format_markets_for_agent(markets)

        prompt = (
            AGENT_SYSTEM_PROMPT.format(
                world_model=world_model,
                calibration_feedback=calibration,
            )
            + "\n\n=== TODAY'S MARKETS ===\n"
            + markets_block
            + "\n\nIdentify key entities across these markets, research them, "
            + "then provide your relational analysis as JSON."
        )

        log.info(
            "agent.call_start",
            market_count=len(markets),
            prompt_chars=len(prompt),
        )

        raw = await self._call_claude_agent(prompt)
        candidates, entity_map = self._parse_response(raw, markets)

        # Apply Platt scaling calibration to raw probabilities
        if self.calibration and candidates:
            for tc in candidates:
                raw_prob = tc.signal.claude_prob
                calibrated = await self.calibration.adjust(
                    raw_prob, tc.market.category or ""
                )
                if abs(calibrated - raw_prob) > 0.001:
                    # Recalculate edge with calibrated probability
                    market_prob = tc.signal.market_prob
                    new_edge = abs(calibrated - market_prob)
                    tc.signal.claude_prob = calibrated
                    tc.signal.edge = new_edge * 100
                    log.info(
                        "agent.calibrated",
                        market_id=tc.market.id,
                        raw=round(raw_prob, 3),
                        calibrated=round(calibrated, 3),
                        edge_before=round(abs(raw_prob - market_prob) * 100, 1),
                        edge_after=round(new_edge * 100, 1),
                    )

        log.info(
            "agent.call_complete",
            candidates=len(candidates),
            entities=len(entity_map),
        )
        return candidates, entity_map

    async def _call_claude_agent(self, prompt: str) -> str:
        """Call Claude CLI in agent mode with web search tools.

        Tool access is restricted to WebSearch + WebFetch only — no file
        system, no code execution.  Cost is bounded by --max-turns and
        the daily call budget enforced by _check_budget().
        """
        cmd = [
            "claude", "-p", prompt,
            "--output-format", "text",
            "--model", self._model,
            "--effort", "max",
            "--max-turns", str(self._max_turns),
            "--allowedTools", _ALLOWED_TOOLS,
        ]

        max_attempts = 2
        backoff = [5, 15]
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            try:
                log.debug(
                    "agent.cli_exec",
                    attempt=attempt + 1,
                    model=self._model,
                    max_turns=self._max_turns,
                )
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self._timeout_seconds,
                )

                if proc.returncode != 0:
                    err = stderr.decode().strip()
                    raise RuntimeError(
                        f"Agent call failed (rc={proc.returncode}): {err}"
                    )

                return stdout.decode().strip()

            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"Agent timed out after {self._timeout_seconds}s"
                )
                log.warning("agent.timeout", attempt=attempt + 1)
                if proc.returncode is None:
                    proc.kill()
            except RuntimeError as e:
                last_error = e
                log.warning("agent.call_error", attempt=attempt + 1, error=str(e))

            if attempt < max_attempts - 1:
                await asyncio.sleep(backoff[attempt])

        raise last_error  # type: ignore[misc]

    def _parse_response(
        self, text: str, markets: list[Market],
    ) -> tuple[list[TradeCandidate], dict]:
        """Parse agent's JSON response into TradeCandidates + entity map."""
        # Extract JSON block
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fenced:
            text = fenced.group(1)

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            log.error("agent.parse_failed", preview=text[:300])
            return [], {}

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            log.error("agent.json_decode_failed", preview=text[:300])
            return [], {}

        # Extract entity map (for world model updates)
        entity_map = data.get("entity_map", {})
        if entity_map:
            log.info(
                "agent.entities_found",
                entities=list(entity_map.keys())[:10],
            )

        # Parse trade candidates
        market_map = {m.id: m for m in markets}
        candidates: list[TradeCandidate] = []

        for item in data.get("candidates", []):
            market_id = item.get("market_id", "")
            market = market_map.get(market_id)
            if market is None:
                log.debug("agent.unknown_market_id", market_id=market_id)
                continue

            try:
                prob = float(item["probability"])
                prob = max(0.01, min(0.99, prob))
                market_prob = market.outcome_yes_price

                raw_edge = prob - market_prob
                if abs(raw_edge) < 0.001:
                    continue

                side_str = item.get("recommended_side", "BUY" if raw_edge > 0 else "SELL")
                side = OrderSide(side_str.upper())
                edge = abs(raw_edge)

                # Include relational context in evidence summary
                related = item.get("related_markets", [])
                reasoning = item.get("reasoning", "")
                if related:
                    reasoning += f" [Related: {', '.join(str(r) for r in related[:5])}]"

                signal = Signal(
                    market_id=market_id,
                    market_question=market.question,
                    claude_prob=prob,
                    claude_confidence=Confidence(item.get("confidence", "MEDIUM").upper()),
                    market_prob=market_prob,
                    edge=edge * 100,
                    evidence_summary=reasoning[:500],
                    recommended_side=side,
                )

                candidates.append(TradeCandidate(market=market, signal=signal))

            except (KeyError, ValueError) as e:
                log.warning("agent.candidate_parse_error", market_id=market_id, error=str(e))

        return candidates, entity_map

    def _update_world_model_from_entities(self, entity_map: dict) -> None:
        """Merge agent's entity discoveries into the persistent world model."""
        if not entity_map:
            return

        try:
            data = _ensure_world_model()

            # Append entity insights as cross-market patterns
            patterns = data.get("cross_market_patterns", [])
            for entity_name, info in entity_map.items():
                state = info.get("current_state", "")
                direction = info.get("update_direction", "")
                if state and direction:
                    pattern = f"[agent] {entity_name}: {state[:100]} → {direction[:100]}"
                    patterns.append(pattern)

            # Keep last 15 patterns
            data["cross_market_patterns"] = patterns[-15:]

            _WORLD_MODEL_PATH.write_text(json.dumps(data, indent=2))
            log.info(
                "agent.world_model_updated",
                new_entities=len(entity_map),
            )
        except Exception as e:
            log.warning("agent.world_model_update_failed", error=str(e))

    async def deep_research(self, market: Market) -> TradeCandidate | None:
        """Deep-dive research on a single high-potential market.

        Unlike batch analysis (breadth), this gives the agent maximum time
        and turns to thoroughly research ONE market. Used when the strategic
        loop identifies a market with potential edge but needs deeper
        conviction before committing capital.
        """
        self._check_budget()

        calibration = await self._get_calibration_feedback()
        world_data = _ensure_world_model()
        world_model = _format_world_model(world_data)

        prompt = f"""\
You are conducting DEEP RESEARCH on a single prediction market. You have \
extensive web search capabilities and should use them thoroughly.

=== YOUR WORLD MODEL ===
{world_model}

=== CALIBRATION ===
{calibration}

=== THE MARKET ===
Question: {market.question}
Description: {market.description[:1000]}
Category: {market.category}
Current YES price: {market.outcome_yes_price:.1%}
End date: {market.end_date.isoformat() if market.end_date else 'Unknown'}
Liquidity: ${market.liquidity:,.0f}

=== YOUR RESEARCH MANDATE ===
Go DEEP on this one market. This is not a scan — this is thorough research.

1. **Search for the 5-8 most relevant pieces of information**
   - Official sources (government, company statements)
   - Expert analysis and forecasts
   - Recent news developments
   - Historical precedents and base rates
   - Contrarian perspectives

2. **Evaluate evidence quality** — rate each source

3. **Apply Fermi decomposition** — break into sub-questions

4. **Check for what's MISSING** — what evidence SHOULD exist if this were likely?

5. **Compare to market price** — is the market pricing this correctly?

=== OUTPUT ===
Respond with JSON:
```json
{{
  "probability": <float 0-1>,
  "confidence": "<LOW|MEDIUM|HIGH>",
  "reasoning": "<thorough analysis with citations>",
  "key_evidence": ["<evidence 1 with source>", "<evidence 2>", ...],
  "base_rate": "<what reference class and base rate did you use?>",
  "decomposition": "<sub-questions and their probabilities>",
  "recommended_side": "<BUY|SELL>",
  "conviction_level": "<STRONG|MODERATE|WEAK>"
}}
```
"""

        log.info("agent.deep_research_start", market_id=market.id, question=market.question[:60])

        try:
            # More turns and longer timeout for deep research
            old_turns = self._max_turns
            old_timeout = self._timeout_seconds
            self._max_turns = 20
            self._timeout_seconds = 900  # 15 minutes

            raw = await self._call_claude_agent(prompt)

            self._max_turns = old_turns
            self._timeout_seconds = old_timeout
            self._daily_calls += 1

            # Parse single-market response
            fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
            text = fenced.group(1) if fenced else raw
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                log.error("agent.deep_research_parse_failed", preview=raw[:300])
                return None

            data = json.loads(match.group(0))
            prob = max(0.01, min(0.99, float(data["probability"])))
            market_prob = market.outcome_yes_price
            raw_edge = prob - market_prob

            if abs(raw_edge) < 0.03:
                log.info("agent.deep_research_no_edge", market_id=market.id, edge=round(raw_edge, 3))
                return None

            side = OrderSide.BUY if raw_edge > 0 else OrderSide.SELL

            signal = Signal(
                market_id=market.id,
                market_question=market.question,
                claude_prob=prob,
                claude_confidence=Confidence(data.get("confidence", "MEDIUM").upper()),
                market_prob=market_prob,
                edge=abs(raw_edge) * 100,
                evidence_summary=data.get("reasoning", "")[:500],
                recommended_side=side,
            )

            log.info(
                "agent.deep_research_complete",
                market_id=market.id,
                prob=round(prob, 3),
                edge=round(abs(raw_edge) * 100, 1),
                conviction=data.get("conviction_level", "?"),
            )

            return TradeCandidate(market=market, signal=signal)

        except Exception as e:
            log.error("agent.deep_research_failed", market_id=market.id, error=str(e))
            return None

    async def _get_calibration_feedback(self) -> str:
        """Load calibration history to ground the agent."""
        rows = await self.db.fetchall(
            """SELECT m.question, s.claude_prob, c.actual_outcome
               FROM signals s
               LEFT JOIN markets m ON s.market_id = m.id
               LEFT JOIN calibration c ON s.market_id = c.market_id
               WHERE c.actual_outcome IS NOT NULL
               ORDER BY s.timestamp DESC LIMIT 30"""
        )

        if not rows:
            return "(No calibration data yet — first run.)"

        correct = 0
        lines = []
        for row in rows:
            predicted = row["claude_prob"]
            actual = "YES" if row["actual_outcome"] else "NO"
            was_right = (predicted > 0.5 and row["actual_outcome"]) or \
                        (predicted <= 0.5 and not row["actual_outcome"])
            if was_right:
                correct += 1
            icon = "correct" if was_right else "WRONG"
            lines.append(f"- [{icon}] \"{row['question'][:60]}\" — you said {predicted:.0%}, resolved {actual}")

        accuracy = correct / len(rows) if rows else 0
        return f"Track record: {correct}/{len(rows)} ({accuracy:.0%} accuracy)\n" + "\n".join(lines)
