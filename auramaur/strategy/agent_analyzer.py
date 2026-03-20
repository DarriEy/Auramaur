"""Agent-based market analyzer — uses Claude as an autonomous reasoning agent.

Instead of the pipeline (data → NLP → signal detection), the agent
browses the web, reasons about probabilities, and returns trade
candidates in one continuous chain of thought.

This replaces: data_sources/*, nlp/analyzer.py, nlp/strategic.py,
nlp/prompts.py, and strategy/signals.py — all collapsed into the
agent's own reasoning.

The agent does NOT have access to risk checks, position sizing, or
order placement.  Those remain external and non-negotiable.
"""

from __future__ import annotations

import asyncio
import json
import re

import structlog

from auramaur.db.database import Database
from auramaur.exchange.models import Confidence, Market, OrderSide, Signal
from auramaur.strategy.protocols import TradeCandidate

log = structlog.get_logger()

# Maximum markets per agent call to keep context focused
_MAX_MARKETS_PER_CALL = 15

AGENT_SYSTEM_PROMPT = """\
You are an autonomous prediction market analyst. You will be given a batch \
of prediction markets. For each one, you must:

1. Research the question using your web browsing tools
2. Form a calibrated probability estimate using superforecasting methodology:
   - Start with a base rate (outside view)
   - Update based on evidence (inside view)
   - Decompose into sub-questions (Fermi)
   - Check for narrative bias and overconfidence
3. Compare your estimate to the current market price
4. Identify markets where you have an informational edge

=== CALIBRATION HISTORY ===
{calibration_feedback}

=== IMPORTANT RULES ===
- You are being paid for ACCURACY, not for having opinions
- If you don't know, stay close to the market price (no edge = no trade)
- Most things DON'T happen — base rates for "Will X happen?" are typically 10-30%
- Your estimates must be between 5% and 95% unless you have extraordinary evidence
- Evidence quality matters: Reuters/AP > news analysis > opinion > social media

=== OUTPUT FORMAT ===
After analyzing all markets, respond with a JSON block:
```json
{{
  "candidates": [
    {{
      "market_id": "<id>",
      "probability": <float 0-1>,
      "confidence": "<LOW|MEDIUM|HIGH>",
      "reasoning": "<your step-by-step analysis>",
      "recommended_side": "<BUY|SELL>"
    }}
  ]
}}
```

Only include markets where you found a meaningful edge (>3% divergence \
from market price). Omit markets where you agree with the market price.
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


class AgentAnalyzer:
    """Implements MarketAnalyzer using an autonomous Claude agent.

    The agent uses tool-use (web browsing, search) to gather its own
    evidence and reasons about probabilities in a single chain of
    thought.  This replaces the entire pipeline middle layer.

    IMPORTANT: This analyzer forces paper trading mode.  It is
    experimental and must not place live orders.
    """

    def __init__(self, settings, db: Database):
        self.settings = settings
        self.db = db
        self._model = settings.nlp.model

    async def analyze_markets(
        self,
        markets: list[Market],
        price_history: dict[str, list[float]] | None = None,
    ) -> list[TradeCandidate]:
        """Launch the agent to analyze markets autonomously."""
        if not markets:
            return []

        # Batch into chunks to keep agent context focused
        all_candidates: list[TradeCandidate] = []
        for i in range(0, len(markets), _MAX_MARKETS_PER_CALL):
            batch = markets[i:i + _MAX_MARKETS_PER_CALL]
            try:
                candidates = await self._run_agent(batch)
                all_candidates.extend(candidates)
            except Exception as e:
                log.error("agent.batch_failed", batch_start=i, error=str(e))

        return all_candidates

    async def _run_agent(self, markets: list[Market]) -> list[TradeCandidate]:
        """Run Claude agent on a batch of markets."""
        calibration = await self._get_calibration_feedback()
        markets_block = _format_markets_for_agent(markets)

        prompt = (
            AGENT_SYSTEM_PROMPT.format(calibration_feedback=calibration)
            + "\n\n=== TODAY'S MARKETS ===\n"
            + markets_block
            + "\n\nResearch each market using web search, then provide your analysis as JSON."
        )

        log.info("agent.call_start", market_count=len(markets), prompt_chars=len(prompt))

        raw = await self._call_claude_agent(prompt)
        candidates = self._parse_response(raw, markets)

        log.info("agent.call_complete", candidates=len(candidates))
        return candidates

    async def _call_claude_agent(self, prompt: str) -> str:
        """Call Claude CLI in agent mode (with tool use for web browsing)."""
        proc = await asyncio.create_subprocess_exec(
            "claude", "-p", prompt,
            "--output-format", "text",
            "--model", self._model,
            "--max-turns", "10",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)

        if proc.returncode != 0:
            err = stderr.decode().strip()
            raise RuntimeError(f"Agent call failed (rc={proc.returncode}): {err}")

        return stdout.decode().strip()

    def _parse_response(self, text: str, markets: list[Market]) -> list[TradeCandidate]:
        """Parse agent's JSON response into TradeCandidate objects."""
        # Extract JSON block
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fenced:
            text = fenced.group(1)

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            log.error("agent.parse_failed", preview=text[:300])
            return []

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            log.error("agent.json_decode_failed", preview=text[:300])
            return []

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

                signal = Signal(
                    market_id=market_id,
                    market_question=market.question,
                    claude_prob=prob,
                    claude_confidence=Confidence(item.get("confidence", "MEDIUM").upper()),
                    market_prob=market_prob,
                    edge=edge * 100,
                    evidence_summary=item.get("reasoning", "")[:500],
                    recommended_side=side,
                )

                candidates.append(TradeCandidate(market=market, signal=signal))

            except (KeyError, ValueError) as e:
                log.warning("agent.candidate_parse_error", market_id=market_id, error=str(e))

        return candidates

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
