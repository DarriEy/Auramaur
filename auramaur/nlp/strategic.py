"""Strategic Analyzer — persistent world model + batch market analysis.

Unlike the stateless ClaudeAnalyzer (which analyzes each market independently),
the StrategicAnalyzer maintains a persistent world model that evolves across
cycles.  It feeds Claude:

1. Its current world model (macro outlook, geopolitical state, key beliefs)
2. Calibration feedback from past predictions
3. A batch of markets to analyze together (enabling cross-market reasoning)
4. Updates the world model based on new evidence and outcomes

This gives Claude continuity — like a human analyst who builds expertise
over time rather than starting from zero on every question.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

from auramaur.data_sources.base import NewsItem
from auramaur.db.database import Database
from auramaur.exchange.models import Market
from auramaur.nlp.prompts import format_evidence

log = structlog.get_logger()

# Persistent world model lives on disk
_WORLD_MODEL_PATH = Path("world_model.json")

# Maximum calibration history to include in context
_MAX_CALIBRATION_ENTRIES = 50

# Maximum world model size (chars) to keep prompts within context window
_MAX_WORLD_MODEL_CHARS = 8000


class WorldModel(BaseModel):
    """Persistent world state that evolves across trading cycles."""

    macro_outlook: str = ""
    geopolitical_state: str = ""
    key_beliefs: list[str] = Field(default_factory=list)
    cross_market_patterns: list[str] = Field(default_factory=list)
    active_themes: list[str] = Field(default_factory=list)
    last_updated: str = ""
    cycle_count: int = 0

    def summary(self) -> str:
        """Render a compact summary for the prompt context."""
        parts = []
        if self.macro_outlook:
            parts.append(f"MACRO OUTLOOK:\n{self.macro_outlook}")
        if self.geopolitical_state:
            parts.append(f"GEOPOLITICAL STATE:\n{self.geopolitical_state}")
        if self.key_beliefs:
            parts.append("KEY BELIEFS:\n" + "\n".join(f"- {b}" for b in self.key_beliefs[-10:]))
        if self.cross_market_patterns:
            parts.append("CROSS-MARKET PATTERNS:\n" + "\n".join(f"- {p}" for p in self.cross_market_patterns[-5:]))
        if self.active_themes:
            parts.append("ACTIVE THEMES: " + ", ".join(self.active_themes[-8:]))
        return "\n\n".join(parts) if parts else "(No world model yet — this is the first cycle.)"


class BatchAnalysisResult(BaseModel):
    """Result for one market from a batch analysis."""

    market_id: str
    probability: float = Field(ge=0, le=1)
    confidence: str = "MEDIUM"
    reasoning: str = ""
    key_factors: list[str] = Field(default_factory=list)
    cross_market_notes: str = ""


class StrategicAnalysis(BaseModel):
    """Full output from a strategic batch analysis."""

    markets: list[BatchAnalysisResult] = Field(default_factory=list)
    world_model_update: str = ""
    new_patterns: list[str] = Field(default_factory=list)
    new_beliefs: list[str] = Field(default_factory=list)
    retired_beliefs: list[str] = Field(default_factory=list)
    active_themes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

STRATEGIC_BATCH_PROMPT = """\
You are an elite superforecaster with a persistent memory. You maintain a \
world model that evolves as you analyze markets and receive feedback. \
You are being paid for ACCURACY, not for having opinions. Getting calibration \
right matters more than being interesting.

=== YOUR CURRENT WORLD MODEL ===
{world_model}

=== CALIBRATION FEEDBACK (how your past predictions resolved) ===
{calibration_feedback}

=== ANALYTICAL FRAMEWORK ===
For EACH market, apply these steps IN ORDER:

1. **OUTSIDE VIEW (Base Rate):** Before reading evidence, what's the base rate \
   for this type of event? Anchor here FIRST. Most people skip this step — \
   don't. If you can't find a specific base rate, use these defaults:
   - "Will X happen by date Y?" → typically 10-30% (most things don't happen)
   - "Will incumbent win?" → typically 60-70%
   - "Will policy/law change?" → typically 15-25%
   - "Will company do X?" → typically 20-40%

2. **INSIDE VIEW (Evidence):** Now read the evidence. How much should it \
   update your base rate? Be specific: "This evidence moves me from 20% to 35% \
   because..." NOT "the evidence suggests it's likely."

3. **FERMI DECOMPOSITION:** Break into sub-questions. Example: \
   "Will X happen?" = P(condition A) × P(condition B|A) × P(condition C|A,B). \
   If ANY condition is required, the joint probability is LOWER than any \
   individual condition. Most people forget this.

4. **EVIDENCE QUALITY CHECK:** Not all sources are equal. \
   Reuters/AP/official statements > news analysis > opinion pieces > social media. \
   Thin evidence = stay closer to base rate. One strong source > five weak ones.

5. **CROSS-MARKET REASONING:** How does this market connect to others in \
   this batch? If you estimate P(Iran war ends) = 20%, that constrains \
   P(oil below $80) and P(Fed cuts rates).

6. **FINAL CALIBRATION CHECK:**
   - Is your estimate between 5-95%? If not, you need EXTRAORDINARY evidence.
   - Would you bet real money at these odds? If not, adjust.
   - Are you confusing "interesting narrative" with "high probability"?

=== TODAY'S MARKETS ===
{markets_block}

=== YOUR RESPONSE ===
Respond with valid JSON only (no markdown, no commentary outside JSON):
{{
  "markets": [
    {{
      "market_id": "<id>",
      "probability": <float 0-1>,
      "confidence": "<LOW|MEDIUM|HIGH>",
      "reasoning": "<step-by-step: base rate → evidence update → decomposition → final>",
      "key_factors": ["<factor>", ...],
      "cross_market_notes": "<connections to other markets in this batch>"
    }},
    ...
  ],
  "world_model_update": "<concise update: what has changed since last cycle? \
    Focus on FACTS, not speculation. Include dates.>",
  "new_patterns": ["<cross-market patterns noticed>", ...],
  "new_beliefs": ["<updated beliefs — explicitly note if you're REVISING \
    a previous belief and why>", ...],
  "retired_beliefs": ["<beliefs from your world model that are now WRONG or \
    OUTDATED based on new evidence — name them explicitly>"],
  "active_themes": ["<key themes driving markets right now>", ...]
}}
"""

STRATEGIC_ADVERSARIAL_PROMPT = """\
You are a Red Team superforecaster reviewing a batch of predictions.

=== WORLD MODEL CONTEXT ===
{world_model}

=== PRIMARY ANALYST'S ESTIMATES ===
{estimates_block}

=== EVIDENCE ===
{evidence_block}

For each market, apply adversarial analysis:
1. Is the base rate correct? Find a BETTER reference class.
2. Pre-mortem: assume the estimate is wrong — why?
3. Decompose: what conditions must ALL be true?
4. Check for narrative bias, anchoring, overconfidence.
5. Look for cross-market inconsistencies: do these estimates contradict each other?

Respond with valid JSON only:
{{
  "markets": [
    {{
      "market_id": "<id>",
      "probability": <float 0-1>,
      "confidence": "<LOW|MEDIUM|HIGH>",
      "reasoning": "<adversarial findings>",
      "key_factors": ["<factor>", ...],
      "cross_market_notes": "<inconsistencies with other estimates>"
    }}
  ]
}}
"""


# ---------------------------------------------------------------------------
# Strategic Analyzer
# ---------------------------------------------------------------------------

class StrategicAnalyzer:
    """Maintains a persistent world model and does batch analysis with full context."""

    def __init__(self, settings, db: Database) -> None:
        self._settings = settings
        self._model = settings.nlp.model
        self._db = db
        self._world_model = self._load_world_model()
        self._daily_calls = 0
        self._daily_calls_date = ""

        # Initialize LLM ensemble if enabled
        self._ensemble = None
        if settings.llm_ensemble.enabled:
            from auramaur.nlp.ensemble_analyzer import EnsembleAnalyzer
            self._ensemble = EnsembleAnalyzer(settings, db)
            log.info(
                "strategic.ensemble_enabled",
                models=settings.llm_ensemble.models,
            )

    # ------------------------------------------------------------------
    # World Model Persistence
    # ------------------------------------------------------------------

    def _load_world_model(self) -> WorldModel:
        """Load world model from disk, or create a fresh one."""
        if _WORLD_MODEL_PATH.exists():
            try:
                data = json.loads(_WORLD_MODEL_PATH.read_text())
                return WorldModel(**data)
            except Exception as e:
                log.warning("strategic.world_model_load_error", error=str(e))
        return WorldModel()

    def _save_world_model(self) -> None:
        """Persist world model to disk."""
        _WORLD_MODEL_PATH.write_text(self._world_model.model_dump_json(indent=2))

    # ------------------------------------------------------------------
    # Calibration Feedback
    # ------------------------------------------------------------------

    async def _get_calibration_feedback(self) -> str:
        """Build calibration feedback from past predictions + resolutions.

        Includes per-category performance summary from the feedback loop
        (if available) plus individual prediction history.
        """
        parts: list[str] = []

        # Per-category performance summary from the feedback loop
        try:
            from auramaur.broker.feedback import PerformanceFeedback
            feedback = PerformanceFeedback(self._db)
            category_summary = await feedback.get_calibration_summary()
            if category_summary and "No per-category" not in category_summary:
                parts.append("=== PER-CATEGORY PERFORMANCE ===\n" + category_summary)
        except Exception as e:
            log.debug("strategic.feedback_import_error", error=str(e))

        # Individual prediction history
        rows = await self._db.fetchall(
            """SELECT s.market_id, m.question, s.claude_prob, s.market_prob, s.edge,
                      c.actual_outcome, c.predicted_prob
               FROM signals s
               LEFT JOIN markets m ON s.market_id = m.id
               LEFT JOIN calibration c ON s.market_id = c.market_id
               WHERE c.actual_outcome IS NOT NULL
               ORDER BY s.timestamp DESC
               LIMIT ?""",
            (_MAX_CALIBRATION_ENTRIES,),
        )

        if not rows and not parts:
            return "(No calibration data yet — predictions haven't resolved.)"

        if rows:
            lines = []
            correct = 0
            total = 0
            for row in rows:
                question = row["question"] or row["market_id"]
                predicted = row["predicted_prob"] or row["claude_prob"]
                actual = "YES" if row["actual_outcome"] else "NO"
                was_right = (predicted > 0.5 and row["actual_outcome"]) or \
                            (predicted <= 0.5 and not row["actual_outcome"])
                if was_right:
                    correct += 1
                total += 1
                icon = "correct" if was_right else "WRONG"
                lines.append(
                    f"- [{icon}] \"{question[:60]}\" — you said {predicted:.0%}, resolved {actual}"
                )

            accuracy = correct / total if total > 0 else 0
            header = f"Track record: {correct}/{total} ({accuracy:.0%} accuracy)\n"
            parts.append(header + "\n".join(lines[:_MAX_CALIBRATION_ENTRIES]))

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Batch Analysis
    # ------------------------------------------------------------------

    def _format_markets_block(
        self, markets: list[Market], evidence_map: dict[str, list[NewsItem]],
    ) -> str:
        """Format multiple markets with their evidence into a single block."""
        blocks = []
        for i, market in enumerate(markets, 1):
            evidence = evidence_map.get(market.id, [])
            ev_text = format_evidence(evidence) if evidence else "(No evidence found)"

            # Market microstructure context
            micro = (
                f"Current YES price: {market.outcome_yes_price:.1%} | "
                f"NO price: {market.outcome_no_price:.1%} | "
                f"Volume: ${market.volume:,.0f} | "
                f"Liquidity: ${market.liquidity:,.0f} | "
                f"Spread: {market.spread:.1%}"
            )

            block = (
                f"--- MARKET {i} (id: {market.id}) ---\n"
                f"Question: {market.question}\n"
                f"Description: {market.description[:300]}\n"
                f"Category: {market.category}\n"
                f"End date: {market.end_date.isoformat() if market.end_date else 'Unknown'}\n"
                f"Market data: {micro}\n"
                f"Evidence ({len(evidence)} items):\n{ev_text}\n"
            )
            blocks.append(block)

        return "\n".join(blocks)

    async def analyze_batch(
        self,
        markets: list[Market],
        evidence_map: dict[str, list[NewsItem]],
    ) -> StrategicAnalysis:
        """Analyze a batch of markets with full world-model context.

        This is the core method — it sends Claude the world model, calibration
        feedback, and all markets at once for cross-market reasoning.
        """
        if not markets:
            return StrategicAnalysis()

        # Build the prompt with full context
        world_model_text = self._world_model.summary()
        if len(world_model_text) > _MAX_WORLD_MODEL_CHARS:
            world_model_text = world_model_text[:_MAX_WORLD_MODEL_CHARS] + "\n...(truncated)"

        calibration = await self._get_calibration_feedback()
        markets_block = self._format_markets_block(markets, evidence_map)

        prompt = STRATEGIC_BATCH_PROMPT.format(
            world_model=world_model_text,
            calibration_feedback=calibration,
            markets_block=markets_block,
        )

        log.info(
            "strategic.batch_start",
            market_count=len(markets),
            world_model_cycle=self._world_model.cycle_count,
            prompt_chars=len(prompt),
        )

        # Primary batch analysis — optionally run through ensemble
        try:
            if self._ensemble and self._settings.llm_ensemble.enabled:
                result = await self._analyze_batch_ensemble(prompt, markets)
            else:
                raw = await self._call_claude_cli(prompt)
                result = self._parse_strategic_response(raw)
        except Exception as e:
            log.error("strategic.batch_failed", error=str(e))
            return StrategicAnalysis()

        # Update world model from Claude's response
        self._update_world_model(result)

        log.info(
            "strategic.batch_complete",
            markets_analyzed=len(result.markets),
            new_patterns=len(result.new_patterns),
            themes=result.active_themes,
        )

        return result

    async def analyze_batch_with_adversarial(
        self,
        markets: list[Market],
        evidence_map: dict[str, list[NewsItem]],
    ) -> StrategicAnalysis:
        """Batch analysis + adversarial review in one context-rich flow."""
        primary = await self.analyze_batch(markets, evidence_map)

        if not primary.markets or self._settings.nlp.skip_second_opinion:
            return primary

        # Build adversarial prompt with primary estimates
        estimates_block = "\n".join(
            f"- {m.market_id}: P(YES) = {m.probability:.0%} ({m.confidence}) — {m.reasoning[:100]}"
            for m in primary.markets
        )

        # Combine all evidence
        all_evidence = []
        for market in markets:
            all_evidence.extend(evidence_map.get(market.id, []))
        evidence_block = format_evidence(all_evidence[:30])  # Cap total evidence

        prompt = STRATEGIC_ADVERSARIAL_PROMPT.format(
            world_model=self._world_model.summary(),
            estimates_block=estimates_block,
            evidence_block=evidence_block,
        )

        try:
            raw = await self._call_claude_cli(prompt)
            adversarial = self._parse_strategic_response(raw)

            # Merge adversarial into primary results
            adversarial_map = {m.market_id: m for m in adversarial.markets}
            for primary_market in primary.markets:
                adv = adversarial_map.get(primary_market.market_id)
                if adv:
                    # Store adversarial as cross-market notes
                    primary_market.cross_market_notes += (
                        f" [Red team: {adv.probability:.0%} — {adv.reasoning[:100]}]"
                    )
        except Exception as e:
            log.warning("strategic.adversarial_failed", error=str(e))

        return primary

    # ------------------------------------------------------------------
    # Ensemble Batch Analysis
    # ------------------------------------------------------------------

    async def _analyze_batch_ensemble(
        self, prompt: str, markets: list[Market],
    ) -> StrategicAnalysis:
        """Run batch analysis through all ensemble models, blend per-market probabilities.

        Each model gets the same prompt. Their per-market probability estimates
        are blended using weights derived from per-model Brier scores.
        Non-probability fields (world model update, patterns, beliefs) come from
        the primary model (first in the list, typically opus).
        """
        model_responses = await self._call_ensemble(prompt)

        if not model_responses:
            raise RuntimeError("All ensemble models failed for batch analysis")

        # Parse each model's response
        parsed: list[tuple[str, StrategicAnalysis]] = []
        for model, raw in model_responses:
            try:
                sa = self._parse_strategic_response(raw)
                parsed.append((model, sa))
            except Exception as e:
                log.warning("strategic.ensemble_parse_failed", model=model, error=str(e))

        if not parsed:
            raise RuntimeError("All ensemble model responses failed to parse")

        # If only one model succeeded, use it directly
        if len(parsed) == 1:
            model, sa = parsed[0]
            # Record predictions
            for mr in sa.markets:
                category = self._get_market_category(mr.market_id, markets)
                await self._ensemble.record_model_prediction(
                    mr.market_id, model, mr.probability, category,
                )
            return sa

        # Load weights for blending
        weights = await self._ensemble.load_model_weights()

        # Use the primary model's result as the base (world model updates, etc.)
        primary_model, primary_result = parsed[0]

        # Build per-market probability maps: {market_id: [(model, prob, weight)]}
        market_probs: dict[str, list[tuple[str, float, float]]] = {}
        for model, sa in parsed:
            for mr in sa.markets:
                if mr.market_id not in market_probs:
                    market_probs[mr.market_id] = []
                w = weights.get(model, self._settings.llm_ensemble.default_weight)
                market_probs[mr.market_id].append((model, mr.probability, w))

        # Blend per-market probabilities
        for market_result in primary_result.markets:
            mid = market_result.market_id
            if mid in market_probs:
                entries = market_probs[mid]
                total_w = sum(w for _, _, w in entries)
                if total_w > 0:
                    blended_prob = sum(p * w for _, p, w in entries) / total_w
                else:
                    blended_prob = sum(p for _, p, _ in entries) / len(entries)

                # Log the blend
                individual = {m: round(p, 4) for m, p, _ in entries}
                spread = max(p for _, p, _ in entries) - min(p for _, p, _ in entries)

                log.info(
                    "strategic.ensemble_blend",
                    market_id=mid,
                    blended=round(blended_prob, 4),
                    individual=individual,
                    spread=round(spread, 4),
                )

                # Append ensemble info to cross_market_notes
                model_notes = ", ".join(
                    f"{m}: {p:.0%} (w={w:.2f})" for m, p, w in entries
                )
                market_result.cross_market_notes += (
                    f" [Ensemble: {model_notes} -> blended {blended_prob:.0%}]"
                )

                # Record per-model predictions for Brier tracking
                category = self._get_market_category(mid, markets)
                for model, prob, _ in entries:
                    await self._ensemble.record_model_prediction(
                        mid, model, prob, category,
                    )

                market_result.probability = blended_prob

        log.info(
            "strategic.ensemble_batch_complete",
            models=[m for m, _ in parsed],
            markets_blended=len(market_probs),
        )

        return primary_result

    @staticmethod
    def _get_market_category(market_id: str, markets: list[Market]) -> str:
        """Look up a market's category from the market list."""
        for m in markets:
            if m.id == market_id:
                return getattr(m, "category", "") or ""
        return ""

    # ------------------------------------------------------------------
    # World Model Updates
    # ------------------------------------------------------------------

    def _update_world_model(self, result: StrategicAnalysis) -> None:
        """Update the persistent world model from Claude's analysis."""
        wm = self._world_model

        if result.world_model_update:
            wm.macro_outlook = result.world_model_update
            wm.geopolitical_state = ""

        # Retire stale beliefs BEFORE adding new ones
        if result.retired_beliefs:
            retired_lower = {b.lower() for b in result.retired_beliefs}
            wm.key_beliefs = [
                b for b in wm.key_beliefs
                if not any(r in b.lower() for r in retired_lower)
            ]
            log.info("strategic.beliefs_retired", count=len(result.retired_beliefs),
                     retired=result.retired_beliefs)

        if result.new_beliefs:
            wm.key_beliefs.extend(result.new_beliefs)
            wm.key_beliefs = wm.key_beliefs[-15:]

        if result.new_patterns:
            wm.cross_market_patterns.extend(result.new_patterns)
            wm.cross_market_patterns = wm.cross_market_patterns[-10:]

        if result.active_themes:
            wm.active_themes = result.active_themes

        wm.cycle_count += 1
        wm.last_updated = datetime.now(timezone.utc).isoformat()

        self._save_world_model()
        log.info(
            "strategic.world_model_updated",
            cycle=wm.cycle_count,
            beliefs=len(wm.key_beliefs),
            patterns=len(wm.cross_market_patterns),
            themes=wm.active_themes,
        )

    # ------------------------------------------------------------------
    # CLI + Parsing
    # ------------------------------------------------------------------

    async def _call_claude_cli(self, prompt: str, *, model: str | None = None) -> str:
        """Call Claude CLI — same as ClaudeAnalyzer but with longer timeout for batch.

        Args:
            prompt: The prompt to send.
            model: Optional model override (used by ensemble to call different models).
        """
        import asyncio
        from datetime import date

        today = date.today().isoformat()
        if self._daily_calls_date != today:
            self._daily_calls = 0
            self._daily_calls_date = today

        use_model = model or self._model

        max_attempts = 3
        backoff = [10, 20, 40]
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "claude", "-p", prompt,
                    "--output-format", "text",
                    "--model", use_model,
                    "--effort", "max",
                    "--max-turns", "3",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                # Longer timeout for deep reasoning (up to 8 min)
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=480)

                if proc.returncode != 0:
                    err = stderr.decode().strip()
                    raise RuntimeError(f"Claude CLI ({use_model}) failed (rc={proc.returncode}): {err}")

                self._daily_calls += 1
                log.info("strategic.claude_call", model=use_model, daily_calls=self._daily_calls)
                return stdout.decode().strip()

            except (TimeoutError, asyncio.TimeoutError, RuntimeError) as e:
                last_error = e
                if attempt < max_attempts:
                    await asyncio.sleep(backoff[attempt - 1])

        raise last_error  # type: ignore[misc]

    async def _call_ensemble(self, prompt: str) -> list[tuple[str, str]]:
        """Call all ensemble models in parallel, returning [(model, raw_response), ...].

        Falls back to single-model call if ensemble is disabled.
        """
        import asyncio

        models = self._settings.llm_ensemble.models
        tasks = [self._call_claude_cli(prompt, model=m) for m in models]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        model_responses: list[tuple[str, str]] = []
        for model, result in zip(models, results):
            if isinstance(result, Exception):
                log.warning("strategic.ensemble_model_failed", model=model, error=str(result))
            else:
                model_responses.append((model, result))

        return model_responses

    @staticmethod
    def _parse_strategic_response(text: str) -> StrategicAnalysis:
        """Parse Claude's batch response into structured results."""
        # Strip markdown fences
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fenced:
            text = fenced.group(1)

        text = text.strip()

        # Try direct parse
        try:
            data = json.loads(text)
            return StrategicAnalysis(**data)
        except (json.JSONDecodeError, Exception):
            pass

        # Extract JSON object
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                data = json.loads(match.group(0))
                return StrategicAnalysis(**data)
            except (json.JSONDecodeError, Exception):
                pass

        log.error("strategic.parse_failed", text_preview=text[:300])
        return StrategicAnalysis()
