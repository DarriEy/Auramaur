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

import structlog
from pydantic import BaseModel, Field

from auramaur.data_sources.base import NewsItem
from auramaur.db.database import Database
from auramaur.exchange.models import Market
from auramaur.nlp.cache import NLPCache, coarse_evidence_digest, make_cache_key
from auramaur.nlp.prompts import format_evidence
from auramaur.runtime import state_dir

log = structlog.get_logger()

# Persistent world model lives on disk, anchored to the state dir — a bare
# relative path resolves against the container's read-only CWD (agent_analyzer
# learned this the same way).
_WORLD_MODEL_PATH = state_dir() / "world_model.json"

# Maximum calibration history to include in context
_MAX_CALIBRATION_ENTRIES = 50

# Maximum world model size (chars) to keep prompts within context window
_MAX_WORLD_MODEL_CHARS = 8000


class EntityRelation(BaseModel):
    """A tracked entity and its relations to other entities/markets."""

    state: str = ""  # Current understanding of this entity
    relations: list[str] = Field(default_factory=list)  # How it connects to others
    market_ids: list[str] = Field(default_factory=list)  # Markets this entity touches


class WorldModel(BaseModel):
    """Persistent world state that evolves across trading cycles.

    The world model is structured relationally: entities exist through
    their relations to other entities, not as isolated facts.  The
    entity_graph is the primary structure; beliefs and patterns are
    derived from it.
    """

    macro_outlook: str = ""
    geopolitical_state: str = ""
    key_beliefs: list[str] = Field(default_factory=list)
    cross_market_patterns: list[str] = Field(default_factory=list)
    active_themes: list[str] = Field(default_factory=list)
    entity_graph: dict[str, EntityRelation] = Field(default_factory=dict)
    last_updated: str = ""
    cycle_count: int = 0

    def summary(self) -> str:
        """Render a compact summary for the prompt context."""
        parts = []
        if self.macro_outlook:
            parts.append(f"MACRO STATE:\n{self.macro_outlook}")

        # Entity graph — the primary relational structure. Surface the most
        # connected entities (by markets touched + relations), not whichever
        # happened to be inserted last.
        if self.entity_graph:
            ranked = sorted(
                self.entity_graph.items(),
                key=lambda kv: (len(kv[1].market_ids), len(kv[1].relations)),
                reverse=True,
            )[:12]
            entity_lines = []
            for name, entity in ranked:
                rel_str = "; ".join(entity.relations[-3:]) if entity.relations else "no tracked relations"
                entity_lines.append(f"- **{name}**: {entity.state[:120]} [{rel_str}]")
            parts.append("ENTITY GRAPH:\n" + "\n".join(entity_lines))

        if self.key_beliefs:
            parts.append("KEY BELIEFS:\n" + "\n".join(f"- {b}" for b in self.key_beliefs[-10:]))
        if self.cross_market_patterns:
            parts.append("RELATIONAL PATTERNS:\n" + "\n".join(f"- {p}" for p in self.cross_market_patterns[-8:]))
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
    # Adversarial (red-team) second opinion — populated by
    # analyze_batch_with_adversarial so the divergence can persist to the
    # signals table (readiness criterion 8) instead of living only in the
    # cross_market_notes display string.
    second_opinion_prob: float | None = None
    divergence: float | None = None


class StrategicAnalysis(BaseModel):
    """Full output from a strategic batch analysis."""

    markets: list[BatchAnalysisResult] = Field(default_factory=list)
    entity_graph: dict = Field(default_factory=dict)
    world_model_update: str = ""
    new_patterns: list[str] = Field(default_factory=list)
    new_beliefs: list[str] = Field(default_factory=list)
    retired_beliefs: list[str] = Field(default_factory=list)
    active_themes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# The static half of the strategic prompt — role, framework, and response
# schema. Kept identical across every call so the CLI's prompt-cache can reuse
# it as a stable prefix (passed via --append-system-prompt with
# --exclude-dynamic-system-prompt-sections). The per-cycle world model,
# calibration, and markets go in the user prompt below.
STRATEGIC_SYSTEM_PROMPT = """\
You are an elite superforecaster with a persistent memory.  You maintain \
a relational world model — entities connected by relations, not isolated \
facts.  You are being paid for ACCURACY, not for having opinions.

=== ANALYTICAL FRAMEWORK ===
You analyze markets in THREE PHASES.  The relational phase comes FIRST \
because reality is relational — probabilities emerge from the web of \
entities and forces, not from markets considered in isolation.

**PHASE 1 — RELATIONAL MAPPING (do this BEFORE estimating any probability)**

Scan ALL markets in this batch and identify:
- What ENTITIES appear across multiple markets? (people, institutions, \
  forces, events)
- What RELATIONS connect them? (causal, constraining, enabling, opposing)
- What CLUSTERS of markets share a common entity or root cause?
- What TENSIONS exist? (if market A implies X but market B implies not-X)

This phase is where your edge comes from.  Markets price each question \
independently.  Reality is relational.  When the market treats coupled \
events as independent, one side is mispriced.

**PHASE 2 — EVIDENCE-INFORMED ESTIMATION (per market, informed by Phase 1)**

For each market:
1. Base rate (outside view) — anchor here FIRST
   - "Will X happen by date Y?" → typically 10-30%
   - "Will incumbent win?" → typically 60-70%
   - "Will policy/law change?" → typically 15-25%
2. Evidence update — be specific: "this moves me from 20% to 35% because..."
3. Fermi decomposition — P(A) × P(B|A) × P(C|A,B).  Joint probability is \
   LOWER than any individual condition.
4. Relational constraint — does your estimate for THIS market contradict \
   your estimate for a RELATED market?  Resolve the contradiction.

Evidence quality: Reuters/AP/official > analysis > opinion > social media. \
Thin evidence = stay closer to base rate.

**PHASE 3 — COHERENCE CHECK**

Review ALL estimates together:
- Are they jointly consistent?  (P(war ends) = 70% but P(oil stays high) = 80%?)
- Would the entity graph support this set of probabilities simultaneously?
- Are you confusing "interesting narrative" with "high probability"?
- Would you bet real money at these odds?

=== YOUR RESPONSE ===
Respond with valid JSON only (no markdown, no commentary outside JSON):
{
  "entity_graph": {
    "<entity_name>": {
      "state": "<current understanding of this entity>",
      "relations": ["<relation to other entity or force>", ...],
      "market_ids": ["<ids of markets this entity touches>"]
    }
  },
  "markets": [
    {
      "market_id": "<id>",
      "probability": <float 0-1>,
      "confidence": "<LOW|MEDIUM|HIGH>",
      "reasoning": "<relational: which entities/relations inform this + base rate → update → final>",
      "key_factors": ["<factor>", ...],
      "cross_market_notes": "<how this estimate constrains or is constrained by other estimates>"
    },
    ...
  ],
  "world_model_update": "<concise update: what has changed since last cycle? \
    Focus on FACTS, not speculation. Include dates.>",
  "new_patterns": ["<relational patterns: entity X's relation to Y changed because...>", ...],
  "new_beliefs": ["<updated beliefs — explicitly note if you're REVISING \
    a previous belief and why>", ...],
  "retired_beliefs": ["<beliefs from your world model that are now WRONG or \
    OUTDATED based on new evidence — name them explicitly>"],
  "active_themes": ["<key themes driving markets right now>", ...]
}
"""

# The per-cycle (dynamic) half of the strategic prompt. Sent as the user
# message; the static framework + response schema live in
# STRATEGIC_SYSTEM_PROMPT above so the cache prefix stays stable.
STRATEGIC_BATCH_PROMPT = """\
=== YOUR CURRENT WORLD MODEL ===
{world_model}

=== CALIBRATION FEEDBACK (how your past predictions resolved) ===
{calibration_feedback}

=== TODAY'S MARKETS ===
{markets_block}

Analyze every market above using your three-phase framework and respond with \
the JSON object exactly as specified in your instructions.
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
        self._cache = NLPCache(db)
        self._world_model = self._load_world_model()
        self._last_batch_at = None  # throttle: last batch+adversarial LLM run

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
            if self._settings.nlp.calibration_buckets:
                parts.append(self._calibration_curve(rows))
            else:
                parts.append(self._calibration_rows(rows))

        return "\n\n".join(parts)

    @staticmethod
    def _calibration_curve(rows: list) -> str:
        """Compact reliability curve: per-band over/under-confidence + top misses.

        Far higher signal-per-token than dumping every resolved outcome — it
        tells the model exactly where its probabilities have been biased.
        """
        samples: list[tuple[float, float, str]] = []  # (predicted, actual, question)
        for row in rows:
            predicted = row["predicted_prob"] or row["claude_prob"]
            if predicted is None:
                continue
            actual = 1.0 if row["actual_outcome"] else 0.0
            samples.append((float(predicted), actual, row["question"] or row["market_id"]))

        if not samples:
            return "(No resolved predictions yet.)"

        n = len(samples)
        correct = sum(1 for p, a, _ in samples if (p > 0.5) == (a > 0.5))
        brier = sum((p - a) ** 2 for p, a, _ in samples) / n

        log.info(
            "strategic.calibration_snapshot",
            resolved=n,
            directional_accuracy=round(correct / n, 3),
            brier=round(brier, 4),
        )

        bands = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        band_lines: list[str] = []
        for lo, hi in bands:
            bucket = [(p, a) for p, a, _ in samples if lo <= p < hi]
            if not bucket:
                continue
            mean_pred = sum(p for p, _ in bucket) / len(bucket)
            hit_rate = sum(a for _, a in bucket) / len(bucket)
            gap = mean_pred - hit_rate
            if abs(gap) < 0.05:
                bias = "well-calibrated"
            elif gap > 0:
                bias = f"OVERconfident by {gap:+.0%} → shade toward 0.5"
            else:
                bias = f"UNDERconfident by {gap:+.0%} → push away from 0.5"
            band_lines.append(
                f"- {lo:.0%}-{hi if hi <= 1 else 1:.0%}: said ~{mean_pred:.0%}, "
                f"resolved YES {hit_rate:.0%} (n={len(bucket)}) — {bias}"
            )

        # Largest errors as concrete cautionary examples.
        worst = sorted(samples, key=lambda s: abs(s[0] - s[1]), reverse=True)[:3]
        miss_lines = [
            f"- \"{q[:60]}\": said {p:.0%}, resolved {'YES' if a > 0.5 else 'NO'}"
            for p, a, q in worst
        ]

        return (
            f"Track record: {correct}/{n} directional ({correct / n:.0%}), "
            f"Brier {brier:.3f} (lower is better)\n"
            "Reliability by confidence band:\n" + "\n".join(band_lines) +
            "\nBiggest misses (avoid repeating):\n" + "\n".join(miss_lines)
        )

    @staticmethod
    def _calibration_rows(rows: list) -> str:
        """Legacy per-row calibration dump (used when buckets are disabled)."""
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
        return header + "\n".join(lines[:_MAX_CALIBRATION_ENTRIES])

    # ------------------------------------------------------------------
    # Batch Analysis
    # ------------------------------------------------------------------

    def _format_markets_block(
        self, markets: list[Market], evidence_map: dict[str, list[NewsItem]],
    ) -> str:
        """Format multiple markets with compressed evidence.

        Instead of raw articles, evidence is compressed into structured
        signal dimensions: facts, directional signals, temporal dynamics,
        source consensus, and key excerpts.
        """
        from auramaur.nlp.evidence_compressor import compress_evidence

        blocks = []
        total_raw = 0
        total_ev_chars = 0
        markets_with_evidence = 0
        for i, market in enumerate(markets, 1):
            evidence = evidence_map.get(market.id, [])

            # Compress evidence into structured dimensions
            if evidence:
                ev_text = compress_evidence(
                    market.question,
                    market.description,
                    evidence,
                    max_chars=1500,  # ~375 tokens per market
                )
                total_raw += len(evidence)
                total_ev_chars += len(ev_text)
                markets_with_evidence += 1
            else:
                ev_text = "(No evidence found)"

            # Compact market context
            micro = (
                f"YES: {market.outcome_yes_price:.0%} | "
                f"Vol: ${market.volume:,.0f} | "
                f"Liq: ${market.liquidity:,.0f}"
            )

            block = (
                f"--- MARKET {i} (id: {market.id}) ---\n"
                f"Q: {market.question}\n"
                f"Resolution criteria: {market.description[:400]}\n"
                f"Cat: {market.category} | {micro} | "
                f"End: {market.end_date.strftime('%Y-%m-%d') if market.end_date else '?'}\n"
                f"Evidence:\n{ev_text}\n"
            )
            blocks.append(block)

        # Prompt-signal logging: how much evidence we ingested vs how compactly
        # it landed in the prompt. Lets us A/B the info-content changes.
        log.info(
            "strategic.evidence_signal",
            markets=len(markets),
            markets_with_evidence=markets_with_evidence,
            raw_evidence_items=total_raw,
            evidence_chars=total_ev_chars,
            avg_items_per_market=round(total_raw / markets_with_evidence, 1) if markets_with_evidence else 0,
            avg_chars_per_market=round(total_ev_chars / markets_with_evidence) if markets_with_evidence else 0,
        )

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

        # Lever 5: reuse fresh, price-stable cached results and only send the
        # markets that actually need (re)analysis to Claude. Quiet cycles where
        # everything is cached skip the call entirely.
        cached_results, to_analyze, cache_keys = await self._partition_cached(
            markets, evidence_map,
        )
        if cached_results:
            log.info(
                "strategic.cache_partition",
                cached=len(cached_results),
                to_analyze=len(to_analyze),
            )
        if not to_analyze:
            log.info("strategic.batch_all_cached", markets=len(cached_results))
            return StrategicAnalysis(markets=cached_results)

        # Build the (dynamic) user prompt; the static framework is sent once as a
        # cacheable system prompt (Lever 4).
        world_model_text = self._world_model.summary()
        if len(world_model_text) > _MAX_WORLD_MODEL_CHARS:
            world_model_text = world_model_text[:_MAX_WORLD_MODEL_CHARS] + "\n...(truncated)"

        calibration = await self._get_calibration_feedback()
        markets_block = self._format_markets_block(to_analyze, evidence_map)

        prompt = STRATEGIC_BATCH_PROMPT.format(
            world_model=world_model_text,
            calibration_feedback=calibration,
            markets_block=markets_block,
        )
        system_prompt = STRATEGIC_SYSTEM_PROMPT

        log.info(
            "strategic.batch_start",
            market_count=len(to_analyze),
            world_model_cycle=self._world_model.cycle_count,
            prompt_chars=len(prompt),
        )

        # Lever 1 + 2: always run a single primary call; only fan out to the
        # ensemble's extra models when the primary already found a tradeable edge.
        try:
            raw = await self._call_llm(
                prompt,
                system_prompt=system_prompt,
                effort=self._settings.nlp.effort_primary,
            )
            result = self._parse_strategic_response(raw)

            ens = self._settings.llm_ensemble
            if self._ensemble and ens.enabled:
                if not ens.gate_on_edge or self._batch_has_edge(result, to_analyze):
                    result = await self._blend_ensemble(
                        prompt, to_analyze, result, system_prompt=system_prompt,
                    )
                else:
                    log.info("strategic.ensemble_skipped_no_edge", markets=len(to_analyze))
        except Exception as e:
            log.error("strategic.batch_failed", error=str(e))
            # Still surface any cached results we had.
            return StrategicAnalysis(markets=cached_results)

        # Update world model from Claude's response (analyzed markets only)
        self._update_world_model(result)

        # Lever 5: cache the freshly-analyzed per-market results.
        await self._store_results(result, to_analyze, cache_keys)

        # Merge reused cached results back in.
        result.markets = list(result.markets) + cached_results

        log.info(
            "strategic.batch_complete",
            markets_analyzed=len(result.markets),
            from_cache=len(cached_results),
            new_patterns=len(result.new_patterns),
            themes=result.active_themes,
        )

        return result

    async def _partition_cached(
        self,
        markets: list[Market],
        evidence_map: dict[str, list[NewsItem]],
    ) -> tuple[list[BatchAnalysisResult], list[Market], dict[str, str]]:
        """Split markets into (reusable cached results, markets to analyze, keys).

        ``cache_keys`` maps market_id -> cache_key for every market, so analyzed
        results can be written back with the same key.
        """
        cache_keys: dict[str, str] = {}
        cached: list[BatchAnalysisResult] = []
        to_analyze: list[Market] = []

        for market in markets:
            evidence = evidence_map.get(market.id, [])
            key = make_cache_key(market.question, coarse_evidence_digest(evidence))
            cache_keys[market.id] = key

            if not self._settings.nlp.strategic_cache_enabled:
                to_analyze.append(market)
                continue

            hit = await self._cache.get(key, current_price=market.outcome_yes_price)
            if hit is not None:
                try:
                    cached.append(BatchAnalysisResult(**hit))
                    continue
                except Exception as e:
                    log.warning("strategic.cache_decode_failed", market_id=market.id, error=str(e))
            to_analyze.append(market)

        return cached, to_analyze, cache_keys

    async def _store_results(
        self,
        result: StrategicAnalysis,
        analyzed: list[Market],
        cache_keys: dict[str, str],
    ) -> None:
        """Write freshly-analyzed per-market results into the NLP cache."""
        if not self._settings.nlp.strategic_cache_enabled:
            return
        ttl = self._settings.nlp.cache_ttl_breaking_seconds
        price_by_id = {m.id: (m.outcome_yes_price or 0.0) for m in analyzed}
        for mr in result.markets:
            key = cache_keys.get(mr.market_id)
            if not key:
                continue
            try:
                await self._cache.put(
                    key, mr.market_id, mr.model_dump(), ttl,
                    market_price=price_by_id.get(mr.market_id, 0.0),
                )
            except Exception as e:
                log.warning("strategic.cache_store_failed", market_id=mr.market_id, error=str(e))

    async def analyze_batch_with_adversarial(
        self,
        markets: list[Market],
        evidence_map: dict[str, list[NewsItem]],
    ) -> StrategicAnalysis:
        """Batch analysis + adversarial review in one context-rich flow."""
        # Cadence throttle: the engine calls this per scan cycle, but with
        # directional signals paper-forced the marginal value of re-batching
        # every ~10 min is low. Outside the interval, serve only fresh cached
        # results and skip BOTH LLM calls (batch + adversarial).
        interval = getattr(self._settings.nlp, "strategic_min_interval_seconds", 0)
        if interval and markets and self._last_batch_at is not None:
            elapsed = (datetime.now(timezone.utc) - self._last_batch_at).total_seconds()
            if elapsed < interval:
                cached, _, _ = await self._partition_cached(markets, evidence_map)
                return StrategicAnalysis(markets=cached)
        if markets:
            self._last_batch_at = datetime.now(timezone.utc)

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
            raw = await self._call_llm(
                prompt, effort=self._settings.nlp.effort_adversarial,
            )
            adversarial = self._parse_strategic_response(raw)

            # Merge adversarial into primary results
            adversarial_map = {m.market_id: m for m in adversarial.markets}
            for primary_market in primary.markets:
                adv = adversarial_map.get(primary_market.market_id)
                if adv:
                    # Persist the red-team estimate structurally (feeds the
                    # signals table -> readiness divergence criterion) ...
                    primary_market.second_opinion_prob = adv.probability
                    primary_market.divergence = abs(
                        primary_market.probability - adv.probability
                    )
                    # ... and keep the human-readable note.
                    primary_market.cross_market_notes += (
                        f" [Red team: {adv.probability:.0%} — {adv.reasoning[:100]}]"
                    )
        except Exception as e:
            log.warning("strategic.adversarial_failed", error=str(e))

        return primary

    # ------------------------------------------------------------------
    # Ensemble Batch Analysis
    # ------------------------------------------------------------------

    def _batch_has_edge(self, result: StrategicAnalysis, markets: list[Market]) -> bool:
        """True if any analyzed market diverges from its price by the ensemble threshold.

        Used to gate the (expensive) extra ensemble models: quiet cycles with no
        tradeable edge run a single model; only cycles where the primary already
        found edge pay for the ensemble.
        """
        threshold = self._settings.llm_ensemble.edge_threshold_pct / 100.0
        price_by_id = {m.id: (m.outcome_yes_price or 0.5) for m in markets}
        for mr in result.markets:
            price = price_by_id.get(mr.market_id, 0.5)
            if abs(mr.probability - price) >= threshold:
                return True
        return False

    async def _blend_ensemble(
        self,
        prompt: str,
        markets: list[Market],
        primary_result: StrategicAnalysis,
        *,
        system_prompt: str | None = None,
    ) -> StrategicAnalysis:
        """Run the non-primary ensemble models and blend them with the primary result.

        The primary model's single call has already happened (and is reused here
        as the base for world-model updates), so this only spends calls on the
        remaining models.
        """
        primary_model = self._model
        extra = [m for m in self._settings.llm_ensemble.models if m != primary_model]

        parsed: list[tuple[str, StrategicAnalysis]] = [(primary_model, primary_result)]
        if extra:
            raw_extra = await self._call_ensemble_extra(
                prompt, extra, system_prompt=system_prompt,
            )
            for model, raw in raw_extra:
                try:
                    parsed.append((model, self._parse_strategic_response(raw)))
                except Exception as e:
                    log.warning("strategic.ensemble_parse_failed", model=model, error=str(e))

        return await self._blend_results(parsed, markets)

    async def _blend_results(
        self, parsed: list[tuple[str, StrategicAnalysis]], markets: list[Market],
    ) -> StrategicAnalysis:
        """Blend per-market probabilities across model responses by Brier weight.

        Non-probability fields (world model update, patterns, beliefs) come from
        the primary model (first in ``parsed``, typically opus). Works for a
        single response too (records its predictions, returns it unchanged).
        """
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

        # Merge entity graph — entities are the primary relational structure
        if result.entity_graph:
            for name, entity_data in result.entity_graph.items():
                if isinstance(entity_data, dict):
                    entity = EntityRelation(
                        state=entity_data.get("state", ""),
                        relations=entity_data.get("relations", []),
                        market_ids=entity_data.get("market_ids", []),
                    )
                elif isinstance(entity_data, EntityRelation):
                    entity = entity_data
                else:
                    continue

                if name in wm.entity_graph:
                    # Update existing entity — merge relations, update state
                    existing = wm.entity_graph[name]
                    existing.state = entity.state or existing.state
                    seen = set(existing.relations)
                    for rel in entity.relations:
                        if rel not in seen:
                            existing.relations.append(rel)
                    existing.relations = existing.relations[-8:]
                    seen_ids = set(existing.market_ids)
                    for mid in entity.market_ids:
                        if mid not in seen_ids:
                            existing.market_ids.append(mid)
                    existing.market_ids = existing.market_ids[-20:]
                else:
                    wm.entity_graph[name] = entity

            # Cap entity graph size — keep most recently updated
            if len(wm.entity_graph) > 30:
                # Keep the 30 entities with the most market connections
                sorted_entities = sorted(
                    wm.entity_graph.items(),
                    key=lambda x: len(x[1].market_ids),
                    reverse=True,
                )
                wm.entity_graph = dict(sorted_entities[:30])

            log.info(
                "strategic.entity_graph_updated",
                entities=len(wm.entity_graph),
                new_entities=len(result.entity_graph),
            )

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
            wm.cross_market_patterns = wm.cross_market_patterns[-15:]

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
            entities=len(wm.entity_graph),
            themes=wm.active_themes,
        )

    # ------------------------------------------------------------------
    # CLI + Parsing
    # ------------------------------------------------------------------

    async def _call_claude_cli(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        effort: str | None = None,
    ) -> str:
        """Call Claude CLI — same as ClaudeAnalyzer but with longer timeout for batch.

        Args:
            prompt: The (per-cycle, dynamic) user prompt to send.
            model: Optional model override (used by ensemble to call different models).
            system_prompt: Optional static system prompt. Appended via
                ``--append-system-prompt`` with ``--exclude-dynamic-system-prompt-sections``
                so the CLI's prompt cache can reuse it as a stable prefix.
            effort: CLI effort level (low|medium|high|max). Defaults to
                ``nlp.effort_primary``.
        """
        import asyncio
        from auramaur.subprocess_security import analysis_subprocess_env

        use_model = model or self._model
        use_effort = effort or self._settings.nlp.effort_primary

        cmd = [
            "claude", "-p", prompt,
            "--output-format", "text",
            "--model", use_model,
            "--effort", use_effort,
        ]
        if system_prompt:
            # Keep the default system prompt (so tooling context survives) but
            # strip its dynamic sections, which would otherwise bust the cache
            # prefix on every call.
            cmd += [
                "--append-system-prompt", system_prompt,
                "--exclude-dynamic-system-prompt-sections",
            ]

        max_attempts = 3
        backoff = [10, 20, 40]
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=analysis_subprocess_env(),
                )
                # Longer timeout for deep reasoning (up to 8 min)
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=480)

                if proc.returncode != 0:
                    err = stderr.decode().strip()
                    raise RuntimeError(f"Claude CLI ({use_model}) failed (rc={proc.returncode}): {err}")

                from auramaur.nlp import call_budget
                log.info(
                    "strategic.claude_call",
                    model=use_model,
                    effort=use_effort,
                    cached_prefix=bool(system_prompt),
                    daily_calls=call_budget.record_call(),
                )
                return stdout.decode().strip()

            except (TimeoutError, asyncio.TimeoutError, RuntimeError) as e:
                last_error = e
                if attempt < max_attempts:
                    await asyncio.sleep(backoff[attempt - 1])

        raise last_error  # type: ignore[misc]

    async def _call_llm(
        self, prompt: str, *, system_prompt: str | None = None, effort: str | None = None,
    ) -> str:
        """Single-model call, routed to Gemini off-hours / when Claude budget low."""
        from functools import partial

        from auramaur.nlp.llm_router import route

        # route() calls the fn with just the prompt; bind the extra CLI options.
        claude_fn = partial(self._call_claude_cli, system_prompt=system_prompt, effort=effort)
        from auramaur.nlp import call_budget
        return await route(self._settings, call_budget.calls_today(), prompt, claude_fn)

    async def _call_ensemble_extra(
        self, prompt: str, extra_models: list[str], *, system_prompt: str | None = None,
    ) -> list[tuple[str, str]]:
        """Call the non-primary ensemble models in parallel at secondary effort.

        Returns [(model, raw_response), ...] for the models that succeeded.
        """
        import asyncio

        effort = self._settings.nlp.effort_ensemble_secondary
        tasks = [
            self._call_claude_cli(prompt, model=m, system_prompt=system_prompt, effort=effort)
            for m in extra_models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        model_responses: list[tuple[str, str]] = []
        for model, result in zip(extra_models, results):
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

        def _norm_item(it: dict) -> dict:
            # The model varies field names vs the schema (observed: id/p_yes and
            # a numeric confidence). Map common aliases so items validate instead
            # of being dropped — the dominant cause of strategic.parse_failed.
            d = dict(it)
            if "market_id" not in d:
                for alt in ("id", "ticker", "marketId"):
                    if alt in d:
                        d["market_id"] = d[alt]
                        break
            if "probability" not in d:
                for alt in ("p_yes", "prob", "fair_prob", "yes_prob", "p"):
                    if alt in d:
                        d["probability"] = d[alt]
                        break
            c = d.get("confidence")
            if isinstance(c, (int, float)):  # 0-1 float -> tier label
                d["confidence"] = "HIGH" if c >= 0.7 else "LOW" if c < 0.4 else "MEDIUM"
            return d

        def _coerce(data):
            # The model frequently returns a BARE ARRAY of per-market results
            # ([{...}, ...]) instead of the full object ({"markets": [...], ...}).
            # Treat a list as the markets payload (normalizing item keys) —
            # otherwise StrategicAnalysis(**list) blew up and every batch was
            # discarded (100% strategic.parse_failed, pure wasted LLM spend).
            if isinstance(data, list):
                items = [_norm_item(it) for it in data if isinstance(it, dict)]
                return StrategicAnalysis(markets=items)
            if isinstance(data.get("markets"), list):
                data = dict(data)
                data["markets"] = [_norm_item(it) for it in data["markets"]
                                   if isinstance(it, dict)]
            return StrategicAnalysis(**data)

        # Try direct parse (object or array)
        try:
            return _coerce(json.loads(text))
        except (json.JSONDecodeError, Exception):
            pass

        # Extract the first JSON array or object embedded in prose.
        for pat in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
            match = re.search(pat, text)
            if match:
                try:
                    return _coerce(json.loads(match.group(0)))
                except (json.JSONDecodeError, Exception):
                    continue

        # Recoverable: caller falls back to an empty StrategicAnalysis. The LLM
        # occasionally returns prose instead of JSON — worth visibility, not an
        # error-level alarm.
        log.warning("strategic.parse_failed", text_preview=text[:300])
        return StrategicAnalysis()
