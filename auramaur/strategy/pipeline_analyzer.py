"""Pipeline-based market analyzer — wraps the current data→NLP→signal flow."""

from __future__ import annotations

import os
import fcntl
import tempfile
import time

import structlog

from auramaur.data_sources.aggregator import Aggregator
from auramaur.db.database import Database
from auramaur.exchange.models import Confidence, Market, OrderSide, Signal
from auramaur.exchange.protocols import ExchangeClient
from auramaur.monitoring.display import show_analysis, show_analyzing, show_evidence
from auramaur.nlp.analyzer import ClaudeAnalyzer
from auramaur.nlp.cache import NLPCache
from auramaur.nlp.calibration import CalibrationTracker
from auramaur.strategy.protocols import TradeCandidate
from auramaur.strategy.signals import detect_edge, POLYMARKET_FEE_PCT

log = structlog.get_logger()

# Shared directory for cross-instance market claim locks
_CLAIM_DIR = os.path.join(tempfile.gettempdir(), "auramaur_claims")
os.makedirs(_CLAIM_DIR, exist_ok=True)


class PipelineAnalyzer:
    """Implements MarketAnalyzer using the existing pipeline.

    Delegates to StrategicAnalyzer (batch + world model) when available,
    otherwise falls back to per-market ClaudeAnalyzer calls.
    """

    def __init__(
        self,
        settings,
        db: Database,
        aggregator: Aggregator,
        analyzer: ClaudeAnalyzer,
        cache: NLPCache,
        exchange: ExchangeClient,
        calibration: CalibrationTracker | None = None,
        flow_tracker=None,
        strategic=None,
    ):
        self.settings = settings
        self.db = db
        self.aggregator = aggregator
        self.analyzer = analyzer
        self.cache = cache
        self.exchange = exchange
        self.calibration = calibration
        self.flow_tracker = flow_tracker
        self.strategic = strategic

    async def analyze_markets(
        self,
        markets: list[Market],
        price_history: dict[str, list[float]] | None = None,
    ) -> list[TradeCandidate]:
        """Analyze markets using the pipeline (strategic batch or per-market)."""
        if self.strategic:
            return await self._analyze_batch_strategic(markets, price_history)
        return await self._analyze_sequential(markets, price_history)

    # ------------------------------------------------------------------
    # Strategic batch path
    # ------------------------------------------------------------------

    async def _analyze_batch_strategic(
        self,
        markets: list[Market],
        price_history: dict[str, list[float]] | None = None,
    ) -> list[TradeCandidate]:
        """Batch analysis with persistent world model."""
        from pathlib import Path
        from auramaur.nlp.query_decomposer import extract_search_queries

        if Path("KILL_SWITCH").exists():
            log.warning("pipeline.kill_switch_active")
            return []

        # Gather evidence for claimable markets
        evidence_map: dict[str, list] = {}
        for market in markets:
            if not self._try_claim_market(market.id):
                continue
            try:
                queries = extract_search_queries(market.question, market.description)
                all_evidence: list = []
                seen_ids: set[str] = set()
                for query in queries:
                    items = await self.aggregator.gather(query, limit_per_source=3)
                    for item in items:
                        if item.id not in seen_ids:
                            seen_ids.add(item.id)
                            all_evidence.append(item)
                evidence_map[market.id] = all_evidence[:self.settings.nlp.evidence_per_source * 3]
            except Exception as e:
                log.error("pipeline.evidence_error", market_id=market.id, error=str(e))
                evidence_map[market.id] = []

        batch_markets = [m for m in markets if m.id in evidence_map]
        if not batch_markets:
            return []

        analysis = await self.strategic.analyze_batch_with_adversarial(batch_markets, evidence_map)

        if not analysis.markets:
            log.warning("pipeline.no_results", batch_size=len(batch_markets))
            return []

        candidates: list[TradeCandidate] = []
        for batch_result in analysis.markets:
            market = next((m for m in batch_markets if m.id == batch_result.market_id), None)
            if market is None:
                continue

            claude_prob = batch_result.probability
            market_prob = market.outcome_yes_price
            raw_edge = claude_prob - market_prob

            if abs(raw_edge) < 0.001:
                continue

            side = OrderSide.BUY if raw_edge > 0 else OrderSide.SELL
            edge = abs(raw_edge) - (POLYMARKET_FEE_PCT / 100)

            signal = Signal(
                market_id=market.id,
                market_question=market.question,
                claude_prob=claude_prob,
                claude_confidence=Confidence(batch_result.confidence),
                market_prob=market_prob,
                edge=edge * 100,
                evidence_summary=batch_result.reasoning[:500],
                recommended_side=side,
            )

            show_analysis(
                signal.claude_prob, signal.market_prob, signal.edge,
                batch_result.confidence, None, None,
            )

            candidates.append(TradeCandidate(market=market, signal=signal))

        return candidates

    # ------------------------------------------------------------------
    # Sequential per-market path
    # ------------------------------------------------------------------

    async def _analyze_sequential(
        self,
        markets: list[Market],
        price_history: dict[str, list[float]] | None = None,
    ) -> list[TradeCandidate]:
        """Analyze markets one at a time (legacy path)."""
        from auramaur.nlp.query_decomposer import extract_search_queries

        candidates: list[TradeCandidate] = []

        for market in markets:
            if not self._try_claim_market(market.id):
                continue

            try:
                result = await self._analyze_single(market)
                if result:
                    candidates.append(result)
            except Exception as e:
                log.error("pipeline.market_error", market_id=market.id, error=str(e))

        return candidates

    async def _analyze_single(self, market: Market) -> TradeCandidate | None:
        """Full pipeline for a single market."""
        from auramaur.nlp.query_decomposer import extract_search_queries

        show_analyzing(market.question, market.id)

        # 1. Gather evidence
        queries = extract_search_queries(market.question, market.description)
        all_evidence: list = []
        seen_ids: set[str] = set()
        per_query_limit = max(1, self.settings.nlp.evidence_per_source // len(queries)) if queries else self.settings.nlp.evidence_per_source
        for query in queries:
            items = await self.aggregator.gather(query, limit_per_source=per_query_limit)
            for item in items:
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    all_evidence.append(item)
        evidence = all_evidence[:self.settings.nlp.evidence_per_source * 3]
        source_counts: dict[str, int] = {}
        for e in evidence:
            source_counts[e.source] = source_counts.get(e.source, 0) + 1
        show_evidence(len(evidence), source_counts)

        # 2. NLP analysis
        analysis = await self.analyzer.analyze(market, evidence, self.cache)
        if analysis.skipped_reason:
            return None

        # 2b. Calibration
        if self.calibration is not None:
            calibrated = await self.calibration.adjust(
                analysis.probability, market.category or ""
            )
            analysis.calibrated_probability = calibrated

        # 2c. Order flow nudge
        if self.flow_tracker is not None:
            try:
                order_book = await self.exchange.get_order_book(market.id)
                self.flow_tracker.record_book_snapshot(market.id, order_book)
            except Exception:
                pass
            nudge = self.flow_tracker.get_probability_nudge(market.id)
            if nudge != 0 and analysis.calibrated_probability is not None:
                analysis.calibrated_probability = max(0.01, min(0.99, analysis.calibrated_probability + nudge))
            elif nudge != 0:
                analysis.probability = max(0.01, min(0.99, analysis.probability + nudge))

        # 3. Signal detection
        signal = detect_edge(market, analysis)
        if signal is None:
            return None

        show_analysis(
            signal.claude_prob, signal.market_prob, signal.edge,
            analysis.confidence, analysis.second_opinion_prob, analysis.divergence,
        )

        return TradeCandidate(market=market, signal=signal)

    # ------------------------------------------------------------------
    # Cross-instance claim lock
    # ------------------------------------------------------------------

    @staticmethod
    def _try_claim_market(market_id: str, ttl_seconds: int = 300) -> bool:
        """Try to claim a market for analysis using a shared lock file."""
        safe_id = market_id.replace("/", "_")
        claim_path = os.path.join(_CLAIM_DIR, f"{safe_id}.claim")

        try:
            fh = open(claim_path, "a+")
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False

        try:
            mtime = os.path.getmtime(claim_path)
            if time.time() - mtime < ttl_seconds:
                size = os.path.getsize(claim_path)
                if size > 0:
                    fh.seek(0)
                    owner = fh.read().strip()
                    if owner and owner != str(os.getpid()):
                        fcntl.flock(fh, fcntl.LOCK_UN)
                        fh.close()
                        return False
        except FileNotFoundError:
            pass

        try:
            fh.seek(0)
            fh.truncate()
            fh.write(str(os.getpid()))
            fh.flush()
            os.fsync(fh.fileno())
            fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()
            return True
        except OSError:
            return False
