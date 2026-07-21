"""Trade decision orchestrator — connects analysis to execution."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

import asyncio
import os
import tempfile
import time

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None
    import msvcrt

from auramaur.data_sources.aggregator import Aggregator
from auramaur.db.database import Database
from auramaur.exchange.models import Market, OrderResult, Signal
from auramaur.exchange.protocols import ExchangeClient, MarketDiscovery

# Shared directory for cross-instance market claim locks
_CLAIM_DIR = os.path.join(tempfile.gettempdir(), "auramaur_claims")
os.makedirs(_CLAIM_DIR, exist_ok=True)


def _lock_claim_file(fh) -> None:
    """Acquire a non-blocking advisory lock on every supported OS."""
    if fcntl is not None:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return

    fh.seek(0, os.SEEK_END)
    if fh.tell() == 0:
        fh.write(b"\0")
        fh.flush()
    fh.seek(0)
    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)


def _unlock_claim_file(fh) -> None:
    if fcntl is not None:
        fcntl.flock(fh, fcntl.LOCK_UN)
        return
    fh.seek(0)
    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)


from auramaur.monitoring.display import (
    show_analysis, show_analyzing, show_evidence, show_risk_decision,
)
from auramaur.nlp.analyzer import ClaudeAnalyzer
from auramaur.nlp.cache import NLPCache
from auramaur.nlp.calibration import CalibrationTracker
from auramaur.broker.allocator import CapitalAllocator
from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.broker.router import SmartOrderRouter
from auramaur.risk.manager import RiskManager
from auramaur.strategy.classifier import ensure_category
from auramaur.strategy.order_flow import OrderFlowTracker
from auramaur.strategy.protocols import MarketAnalyzer
from auramaur.strategy.signals import detect_edge
from auramaur.strategy.engine_cycle import CycleOrchestrationMixin

log = structlog.get_logger()


class TradingEngine(CycleOrchestrationMixin):
    """Orchestrates the full pipeline: data → analysis → risk → execution."""

    def __init__(
        self,
        settings,
        db: Database,
        discovery: MarketDiscovery,
        aggregator: Aggregator,
        analyzer: ClaudeAnalyzer,
        cache: NLPCache,
        risk_manager: RiskManager,
        exchange: ExchangeClient,
        calibration: CalibrationTracker | None = None,
        flow_tracker: OrderFlowTracker | None = None,
        router: SmartOrderRouter | None = None,
        allocator: CapitalAllocator | None = None,
        market_analyzer: MarketAnalyzer | None = None,
        technical_analyzer: MarketAnalyzer | None = None,
    ):
        self.settings = settings
        self.db = db
        self.discovery = discovery
        self.aggregator = aggregator
        self.analyzer = analyzer
        self.cache = cache
        self.risk_manager = risk_manager
        self.exchange = exchange
        self.calibration = calibration
        self.flow_tracker = flow_tracker
        self.router = router
        self.allocator = allocator
        self.market_analyzer = market_analyzer
        self.technical_analyzer = technical_analyzer
        self.exchange_name = ""  # Set by bot after init (e.g. "polymarket", "kalshi")
        self._hybrid = False  # Set by bot when --hybrid flag is active
        self.strategic = None  # StrategicAnalyzer, set by bot after init
        self._components_pnl = None  # Set by bot after init
        self._exec_gateway: ExecutionGateway | None = None  # Lazily built; see _gateway
        # News-flagged markets: market_id -> UTC timestamp of flag.
        # Populated by NewsReactor when a headline matches; read by run_cycle
        # to boost these markets into the strategic batch without triggering
        # per-market analyze_market (which would spend 2 Claude calls each).
        self._news_flagged: dict[str, float] = {}
        self._news_flag_ttl_seconds: float = 1800.0  # 30 minutes

    async def _maybe_refine_with_tool_use(
        self,
        batch_results: list,
        batch_markets: list,
    ) -> list:
        """Refine strategic batch results using tool-use Claude where
        configured. Returns a list of BatchAnalysisResult with the same
        length/ordering as ``batch_results`` — refined entries replace
        their originals; unchanged entries pass through.

        Selection:
          * ``analysis_mode=strategic_batch`` — skip entirely.
          * ``analysis_mode=tool_use`` — refine every batch result.
          * ``analysis_mode=auto`` (default) — refine markets whose
            |edge| vs market price exceeds the configured threshold,
            capped at ``tool_use_max_markets_per_cycle``.
        """
        mode = self.settings.nlp.analysis_mode
        if mode == "strategic_batch":
            return batch_results

        # Pair each result with its Market so we can compute edge.
        market_by_id = {m.id: m for m in batch_markets}

        candidates: list[tuple[int, float]] = []  # (index, abs_edge)
        threshold = self.settings.nlp.tool_use_edge_threshold_pct / 100.0
        for idx, br in enumerate(batch_results):
            market = market_by_id.get(br.market_id)
            if market is None:
                continue
            market_prob = market.outcome_yes_price or 0.5
            abs_edge = abs(br.probability - market_prob)
            if mode == "tool_use" or abs_edge >= threshold:
                candidates.append((idx, abs_edge))

        if not candidates:
            return batch_results

        # Sort by edge desc, cap by budget.
        candidates.sort(key=lambda x: x[1], reverse=True)
        cap = self.settings.nlp.tool_use_max_markets_per_cycle
        selected = candidates[:cap]

        log.info(
            "tool_use.refining",
            mode=mode,
            selected=len(selected),
            threshold_pct=self.settings.nlp.tool_use_edge_threshold_pct,
        )

        from auramaur.nlp.tool_use_analyzer import ToolUseAnalyzer
        analyzer = ToolUseAnalyzer(self.settings)

        # Refine concurrently — each call shells out to `claude` and can
        # take 30-120s. Parallelism keeps wall-clock reasonable.
        async def _one(idx: int):
            market = market_by_id[batch_results[idx].market_id]
            refined = await analyzer.refine(market, batch_results[idx])
            return idx, refined

        outcomes = await asyncio.gather(
            *(_one(i) for i, _ in selected), return_exceptions=False,
        )

        new_results = list(batch_results)
        for idx, refined in outcomes:
            if refined is None:
                continue
            original = batch_results[idx]
            log.info(
                "tool_use.refined",
                market_id=original.market_id,
                before=round(original.probability, 3),
                after=round(refined.probability, 3),
                delta=round(refined.probability - original.probability, 3),
            )
            new_results[idx] = refined

        return new_results

    def flag_market_from_news(self, market_id: str) -> None:
        """Register a market for priority analysis in the next cycle.

        News-driven hot markets get boosted to the front of the ranked list
        so the next strategic batch evaluates them — without spending a
        full per-market Claude+second-opinion pair per headline.
        """
        self._news_flagged[market_id] = time.time()

    def _prune_news_flags(self) -> set[str]:
        """Drop expired flags and return the active set."""
        now = time.time()
        ttl = self._news_flag_ttl_seconds
        self._news_flagged = {
            mid: ts for mid, ts in self._news_flagged.items() if now - ts < ttl
        }
        return set(self._news_flagged.keys())

    @staticmethod
    def _get_event_key(market_id: str) -> str:
        """Extract event base from market ID for grouping."""
        if market_id.startswith("KX") and market_id.count("-") >= 2:
            return market_id.rsplit("-", 1)[0]
        return market_id

    async def _apply_rejection_cooldown(self, candidates: list) -> tuple[list, int]:
        """Drop candidates with a fresh risk rejection. Returns (kept, benched).

        Three things lift the bench, all of them "new information" signals:
        the cooldown expiring, the yes-price moving by at least the reprice
        threshold since the rejection, or an active news flag on the market.
        """
        try:
            cooldown_min = int(self.settings.nlp.rejection_cooldown_minutes)
            reprice = float(self.settings.nlp.rejection_reprice_threshold)
        except Exception:
            return candidates, 0
        if cooldown_min <= 0 or not candidates:
            return candidates, 0
        try:
            rows = await self.db.fetchall(
                """SELECT market_id, yes_price FROM signal_rejections
                   WHERE rejected_at > datetime('now', ?)""",
                (f"-{cooldown_min} minutes",),
            )
        except Exception:
            return candidates, 0  # Table may not exist yet
        benched = {r["market_id"]: r["yes_price"] for r in rows}
        if not benched:
            return candidates, 0
        flagged = set(self._news_flagged.keys())
        kept = [
            m for m in candidates
            if m.id not in benched
            or m.id in flagged
            or abs(m.outcome_yes_price - benched[m.id]) >= reprice
        ]
        benched_count = len(candidates) - len(kept)
        if benched_count > 0:
            log.info("engine.rejection_benched", count=benched_count, exchange=self.exchange_name)
        return kept, benched_count

    async def _record_rejection_state(self, market, approved: bool, reason: str) -> None:
        """Track the risk verdict for the rejection cooldown.

        A rejection benches the market (upsert refreshes the clock and bumps
        the streak); an approval clears it — the market earned its way back.
        """
        try:
            if approved:
                await self.db.execute(
                    "DELETE FROM signal_rejections WHERE market_id = ?",
                    (market.id,),
                )
            else:
                await self.db.execute(
                    """INSERT INTO signal_rejections
                       (market_id, exchange, rejected_at, yes_price, reason, streak)
                       VALUES (?, ?, datetime('now'), ?, ?, 1)
                       ON CONFLICT(market_id) DO UPDATE SET
                           rejected_at = excluded.rejected_at,
                           yes_price = excluded.yes_price,
                           reason = excluded.reason,
                           streak = streak + 1""",
                    (
                        market.id,
                        market.exchange or self.exchange_name or "",
                        market.outcome_yes_price,
                        (reason or "")[:200],
                    ),
                )
            await self.db.commit()
        except Exception as e:
            log.debug("engine.rejection_record_error", market_id=market.id, error=str(e))

    async def _get_available_cash(self) -> float:
        """Get available cash from syncer, exchange, or paper balance."""
        syncer = getattr(self, '_components_syncer', None)
        if syncer:
            return await syncer.get_cash_balance()
        # No syncer — try querying the exchange directly (e.g. Kalshi)
        if hasattr(self.exchange, 'get_balance'):
            try:
                return await self.exchange.get_balance()
            except Exception:
                pass
        return self.settings.execution.paper_initial_balance

    async def _get_positions_and_cash(self) -> tuple[list, float]:
        """Get current positions and cash for allocation."""
        syncer = getattr(self, '_components_syncer', None)
        if syncer:
            positions = await syncer.sync()
            cash = await syncer.get_cash_balance()
            return positions, cash
        # No syncer — get positions from portfolio table for this exchange
        from auramaur.exchange.models import LivePosition
        positions: list[LivePosition] = []
        try:
            if self.exchange_name:
                # Join portfolio with markets to filter by exchange
                rows = await self.db.fetchall(
                    """SELECT p.market_id, p.size, p.avg_price, p.current_price
                       FROM portfolio p
                       JOIN markets m ON p.market_id = m.id
                       WHERE p.size > 0 AND m.exchange = ?""",
                    (self.exchange_name,),
                )
            else:
                rows = await self.db.fetchall(
                    """SELECT market_id, size, avg_price, current_price
                       FROM portfolio WHERE size > 0"""
                )
            for row in rows:
                positions.append(LivePosition(
                    market_id=row["market_id"],
                    size=row["size"],
                    avg_cost=row.get("avg_price", 0) or 0,
                    current_price=row.get("current_price", 0) or 0,
                ))
        except Exception:
            pass
        cash = await self._get_available_cash()
        return positions, cash

    async def _held_market_ids(self) -> set[str]:
        """market_ids we already hold in the current mode + exchange.

        Used to keep already-held markets out of the BUY-candidate set: the
        allocator won't average into an open position (skip_held), so analyzing
        them only burns LLM calls and yields approved-but-unexecutable
        candidates. Exit/management of held positions runs through the dedicated
        venue-exit path and the portfolio monitor, not this cycle, so dropping
        them here does not affect closing positions. The allocator's skip_held
        remains the correctness backstop; this is a pure efficiency filter.
        """
        is_paper_flag = 0 if self.settings.is_live else 1
        try:
            if self.exchange_name:
                rows = await self.db.fetchall(
                    """SELECT p.market_id FROM portfolio p
                       JOIN markets m ON p.market_id = m.id
                       WHERE p.size > 0 AND p.is_paper = ? AND m.exchange = ?""",
                    (is_paper_flag, self.exchange_name),
                )
            else:
                rows = await self.db.fetchall(
                    "SELECT market_id FROM portfolio WHERE size > 0 AND is_paper = ?",
                    (is_paper_flag,),
                )
            return {r["market_id"] for r in rows}
        except Exception:
            return set()

    async def scan_and_store_markets(self, limit: int = 300) -> list[Market]:
        """Fetch markets from Gamma API and store in DB."""
        markets = await self.discovery.get_markets(limit=limit)

        # Classify BEFORE opening the transaction — keeps the write burst
        # db-only so the write lock is held for ms, not the classify pass.
        for market in markets:
            # Venue-tag categories (set by the gamma parser from event tags)
            # are authoritative; keyword-classify only when tags were absent
            # or inconclusive.
            market.category = ensure_category(
                market.question, market.description, market.category)

        async with self.db.transaction():
            for market in markets:
                await self.db.execute(
                    """INSERT OR REPLACE INTO markets
                       (id, exchange, condition_id, question, description, category, end_date, active,
                        outcome_yes_price, outcome_no_price, volume, liquidity,
                        clob_token_yes, clob_token_no, last_updated)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        market.id, market.exchange or self.exchange_name or "polymarket",
                        market.condition_id, market.question,
                        market.description, market.category,
                        market.end_date.isoformat() if market.end_date else None,
                        int(market.active), market.outcome_yes_price, market.outcome_no_price,
                        market.volume, market.liquidity,
                        market.clob_token_yes, market.clob_token_no,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )

        # Price snapshots + prune as a second short transaction so the
        # markets upsert above commits (and releases the write lock) first.
        async with self.db.transaction():
            for market in markets:
                await self.db.execute(
                    "INSERT INTO price_history (market_id, price, exchange)"
                    " VALUES (?, ?, ?)",
                    (market.id, market.outcome_yes_price,
                     market.exchange or self.exchange_name or "polymarket"),
                )
            # Prune old price history to bound table growth. Extended 7 -> 30
            # days so reversion/intraday research has a multi-week sample; the
            # 7-day window was too short to rule out a regime artifact.
            await self.db.execute(
                "DELETE FROM price_history WHERE recorded_at < datetime('now', '-30 days')"
            )

        return markets

    @staticmethod
    def _is_junk_market(market: Market) -> str | None:
        """Return a reason string if the market should be skipped, else None."""
        q = market.question.lower()

        # Novelty/meme markets
        _NOVELTY = [
            "before gta", "jesus christ", "second coming", "flat earth",
            "zombie", "apocalypse", "rapture", "end of the world",
            "simulation theory", "time travel",
        ]
        if any(p in q for p in _NOVELTY):
            return "novelty/meme market"

        # Unresearchable/speculation markets — insider knowledge, tabloid, unverifiable
        _UNRESEARCHABLE = [
            "epstein", "visited", "island", "sex tape", "leaked",
            "affair", "cheating", "nude", "onlyfans",
            "die before", "death", "assassinat",
            "up or down", "bitcoin up", "ethereum up", "xrp up",  # coin flip crypto
            "next video get between", "views on",  # social media metrics
            "highest temperature", "°c on", "°f on",  # exact weather
            "exactly", "be between",  # narrow numeric ranges are noise
        ]

        # Game-outcome sports markets — efficiently priced by sharps,
        # no informational edge from news analysis.  News-driven sports
        # questions (trades, draft, firings, playoffs) are kept.
        _GAME_OUTCOMES = [
            " vs ", " v ", "win game", "win tonight", "beat the",
            "cover the spread", "over/under", "total points",
            "total goals", "moneyline", "score more",
        ]
        if market.category in {"sports", "esports"}:
            if any(p in q for p in _GAME_OUTCOMES):
                return "game-outcome market — no edge vs sharps"
        if any(p in q for p in _UNRESEARCHABLE):
            return "unresearchable/speculation market"

        # Extreme prices — already settled
        if market.outcome_yes_price < 0.03 or market.outcome_yes_price > 0.97:
            return f"price {market.outcome_yes_price:.0%} — already settled"

        # No-edge categories — only block gambling (pure chance) and
        # exact-outcome weather (unresearchable).  Sports, esports, and
        # general weather are left to the feedback loop: if the bot
        # loses money on them, the avoid-category system blocks them
        # dynamically.  News-driven sports questions (trades, firings,
        # playoff qualification) can have real informational edge.
        if market.category in {"gambling"}:
            return f"{market.category} — pure chance, no informational edge"

        # Too short-term
        if market.end_date is not None:
            now = datetime.now(timezone.utc)
            end = market.end_date if market.end_date.tzinfo else market.end_date.replace(tzinfo=timezone.utc)
            hours_left = (end - now).total_seconds() / 3600
            if hours_left < 2:
                return f"resolves in {hours_left:.1f}h — too short-term"
            if hours_left > 10 * 365 * 24:
                return f"resolves in {hours_left / (24 * 365):.1f}y — too speculative"

        return None

    @staticmethod
    def _try_claim_market(market_id: str, ttl_seconds: int = 300) -> bool:
        """Try to claim a market for analysis using a shared lock file.

        Returns True if this instance claimed it, False if another instance
        already has it.  Claims expire after *ttl_seconds* so stale locks
        from crashed instances are automatically released.

        Uses atomic file locking: acquire the lock first, THEN check
        freshness.  This avoids the TOCTOU race where two instances both
        see a stale/missing file and both proceed to claim.
        """
        import time

        safe_id = market_id.replace("/", "_")
        claim_path = os.path.join(_CLAIM_DIR, f"{safe_id}.claim")

        # Try to acquire exclusive lock first — if we can't, another instance has it
        try:
            fh = open(claim_path, "a+b")
            _lock_claim_file(fh)
        except OSError:
            return False

        # Lock acquired — now check if the existing claim is still fresh
        try:
            mtime = os.path.getmtime(claim_path)
            if time.time() - mtime < ttl_seconds:
                # File was recently touched by us or another instance that
                # has since released the lock — treat as already claimed
                size = os.path.getsize(claim_path)
                if size > 0:
                    fh.seek(0)
                    owner = fh.read().strip(b"\0").decode(errors="replace")
                    if owner and owner != str(os.getpid()):
                        _unlock_claim_file(fh)
                        fh.close()
                        return False
        except FileNotFoundError:
            pass

        # Claim it: truncate and write our PID, update mtime
        try:
            fh.seek(0)
            fh.truncate()
            fh.write(str(os.getpid()).encode())
            fh.flush()
            os.fsync(fh.fileno())
            # Release lock so other instances can check freshness
            _unlock_claim_file(fh)
            fh.close()
            return True
        except OSError:
            return False

    async def analyze_market(
        self,
        market: Market,
        *,
        place_order: bool = True,
        price_history: dict[str, list[float]] | None = None,
        strategy_source: str = "llm",
    ) -> dict | None:
        """Full pipeline for a single market.

        When *place_order* is False the method evaluates the market and runs
        risk checks but does **not** place an order.  This is used by the
        two-phase allocate-then-execute cycle in ``run_cycle``.
        """
        # Pre-screen: skip junk markets regardless of caller
        skip_reason = self._is_junk_market(market)
        if skip_reason:
            log.info("engine.skipped_junk", market_id=market.id, reason=skip_reason)
            return None

        # Cross-instance dedup: skip if another bot is already analyzing this
        if not self._try_claim_market(market.id):
            log.debug("engine.already_claimed", market_id=market.id)
            return None

        show_analyzing(market.question, market.id)

        # 1. Gather evidence (using decomposed queries for better results)
        from auramaur.data_sources.base import NewsItem as _NI
        from auramaur.nlp.query_decomposer import extract_search_queries

        # Enrich market description from CLOB if thin
        if len(market.description) < 50 and market.condition_id:
            try:
                self.exchange._init_clob_client()
                clob_info = self.exchange._clob_client.get_market(market.condition_id)
                if clob_info and clob_info.get("description"):
                    market.description = clob_info["description"][:1000]
            except Exception:
                pass

        queries = extract_search_queries(market.question, market.description, market.category or "")
        all_evidence: list = []
        seen_ids: set[str] = set()

        # Add market description as synthetic evidence (resolution criteria)
        if market.description and len(market.description) > 20:
            all_evidence.append(_NI(
                id=f"polymarket_desc:{market.id}",
                source="polymarket_context",
                title=f"Resolution criteria: {market.question}",
                content=market.description[:800],
                url=f"https://polymarket.com/event/{market.id}",
            ))
            seen_ids.add(f"polymarket_desc:{market.id}")

        per_query_limit = max(1, self.settings.nlp.evidence_per_source // len(queries)) if queries else self.settings.nlp.evidence_per_source
        for query in queries:
            items = await self.aggregator.gather(
                query, limit_per_source=per_query_limit, category=market.category or None,
                market_id=market.id, market_price=market.outcome_yes_price,
            )
            for item in items:
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    all_evidence.append(item)
        evidence = all_evidence[:self.settings.nlp.evidence_per_source * 3]
        evidence_run_ids = sorted({e.ingestion_run_id for e in evidence if e.ingestion_run_id})
        source_counts: dict[str, int] = {}
        for e in evidence:
            source_counts[e.source] = source_counts.get(e.source, 0) + 1
        show_evidence(len(evidence), source_counts)

        # 2. NLP Analysis
        analysis = await self.analyzer.analyze(market, evidence, self.cache)
        if analysis.skipped_reason:
            return None

        # 2b. Calibration adjustment
        if self.calibration is not None:
            calibrated = await self.calibration.adjust(
                analysis.probability, market.category or ""
            )
            analysis.calibrated_probability = calibrated

        # Existing calibration remains on its established connection. New
        # lineage is observer-only: enqueueing is non-blocking and its worker
        # owns a separate connection, so it cannot roll back or stall trading.
        if self.calibration is not None:
            await self.calibration.record_prediction(
                market.id, analysis.probability, market.category or "",
            )
        observer = getattr(self.aggregator, "observer", None)
        if observer is not None:
            observer.forecast(
                market_id=market.id,
                exchange=market.exchange or self.exchange_name or "polymarket",
                category=market.category or "", raw_probability=analysis.probability,
                calibrated_probability=analysis.calibrated_probability,
                market_yes_price=market.outcome_yes_price,
                market_no_price=market.outcome_no_price,
                observed_at=datetime.now(timezone.utc).isoformat(),
                evidence_run_ids=evidence_run_ids,
                model=getattr(self.analyzer, "_model", ""),
                strategy_source=strategy_source,
                config=self.settings.nlp.model_dump(mode="json"),
            )

        # 2c. Order flow nudge
        if self.flow_tracker is not None:
            try:
                order_book = await self.exchange.get_order_book(market.clob_token_yes)
                self.flow_tracker.record_book_snapshot(market.id, order_book)
            except Exception:
                pass
            nudge = self.flow_tracker.get_probability_nudge(market.id)
            if nudge != 0 and analysis.calibrated_probability is not None:
                analysis.calibrated_probability = max(0.01, min(0.99, analysis.calibrated_probability + nudge))
            elif nudge != 0:
                analysis.probability = max(0.01, min(0.99, analysis.probability + nudge))

        # 3. Signal detection
        signal = detect_edge(market, analysis, exchange_fees=self.settings.arbitrage.exchange_fees)
        if signal is None:
            return None
        signal.strategy_source = strategy_source

        show_analysis(
            signal.claude_prob, signal.market_prob, signal.edge,
            analysis.confidence, analysis.second_opinion_prob, analysis.divergence,
        )

        # Debug-log raw market data for suspiciously large edges (>30%)
        # to help diagnose calibration or inverted-semantics issues.
        if signal.edge > 30:
            log.warning(
                "signal.debug_dump",
                market_id=market.id,
                exchange=market.exchange,
                question=market.question[:200],
                description=(market.description or "")[:200],
                category=market.category,
                yes_price=market.outcome_yes_price,
                no_price=market.outcome_no_price,
                volume=market.volume,
                liquidity=market.liquidity,
                spread=market.spread,
                claude_prob=signal.claude_prob,
                claude_confidence=analysis.confidence,
                edge_pct=signal.edge,
                side=signal.recommended_side.value if signal.recommended_side else None,
                reasoning=(analysis.reasoning or "")[:300],
            )

        # 4. Ensure market exists in DB then store signal
        await self.db.execute(
            """INSERT OR IGNORE INTO markets (id, exchange, condition_id, question, description,
               category, active, outcome_yes_price, outcome_no_price,
               volume, liquidity, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, datetime('now'))""",
            (market.id, market.exchange or self.exchange_name or "polymarket",
             market.condition_id, market.question,
             market.description[:500],
             ensure_category(market.question, market.description, market.category),
             market.outcome_yes_price, market.outcome_no_price,
             market.volume, market.liquidity),
        )
        await self.db.execute(
            """INSERT INTO signals (market_id, claude_prob, claude_confidence, market_prob,
                                     edge, second_opinion_prob, divergence, evidence_summary, action,
                                     strategy_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal.market_id, signal.claude_prob, signal.claude_confidence.value,
                signal.market_prob, signal.edge, signal.second_opinion_prob,
                signal.divergence, signal.evidence_summary,
                signal.recommended_side.value if signal.recommended_side else None,
                signal.strategy_source,
            ),
        )
        await self.db.commit()

        # 5. Risk evaluation (pass actual cash for correct Kelly sizing)
        cash = await self._get_available_cash()
        decision = await self.risk_manager.evaluate(signal, market, price_history=price_history, available_cash=cash)
        checks_passed = sum(1 for c in decision.checks if c.passed)
        checks_failed = sum(1 for c in decision.checks if not c.passed)
        show_risk_decision(
            decision.approved, decision.reason, checks_passed,
            checks_failed, decision.position_size, market_id=signal.market_id,
            strategy=signal.strategy_source,
            graduation=getattr(decision, "graduation_status", ""),
            mispricing=getattr(signal, "mispricing_reason", ""))
        await self._record_rejection_state(market, decision.approved, decision.reason)

        if not decision.approved or decision.position_size <= 0:
            return {"market": market, "signal": signal, "decision": decision, "order": None}

        if not place_order:
            # Evaluate-only mode: return candidate for the allocator
            return {"market": market, "signal": signal, "decision": decision, "order": None}

        # 6. Build and place order — use smart router if available
        order = await self._build_and_place_order(
            signal, market, decision.position_size,
            force_paper=decision.force_paper)
        if order is None:
            return {"market": market, "signal": signal, "decision": decision, "order": None}

        return {"market": market, "signal": signal, "decision": decision, "order": order}

    @property
    def _gateway(self) -> ExecutionGateway:
        """Lazily build (and keep in sync with) the ExecutionGateway.

        ``exchange_name`` and ``_components_pnl`` are bound by the bot AFTER
        engine init, so the gateway is built on first use and rebuilt if either
        late-bound collaborator changes.
        """
        gw = self._exec_gateway
        if (gw is None
                or gw.exchange_name != self.exchange_name
                or gw.pnl_tracker is not self._components_pnl):
            gw = ExecutionGateway(
                router=self.router,
                exchange=self.exchange,
                exchange_name=self.exchange_name,
                settings=self.settings,
                db=self.db,
                pnl_tracker=self._components_pnl,
            )
            self._exec_gateway = gw
        return gw

    async def _build_and_place_order(
        self, signal: Signal, market: Market, size_dollars: float,
        force_paper: bool = False,
    ) -> OrderResult | None:
        """Build an order (via router or direct) and place it.

        Delegates to :class:`ExecutionGateway`; returns the underlying
        ``OrderResult`` (or ``None`` when the trade was skipped before
        submission — unmarketable / build failure), preserving the prior
        return contract. ``force_paper`` (graduation ladder) downgrades this
        ENTRY to dry-run regardless of global live mode.
        """
        intent = TradeIntent(
            signal=signal, market=market, size_dollars=size_dollars,
            force_paper=force_paper,
        )
        return (await self._gateway.submit(intent)).result

    async def _record_trade_for_attribution(self, market: Market, signal, decision) -> None:
        """Deprecated hook kept for compatibility.

        The portfolio table represents current position state and is owned by
        the PnL tracker plus exchange reconcilers.  Writing synthetic rows here
        used the wrong primary key and mixed dollar notional with token size,
        which corrupted attribution instead of improving it.
        """
        return None
