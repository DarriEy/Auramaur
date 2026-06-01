"""Trade decision orchestrator — connects analysis to execution."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

import asyncio
import fcntl
import os
import tempfile
import time

from auramaur.data_sources.aggregator import Aggregator
from auramaur.db.database import Database
from auramaur.exchange.models import Market, OrderResult, OrderSide, Signal
from auramaur.exchange.protocols import ExchangeClient, MarketDiscovery

# Shared directory for cross-instance market claim locks
_CLAIM_DIR = os.path.join(tempfile.gettempdir(), "auramaur_claims")
os.makedirs(_CLAIM_DIR, exist_ok=True)
from auramaur.monitoring.display import (
    show_analysis, show_analyzing, show_cycle_summary,
    show_evidence, show_order, show_order_dropped, show_risk_decision, show_scan_results,
)
from auramaur.nlp.analyzer import ClaudeAnalyzer
from auramaur.nlp.cache import NLPCache
from auramaur.nlp.calibration import CalibrationTracker
from auramaur.broker.allocator import CandidateTrade, CapitalAllocator
from auramaur.broker.router import SmartOrderRouter
from auramaur.risk.manager import RiskManager
from auramaur.strategy.classifier import classify_market
from auramaur.strategy.order_flow import OrderFlowTracker
from auramaur.strategy.protocols import MarketAnalyzer, TradeCandidate
from auramaur.strategy.signals import detect_edge

log = structlog.get_logger()


class TradingEngine:
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

    async def scan_and_store_markets(self, limit: int = 300) -> list[Market]:
        """Fetch markets from Gamma API and store in DB."""
        markets = await self.discovery.get_markets(limit=limit)

        for market in markets:
            category = classify_market(market.question, market.description)
            market.category = category

            await self.db.execute(
                """INSERT OR REPLACE INTO markets
                   (id, exchange, condition_id, question, description, category, end_date, active,
                    outcome_yes_price, outcome_no_price, volume, liquidity, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    market.id, market.exchange or self.exchange_name or "polymarket",
                    market.condition_id, market.question,
                    market.description, category,
                    market.end_date.isoformat() if market.end_date else None,
                    int(market.active), market.outcome_yes_price, market.outcome_no_price,
                    market.volume, market.liquidity, datetime.now(timezone.utc).isoformat(),
                ),
            )

        await self.db.commit()

        # Record price snapshots for momentum tracking
        for market in markets:
            await self.db.execute(
                "INSERT INTO price_history (market_id, price) VALUES (?, ?)",
                (market.id, market.outcome_yes_price),
            )
        await self.db.commit()

        # Prune old price history to prevent unbounded table growth
        await self.db.execute(
            "DELETE FROM price_history WHERE recorded_at < datetime('now', '-7 days')"
        )
        await self.db.commit()

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
            fh = open(claim_path, "a+")
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
                    owner = fh.read().strip()
                    if owner and owner != str(os.getpid()):
                        fcntl.flock(fh, fcntl.LOCK_UN)
                        fh.close()
                        return False
        except FileNotFoundError:
            pass

        # Claim it: truncate and write our PID, update mtime
        try:
            fh.seek(0)
            fh.truncate()
            fh.write(str(os.getpid()))
            fh.flush()
            os.fsync(fh.fileno())
            # Release lock so other instances can check freshness
            fcntl.flock(fh, fcntl.LOCK_UN)
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
            )
            for item in items:
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    all_evidence.append(item)
        evidence = all_evidence[:self.settings.nlp.evidence_per_source * 3]
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
            await self.calibration.record_prediction(
                market.id, analysis.probability, market.category or ""
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
             market.description[:500], market.category,
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
        show_risk_decision(decision.approved, decision.reason, checks_passed, checks_failed, decision.position_size)

        if not decision.approved or decision.position_size <= 0:
            return {"market": market, "signal": signal, "decision": decision, "order": None}

        if not place_order:
            # Evaluate-only mode: return candidate for the allocator
            return {"market": market, "signal": signal, "decision": decision, "order": None}

        # 6. Build and place order — use smart router if available
        order = await self._build_and_place_order(signal, market, decision.position_size)
        if order is None:
            return {"market": market, "signal": signal, "decision": decision, "order": None}

        return {"market": market, "signal": signal, "decision": decision, "order": order}

    async def _build_and_place_order(
        self, signal: Signal, market: Market, size_dollars: float,
    ) -> OrderResult | None:
        """Build an order (via router or direct) and place it."""
        if self.router:
            order = await self.router.route(signal, market, size_dollars, self.settings.is_live)
        else:
            order = self.exchange.prepare_order(signal, market, size_dollars, self.settings.is_live)

        if order is None:
            show_order_dropped(market.id, f"order build failed (${size_dollars:.2f} — could not build a valid order)")
            log.warning(
                "engine.order_dropped",
                market_id=market.id,
                size_dollars=size_dollars,
                reason="prepare_order returned None (bad price/token or router rejection)",
            )
            # Sub-minimum sizing is now bumped up in prepare_order, so a None
            # order here is a genuine build failure (bad price/token). Block
            # only briefly so a transient issue can retry next cycle.
            try:
                await self.db.execute(
                    """INSERT OR REPLACE INTO order_build_drops
                       (market_id, blocked_until, reason)
                       VALUES (?, datetime('now', '+2 hours'), ?)""",
                    (market.id, f"order build failed at ${size_dollars:.2f}"),
                )
                await self.db.commit()
            except Exception:
                pass  # Table may not exist yet
            return None

        result = await self.exchange.place_order(order)
        show_order(result.status, result.order_id, order.side.value, order.size, order.price, result.is_paper, exchange=self.exchange_name, error_message=result.error_message, market_id=order.market_id)

        # Cooldown on API errors — retry in 30 min, not every cycle
        if result.status == "rejected" and result.order_id == "ERROR":
            try:
                await self.db.execute(
                    """INSERT OR REPLACE INTO order_build_drops
                       (market_id, blocked_until, reason)
                       VALUES (?, datetime('now', '+30 minutes'), ?)""",
                    (order.market_id, "place_order API error"),
                )
                await self.db.commit()
            except Exception:
                pass

        # Log slippage only for actual executions.  Live pending orders echo
        # the limit price but have not filled yet.
        if result.status in ("filled", "paper", "partial") and result.filled_price > 0:
            slippage_bps = (result.filled_price - order.price) / order.price * 10000
            if order.side == OrderSide.SELL:
                slippage_bps = -slippage_bps  # For sells, lower fill = worse
            try:
                await self.db.execute(
                    """INSERT INTO slippage_log (market_id, exchange, side, expected_price, filled_price, slippage_bps, size, order_type)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (order.market_id, order.exchange or self.exchange_name, order.side.value,
                     order.price, result.filled_price, round(slippage_bps, 2), order.size,
                     order.order_type.value if hasattr(order, 'order_type') else 'limit'),
                )
                await self.db.commit()
            except Exception:
                pass

        fill_size = result.filled_size if result.filled_size > 0 else order.size
        fill_price = result.filled_price if result.filled_price > 0 else order.price

        # Record P&L only for actual executions.  Pending live orders are
        # mirrored to trades below, then finalized by the order monitor.
        if result.status in ("filled", "paper", "partial") and result.filled_size > 0:
            from auramaur.exchange.models import Fill
            fill = Fill(
                order_id=result.order_id,
                market_id=order.market_id,
                token_id=order.token_id,
                side=order.side,
                token=order.token,
                size=result.filled_size,
                price=fill_price,
                is_paper=result.is_paper,
            )
            pnl_tracker = self._components_pnl
            if pnl_tracker:
                await pnl_tracker.record_fill(fill)

        if result.status in ("filled", "paper", "partial", "pending"):
            # Mirror into legacy `trades` table so the CLI stats view,
            # order monitor, and holding-period lookups stay in sync.
            # PnLTracker writes authoritative execution rows to `fills`.
            try:
                trade_status = "filled" if result.status == "paper" else result.status
                await self.db.execute(
                    """INSERT INTO trades
                       (market_id, signal_id, side, size, price, is_paper,
                        order_id, status, kelly_fraction, exchange, strategy_source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        order.market_id,
                        getattr(signal, "id", None),
                        order.side.value,
                        fill_size,
                        fill_price,
                        1 if result.is_paper else 0,
                        result.order_id,
                        trade_status,
                        None,
                        order.exchange or self.exchange_name,
                        signal.strategy_source,
                    ),
                )
                await self.db.commit()
            except Exception as e:
                log.debug("engine.trade_mirror_error", error=str(e))

        return result

    async def _record_trade_for_attribution(self, market: Market, signal, decision) -> None:
        """Deprecated hook kept for compatibility.

        The portfolio table represents current position state and is owned by
        the PnL tracker plus exchange reconcilers.  Writing synthetic rows here
        used the wrong primary key and mixed dollar notional with token size,
        which corrupted attribution instead of improving it.
        """
        return None

    async def _run_cycle_starved(self) -> list[dict]:
        """Watch mode when cash-starved: refresh prices only, zero Claude calls.

        No point analyzing markets we can't trade. Just:
        1. Refresh market prices (API calls to exchanges, not Claude)
        2. Log a quiet status line
        3. Return — exits and rebalancing are handled by the bot's
           portfolio monitor and Kalshi sync, not the trading cycle.
        """
        import time
        from auramaur.monitoring.display import show_cycle_summary

        start = time.monotonic()

        # Refresh prices for held positions — cheap exchange API calls only
        await self.scan_and_store_markets()

        elapsed = time.monotonic() - start
        log.debug(
            "engine.watch_mode",
            exchange=self.exchange_name,
            elapsed=round(elapsed, 1),
        )
        show_cycle_summary(0, 0, elapsed, exchange=self.exchange_name)
        return []

    async def run_cycle(self, cash_available: float | None = None) -> list[dict]:
        """Run one full trading cycle.

        When ``cash_available`` is below $5, switches to "starved mode":
        only re-prices held positions and checks whether any new
        opportunity is compelling enough to justify liquidating an
        existing stake. Otherwise runs the full scan.
        """
        # Starved check uses the LOWER of aggregate cash and this exchange's own
        # deployable balance. An exchange with separate funding (e.g. Kalshi) can
        # be starved even when aggregate cash looks healthy — without this it
        # would burn full cycles scanning for BUYs it cannot fund. Starved mode
        # still manages/closes existing positions, so we don't abandon the book.
        effective_cash = cash_available
        try:
            exch_cash = await self._get_available_cash()
            if exch_cash is not None:
                effective_cash = exch_cash if effective_cash is None else min(effective_cash, exch_cash)
        except Exception:
            pass
        if effective_cash is not None and effective_cash < 5.0:
            return await self._run_cycle_starved()

        import time
        start = time.monotonic()

        markets = await self.scan_and_store_markets()

        # Kalshi's book is thinner than Polymarket's, so it gets a lower activity
        # floor (kalshi_min_liquidity) to surface more genuinely-tradeable markets.
        min_liq = (
            self.settings.risk.kalshi_min_liquidity
            if self.exchange_name == "kalshi"
            else self.settings.risk.min_liquidity
        )
        candidates = [
            m for m in markets
            if m.active
            # Use the HIGHER of liquidity and volume as the activity measure.
            # Polymarket reports deep liquidity; Kalshi reports thin top-of-book
            # liquidity but high volume on active markets. Using max() ensures
            # active Kalshi markets aren't filtered out by the Polymarket-tuned
            # min_liquidity threshold.
            and max(m.liquidity or 0, m.volume or 0) >= min_liq
            and m.spread <= self.settings.risk.max_spread_pct / 100
            and self.settings.risk.implied_prob_min <= m.outcome_yes_price <= self.settings.risk.implied_prob_max
            # Skip near-dead markets (no volume = orders won't fill)
            and m.volume >= 100
        ]

        # --- Load avoid-categories from performance feedback ---
        avoid_categories: set[str] = set()
        feedback: PerformanceFeedback | None = None
        try:
            from auramaur.broker.feedback import PerformanceFeedback
            feedback = PerformanceFeedback(self.db)
            avoid_categories = await feedback.get_avoid_categories()
            if avoid_categories:
                log.info("engine.avoid_categories", categories=sorted(avoid_categories))
        except Exception as e:
            log.debug("engine.feedback_import_error", error=str(e))

        # --- Hybrid domain filter: restrict LLM to proven categories ---
        hybrid_whitelist: set[str] = set()
        hybrid_probationary: set[str] = set()
        if self._hybrid and self.settings.hybrid.llm_domain_filter and feedback is not None:
            try:
                hybrid_whitelist, hybrid_probationary = await feedback.get_whitelist_categories(
                    min_accuracy=self.settings.hybrid.llm_whitelist_min_accuracy,
                    min_trades=self.settings.hybrid.llm_whitelist_min_trades,
                )
                if hybrid_whitelist:
                    log.info(
                        "hybrid.domain_filter",
                        whitelisted=sorted(hybrid_whitelist),
                        probationary=sorted(hybrid_probationary),
                    )
            except Exception as e:
                log.debug("hybrid.domain_filter_error", error=str(e))

        # --- Filter markets where Claude has no informational edge ---
        edge_candidates: list[Market] = []
        filtered_count = 0

        blocked = set(self.settings.risk.blocked_categories)

        for m in candidates:
            if m.category in blocked:
                filtered_count += 1
                continue
            # Check performance-based avoid list
            if m.category in avoid_categories:
                log.info(
                    "engine.filtered_poor_performance",
                    market_id=m.id,
                    category=m.category,
                )
                filtered_count += 1
                continue
            # Hybrid mode: explore categories with no track record yet instead of
            # filtering them. Proven-poor categories are already removed by
            # avoid_categories above; whitelisted/probationary both pass. The old
            # filter dropped categories absent from BOTH sets — i.e. ones with no
            # resolved data — which created a chicken-and-egg: a newly-active venue
            # (Kalshi) could never build a record in a category it was filtered
            # out of. Such unknown categories now stay explorable (still gated by
            # the edge bar + all risk checks downstream).

            reason = self._is_junk_market(m)
            if reason:
                log.info("engine.filtered_no_edge", market_id=m.id, reason=reason)
                filtered_count += 1
            else:
                edge_candidates.append(m)

        # Skip markets analyzed recently (within cache TTL)
        recent_rows = await self.db.fetchall(
            """SELECT DISTINCT market_id FROM signals
               WHERE timestamp > datetime('now', '-15 minutes')"""
        )
        recently_analyzed = {r["market_id"] for r in recent_rows}
        fresh_candidates = [m for m in edge_candidates if m.id not in recently_analyzed]

        # Fall back to all candidates if everything was recently analyzed
        if not fresh_candidates:
            fresh_candidates = edge_candidates

        # Block markets whose event was recently rebalanced (prevents buy-sell loops)
        try:
            blocked_rows = await self.db.fetchall(
                "SELECT event_key FROM rebalance_blocks WHERE blocked_until > datetime('now')"
            )
            blocked_events = {r["event_key"] for r in blocked_rows}
            if blocked_events:
                before = len(fresh_candidates)
                fresh_candidates = [
                    m for m in fresh_candidates
                    if self._get_event_key(m.id) not in blocked_events
                ]
                blocked_count = before - len(fresh_candidates)
                if blocked_count > 0:
                    log.info("engine.rebalance_blocked", count=blocked_count, events=sorted(blocked_events))
                    filtered_count += blocked_count
        except Exception:
            pass  # Table may not exist yet

        # Block markets whose order build recently failed (e.g. below CLOB minimum)
        try:
            drop_rows = await self.db.fetchall(
                "SELECT market_id FROM order_build_drops WHERE blocked_until > datetime('now')"
            )
            dropped_markets = {r["market_id"] for r in drop_rows}
            if dropped_markets:
                before = len(fresh_candidates)
                fresh_candidates = [m for m in fresh_candidates if m.id not in dropped_markets]
                drop_count = before - len(fresh_candidates)
                if drop_count > 0:
                    log.info("engine.order_build_blocked", count=drop_count, markets=sorted(dropped_markets))
                    filtered_count += drop_count
        except Exception:
            pass  # Table may not exist yet

        show_scan_results(len(markets), len(fresh_candidates), filtered_count, exchange=self.exchange_name)

        # Smart ranking — prioritize markets most likely to be mispriced
        from auramaur.strategy.market_selector import rank_markets
        import random

        price_history = await self._get_price_history(hours=24)
        ranked = rank_markets(fresh_candidates, price_history=price_history)
        max_markets = self.settings.nlp.max_markets_per_cycle

        # Deduplicate by event — max 2 variants per underlying event
        # Prevents putting all eggs in one basket (e.g. 5 Taylor Swift bridesmaids)
        _MAX_PER_EVENT = 2
        event_counts: dict[str, int] = {}
        deduped: list = []
        for market, score in ranked:
            # Extract event base: "KXNEWPOPE-70-PPIZ" → "KXNEWPOPE-70"
            # For Polymarket numeric IDs, use question-based grouping
            mid = market.id
            if mid[0:2] == "KX" and mid.count("-") >= 2:
                event_key = mid.rsplit("-", 1)[0]
            else:
                # Polymarket: use first 6 words of question as event key
                event_key = " ".join(market.question.lower().split()[:6])
            count = event_counts.get(event_key, 0)
            if count < _MAX_PER_EVENT:
                deduped.append((market, score))
                event_counts[event_key] = count + 1
        ranked = deduped

        # Shuffle within similar scores to avoid always picking the same ones
        if len(ranked) > max_markets:
            top_batch = ranked[:max_markets * 2]
            random.shuffle(top_batch)
            ranked = top_batch

        # Promote news-flagged markets to the front of the batch so fresh
        # headlines get evaluated first. News flagging replaces the old
        # per-market analyze_market path from NewsReactor, which spent two
        # Claude calls per headline; the flagged markets now ride the single
        # strategic batch call.
        flagged_ids = self._prune_news_flags()
        if flagged_ids:
            flagged_entries = [(m, s) for m, s in ranked if m.id in flagged_ids]
            other_entries = [(m, s) for m, s in ranked if m.id not in flagged_ids]
            if flagged_entries:
                ranked = flagged_entries + other_entries
                log.info(
                    "engine.news_flagged_promoted",
                    count=len(flagged_entries),
                    market_ids=[m.id for m, _ in flagged_entries[:5]],
                )

        trade_candidates: list[TradeCandidate] = []

        if self.market_analyzer:
            # Protocol-based path: analyzer returns TradeCandidates,
            # engine handles risk + allocation + execution
            analysis_markets = [m for m, _score in ranked[:max_markets]]
            trade_candidates = await self.market_analyzer.analyze_markets(
                analysis_markets, price_history=price_history,
            )
        elif self.strategic:
            # Strategic mode: batch analysis with world model + allocate
            # (Execution happens inside _run_cycle_strategic)
            results = await self._run_cycle_strategic(ranked[:max_markets], price_history=price_history)
        elif self.allocator:
            # Legacy: sequential analyze + trade one at a time
            results = []
            for market, _score in ranked[:max_markets]:
                try:
                    result = await self.analyze_market(market, price_history=price_history)
                    if result:
                        results.append(result)
                except Exception as e:
                    log.error("engine.market_error", market_id=market.id, error=str(e))

        # Technical Strategy (The Second Leg):
        # Run rule-based technical analysis on all fresh candidates.
        # This analyzer is fast and free (no LLM), so it can scan a
        # larger universe.
        if self.technical_analyzer:
            tech_candidates = await self.technical_analyzer.analyze_markets(
                fresh_candidates, price_history=price_history
            )
            if tech_candidates:
                # Filter out any that were already analyzed by the LLM
                # to prevent duplicate signals/executions.
                # (Note: for strategic mode, we'd need to track which markets
                # were already traded. For now, we just merge candidates if
                # they exist).
                llm_ids = {tc.market.id for tc in trade_candidates}
                unique_tech = [
                    tc for tc in tech_candidates
                    if tc.market.id not in llm_ids
                ]
                if unique_tech:
                    log.info(
                        "engine.technical_signals_detected",
                        count=len(unique_tech),
                    )
                    # For strategic mode, _run_cycle_strategic already executed.
                    # We need to execute these tech candidates now.
                    tech_results = await self._execute_candidates(unique_tech, price_history)
                    if 'results' in locals():
                        results.extend(tech_results)
                    else:
                        results = tech_results

        if trade_candidates and not results:
            results = await self._execute_candidates(trade_candidates, price_history)

        elapsed = time.monotonic() - start
        trades = [r for r in results if r.get("order")]
        show_cycle_summary(len(results), len(trades), elapsed, exchange=self.exchange_name)

        return results

    async def _execute_candidates(
        self,
        trade_candidates: list[TradeCandidate],
        price_history: dict[str, list[float]] | None = None,
    ) -> list[dict]:
        """Run risk checks + allocation + execution on TradeCandidate objects.

        This is the shared downstream path used by any MarketAnalyzer
        implementation.  The analyzer produces candidates; this method
        decides which ones to trade and at what size.
        """
        results: list[dict] = []
        alloc_candidates: list[CandidateTrade] = []
        cycle_cash = await self._get_available_cash()

        for tc in trade_candidates:
            # Ensure market exists in DB (FK requirement)
            m = tc.market
            await self.db.execute(
                """INSERT OR IGNORE INTO markets (id, exchange, condition_id, question, description,
                   category, active, outcome_yes_price, outcome_no_price,
                   volume, liquidity, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, datetime('now'))""",
                (m.id, m.exchange or self.exchange_name or "polymarket",
                 m.condition_id, m.question, m.description[:500],
                 m.category, m.outcome_yes_price, m.outcome_no_price,
                 m.volume, m.liquidity),
            )
            # Store signal
            await self.db.execute(
                """INSERT INTO signals (market_id, claude_prob, claude_confidence, market_prob,
                                         edge, evidence_summary, action, strategy_source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (tc.signal.market_id, tc.signal.claude_prob, tc.signal.claude_confidence.value,
                 tc.signal.market_prob, tc.signal.edge, tc.signal.evidence_summary,
                 tc.signal.recommended_side.value if tc.signal.recommended_side else None,
                 tc.signal.strategy_source),
            )
            await self.db.commit()

            # Risk evaluation (pass actual cash for correct Kelly sizing)
            decision = await self.risk_manager.evaluate(
                tc.signal, tc.market, price_history=price_history,
                available_cash=cycle_cash,
            )
            checks_passed = sum(1 for c in decision.checks if c.passed)
            checks_failed = sum(1 for c in decision.checks if not c.passed)
            show_risk_decision(
                decision.approved, decision.reason,
                checks_passed, checks_failed, decision.position_size,
            )

            result = {"market": tc.market, "signal": tc.signal, "decision": decision, "order": None}
            results.append(result)

            if decision.approved and decision.position_size > 0 and self.allocator:
                ev = CapitalAllocator.compute_expected_value(tc.signal, decision.position_size)
                alloc_candidates.append(CandidateTrade(
                    market=tc.market, signal=tc.signal, risk_decision=decision,
                    kelly_size=decision.position_size, expected_value=ev,
                ))

        # Allocate and execute
        if alloc_candidates and self.allocator:
            positions, cash = await self._get_positions_and_cash()

            allocated = self.allocator.allocate(alloc_candidates, cash, positions)
            for candidate in allocated:
                try:
                    order_result = await self._build_and_place_order(
                        candidate.signal, candidate.market, candidate.allocated_size,
                    )
                    if order_result:
                        for r in results:
                            if r["market"].id == candidate.market.id:
                                r["order"] = order_result
                                break
                except Exception as e:
                    log.error("engine.execute_error", market_id=candidate.market.id, error=str(e))

        return results

    async def _run_cycle_strategic(
        self,
        ranked_markets: list,
        price_history: dict[str, list[float]] | None = None,
    ) -> list[dict]:
        """Strategic cycle: batch analysis with persistent world model."""
        from pathlib import Path
        from auramaur.nlp.strategic import StrategicAnalyzer
        from auramaur.exchange.models import Confidence
        exchange_fees = self.settings.arbitrage.exchange_fees

        # Kill switch check — halt immediately if active
        if Path("KILL_SWITCH").exists():
            log.warning("engine.kill_switch_active", method="strategic")
            return []

        # Cap batch to 12 markets to keep prompt under context window
        # (35 markets × 12k chars each = 420k chars → too large for Opus)
        _MAX_STRATEGIC_BATCH = 12
        markets = [m for m, _score in ranked_markets[:_MAX_STRATEGIC_BATCH]]

        # Gather evidence for all markets first
        from auramaur.data_sources.base import NewsItem as _NI
        from auramaur.nlp.query_decomposer import extract_search_queries

        # Evidence gathering is the dominant cycle cost — each market fans out
        # to ~15 data sources. Running markets concurrently (instead of one at a
        # time) cuts cycle wall-clock from ~10min toward ~2-3min, which is what
        # makes the news-speed pillar actually fast. The CLOB description enrich
        # is a blocking, not-thread-safe call, so it's serialized behind a lock
        # and pushed to a worker thread to keep the event loop responsive.
        _EVIDENCE_CONCURRENCY = 6
        sem = asyncio.Semaphore(_EVIDENCE_CONCURRENCY)
        clob_lock = asyncio.Lock()

        async def _gather_market_evidence(market) -> tuple[str, list] | None:
            if self._is_junk_market(market):
                return None
            if not self._try_claim_market(market.id):
                return None
            async with sem:
                try:
                    # Enrich market description from CLOB if thin
                    if len(market.description) < 50 and market.condition_id:
                        try:
                            def _enrich():
                                self.exchange._init_clob_client()
                                return self.exchange._clob_client.get_market(market.condition_id)
                            async with clob_lock:
                                clob_info = await asyncio.to_thread(_enrich)
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

                    for query in queries:
                        items = await self.aggregator.gather(
                            query, limit_per_source=8, category=market.category or None,
                        )
                        for item in items:
                            if item.id not in seen_ids:
                                seen_ids.add(item.id)
                                all_evidence.append(item)
                    # Cap evidence per market to keep total prompt under context window
                    # 12 markets × 8 items × ~500 chars = ~48k chars of evidence
                    return (market.id, all_evidence[:8])
                except Exception as e:
                    log.error("strategic.evidence_error", market_id=market.id, error=str(e))
                    return (market.id, [])

        gathered = await asyncio.gather(*[_gather_market_evidence(m) for m in markets])
        evidence_map: dict[str, list] = {}
        for _res in gathered:
            if _res is not None:
                evidence_map[_res[0]] = _res[1]

        # Filter to markets with evidence gathered
        batch_markets = [m for m in markets if m.id in evidence_map]
        if not batch_markets:
            return []

        # Batch analysis with world model
        strategic: StrategicAnalyzer = self.strategic
        analysis = await strategic.analyze_batch_with_adversarial(batch_markets, evidence_map)

        if not analysis.markets:
            log.warning("strategic.no_market_results", batch_size=len(batch_markets))
            return []

        # Tool-use refinement: for top-edge markets (per settings), re-ask
        # Claude with WebSearch/WebFetch enabled. Replaces the batch result
        # for those markets only. Fails open to batch_result on any error.
        analysis.markets = await self._maybe_refine_with_tool_use(
            analysis.markets, batch_markets,
        )

        log.info(
            "strategic.processing_results",
            result_count=len(analysis.markets),
            result_ids=[m.market_id for m in analysis.markets[:5]],
            batch_ids=[m.id for m in batch_markets[:5]],
        )

        # Convert strategic results to signals and run risk checks
        results: list[dict] = []
        candidates: list[CandidateTrade] = []
        cycle_cash = await self._get_available_cash()

        for batch_result in analysis.markets:
            market = next((m for m in batch_markets if m.id == batch_result.market_id), None)
            if market is None:
                log.debug("strategic.market_id_mismatch", result_id=batch_result.market_id)
                continue

            claude_prob = batch_result.probability
            # Apply Platt scaling calibration to raw probability
            if self.calibration:
                claude_prob = await self.calibration.adjust(claude_prob, market.category or "")
            market_prob = market.outcome_yes_price

            # Edge calculation (same as detect_edge)
            raw_edge = claude_prob - market_prob
            log.info("strategic.edge_calc", market_id=market.id,
                     claude=round(claude_prob, 3), market=round(market_prob, 3),
                     raw_edge=round(raw_edge, 3))
            if abs(raw_edge) < 0.001:
                continue
            side = OrderSide.BUY if raw_edge > 0 else OrderSide.SELL
            # Fee coefficient applies to per-contract P*(1-P), not a flat
            # percentage (see detect_edge in signals.py).
            fee_rate = exchange_fees.get(market.exchange or self.exchange_name, 0.0)
            edge = abs(raw_edge) - fee_rate * market_prob * (1.0 - market_prob)

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

            from auramaur.monitoring.display import show_analysis
            show_analysis(
                signal.claude_prob, signal.market_prob, signal.edge,
                batch_result.confidence, None, None,
            )

            # Ensure market exists in DB (FK requirement for signals table)
            await self.db.execute(
                """INSERT OR IGNORE INTO markets (id, condition_id, question, description,
                   category, active, outcome_yes_price, outcome_no_price,
                   volume, liquidity, last_updated)
                   VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, datetime('now'))""",
                (market.id, market.condition_id, market.question,
                 market.description[:500], market.category,
                 market.outcome_yes_price, market.outcome_no_price,
                 market.volume, market.liquidity),
            )

            # Store signal
            await self.db.execute(
                """INSERT INTO signals (market_id, claude_prob, claude_confidence, market_prob,
                                         edge, evidence_summary, action, strategy_source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (signal.market_id, signal.claude_prob, signal.claude_confidence.value,
                 signal.market_prob, signal.edge, signal.evidence_summary,
                 signal.recommended_side.value if signal.recommended_side else None,
                 "llm"),
            )
            await self.db.commit()

            # Risk evaluation (pass actual cash for correct Kelly sizing)
            decision = await self.risk_manager.evaluate(signal, market, price_history=price_history, available_cash=cycle_cash)
            from auramaur.monitoring.display import show_risk_decision
            checks_passed = sum(1 for c in decision.checks if c.passed)
            checks_failed = sum(1 for c in decision.checks if not c.passed)
            show_risk_decision(decision.approved, decision.reason, checks_passed, checks_failed, decision.position_size)

            result = {"market": market, "signal": signal, "decision": decision, "order": None}
            results.append(result)

            if decision.approved and decision.position_size > 0 and self.allocator:
                ev = CapitalAllocator.compute_expected_value(signal, decision.position_size)
                candidates.append(CandidateTrade(
                    market=market, signal=signal, risk_decision=decision,
                    kelly_size=decision.position_size, expected_value=ev,
                ))

        # Allocate and execute
        if candidates and self.allocator:
            positions, cash = await self._get_positions_and_cash()

            allocated = self.allocator.allocate(candidates, cash, positions)
            for candidate in allocated:
                try:
                    order_result = await self._build_and_place_order(
                        candidate.signal, candidate.market, candidate.allocated_size,
                    )
                    if order_result:
                        for r in results:
                            if r["market"].id == candidate.market.id:
                                r["order"] = order_result
                                break
                except Exception as e:
                    log.error("strategic.execute_error", market_id=candidate.market.id, error=str(e))

        return results

    async def _run_cycle_allocated(
        self,
        ranked_markets: list,
        price_history: dict[str, list[float]] | None = None,
    ) -> list[dict]:
        """Two-phase cycle: evaluate all candidates, then allocate and execute."""
        # Phase 1: Evaluate (no orders placed)
        evaluated: list[dict] = []
        for market, _score in ranked_markets:
            try:
                result = await self.analyze_market(market, place_order=False, price_history=price_history)
                if result:
                    evaluated.append(result)
            except Exception as e:
                log.error("engine.market_error", market_id=market.id, error=str(e))

        # Collect approved candidates
        candidates: list[CandidateTrade] = []
        for r in evaluated:
            decision = r["decision"]
            if decision.approved and decision.position_size > 0:
                signal = r["signal"]
                ev = CapitalAllocator.compute_expected_value(signal, decision.position_size)
                candidates.append(CandidateTrade(
                    market=r["market"],
                    signal=signal,
                    risk_decision=decision,
                    kelly_size=decision.position_size,
                    expected_value=ev,
                ))

        if not candidates:
            return evaluated

        # Phase 2: Allocate capital across candidates
        current_positions, cash = await self._get_positions_and_cash()

        allocated = self.allocator.allocate(candidates, cash, current_positions)

        # Phase 3: Execute allocated trades via smart router
        results = list(evaluated)  # Include all evaluated (even rejected) for reporting
        for candidate in allocated:
            try:
                order_result = await self._build_and_place_order(
                    candidate.signal, candidate.market, candidate.allocated_size,
                )
                if order_result:
                    # Update the result dict for this market
                    for r in results:
                        if r["market"].id == candidate.market.id:
                            r["order"] = order_result
                            break
            except Exception as e:
                log.error("engine.execute_error", market_id=candidate.market.id, error=str(e))

        return results

    async def _get_price_history(self, hours: int = 24) -> dict[str, list[float]]:
        """Load recent price history for momentum scoring."""
        rows = await self.db.fetchall(
            """SELECT market_id, price FROM price_history
               WHERE recorded_at > datetime('now', ?)
               ORDER BY market_id, recorded_at""",
            (f"-{hours} hours",),
        )
        history: dict[str, list[float]] = {}
        for row in rows:
            history.setdefault(row["market_id"], []).append(row["price"])
        return history
