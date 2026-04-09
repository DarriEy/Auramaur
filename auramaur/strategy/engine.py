"""Trade decision orchestrator — connects analysis to execution."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

import fcntl
import os
import tempfile

from auramaur.data_sources.aggregator import Aggregator
from auramaur.db.database import Database
from auramaur.exchange.models import Market, Order, OrderSide, TokenType
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
        self.exchange_name = ""  # Set by bot after init (e.g. "polymarket", "kalshi")
        self.strategic = None  # StrategicAnalyzer, set by bot after init
        self._components_pnl = None  # Set by bot after init

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
                                     edge, second_opinion_prob, divergence, evidence_summary, action)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal.market_id, signal.claude_prob, signal.claude_confidence.value,
                signal.market_prob, signal.edge, signal.second_opinion_prob,
                signal.divergence, signal.evidence_summary,
                signal.recommended_side.value if signal.recommended_side else None,
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
        from auramaur.exchange.models import OrderResult

        if self.router:
            order = await self.router.route(signal, market, size_dollars, self.settings.is_live)
        else:
            order = self.exchange.prepare_order(signal, market, size_dollars, self.settings.is_live)

        if order is None:
            show_order_dropped(market.id, f"order build failed (${size_dollars:.2f} too small for CLOB minimum)")
            log.warning(
                "engine.order_dropped",
                market_id=market.id,
                size_dollars=size_dollars,
                reason="prepare_order returned None (likely below CLOB minimum or router rejection)",
            )
            # Block re-evaluation for 24 hours — price won't change enough to matter
            # and each re-analysis wastes a Claude API call for nothing.
            try:
                await self.db.execute(
                    """INSERT OR REPLACE INTO order_build_drops
                       (market_id, blocked_until, reason)
                       VALUES (?, datetime('now', '+24 hours'), ?)""",
                    (market.id, f"order build failed at ${size_dollars:.2f}"),
                )
                await self.db.commit()
            except Exception:
                pass  # Table may not exist yet
            return None

        result = await self.exchange.place_order(order)
        show_order(result.status, result.order_id, order.side.value, order.size, order.price, result.is_paper, exchange=self.exchange_name, error_message=result.error_message)

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

        # Log slippage
        if result.status in ("filled", "paper", "pending") and result.filled_price > 0:
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

        # Record fill for P&L tracking
        if result.status in ("filled", "paper", "pending"):
            from auramaur.exchange.models import Fill
            fill = Fill(
                order_id=result.order_id,
                market_id=order.market_id,
                token_id=order.token_id,
                side=order.side,
                token=order.token,
                size=result.filled_size if result.filled_size > 0 else order.size,
                price=result.filled_price if result.filled_price > 0 else order.price,
                is_paper=result.is_paper,
            )
            pnl_tracker = self._components_pnl
            if pnl_tracker:
                await pnl_tracker.record_fill(fill)

        # Record trade metadata for later PnL attribution
        await self._record_trade_for_attribution(market, signal,
            type("D", (), {"position_size": size_dollars}))

        return result

    async def _record_trade_for_attribution(self, market: Market, signal, decision) -> None:
        """Store trade metadata for later PnL attribution when the market resolves."""
        # Determine token type based on signal side
        from auramaur.exchange.models import TokenType
        if signal.recommended_side == OrderSide.SELL:
            token = TokenType.NO.value
            price = market.outcome_no_price if market.outcome_no_price > 0.01 else (1.0 - market.outcome_yes_price)
        else:
            token = TokenType.YES.value
            price = market.outcome_yes_price
        token_id = market.clob_token_yes if token == "YES" else market.clob_token_no

        await self.db.execute(
            """INSERT INTO portfolio
               (market_id, side, size, avg_price, current_price, category, token, token_id, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(market_id) DO UPDATE SET
                   size = excluded.size,
                   avg_price = excluded.avg_price,
                   current_price = excluded.current_price,
                   category = excluded.category,
                   token = excluded.token,
                   token_id = excluded.token_id,
                   updated_at = excluded.updated_at""",
            (
                market.id,
                "BUY",  # Always BUY on Polymarket
                decision.position_size,
                price, price,
                market.category,
                token, token_id,
            ),
        )
        await self.db.commit()

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

        When cash_available is provided and below $5, switches to
        "starved mode": only re-prices held positions and checks
        whether any new opportunity is so compelling it's worth
        liquidating an existing stake.

        When the engine's own exchange balance is below $1 (and there's
        no syncer, e.g. Kalshi), skip the scan entirely to avoid wasting
        Claude API calls on signals that can never become orders.
        """
        if not getattr(self, '_components_syncer', None):
            try:
                own_cash = await self._get_available_cash()
                if own_cash < 1.0:
                    log.info(
                        "engine.skip_no_capital",
                        exchange=self.exchange_name,
                        cash=own_cash,
                    )
                    return []
            except Exception:
                pass
        if cash_available is not None and cash_available < 5.0:
            return await self._run_cycle_starved()

        import time
        start = time.monotonic()

        markets = await self.scan_and_store_markets()

        candidates = [
            m for m in markets
            if m.active
            # Use the HIGHER of liquidity and volume as the activity measure.
            # Polymarket reports deep liquidity; Kalshi reports thin top-of-book
            # liquidity but high volume on active markets. Using max() ensures
            # active Kalshi markets aren't filtered out by the Polymarket-tuned
            # min_liquidity threshold.
            and max(m.liquidity or 0, m.volume or 0) >= self.settings.risk.min_liquidity
            and m.spread <= self.settings.risk.max_spread_pct / 100
            and self.settings.risk.implied_prob_min <= m.outcome_yes_price <= self.settings.risk.implied_prob_max
            # Skip near-dead markets (no volume = orders won't fill)
            and m.volume >= 100
        ]

        # --- Load avoid-categories from performance feedback ---
        avoid_categories: set[str] = set()
        try:
            from auramaur.broker.feedback import PerformanceFeedback
            feedback = PerformanceFeedback(self.db)
            avoid_categories = await feedback.get_avoid_categories()
            if avoid_categories:
                log.info("engine.avoid_categories", categories=sorted(avoid_categories))
        except Exception as e:
            log.debug("engine.feedback_import_error", error=str(e))

        # --- Filter markets where Claude has no informational edge ---
        edge_candidates: list[Market] = []
        filtered_count = 0

        for m in candidates:
            # Check performance-based avoid list first
            if m.category in avoid_categories:
                log.info(
                    "engine.filtered_poor_performance",
                    market_id=m.id,
                    category=m.category,
                )
                filtered_count += 1
                continue

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

        if self.market_analyzer:
            # Protocol-based path: analyzer returns TradeCandidates,
            # engine handles risk + allocation + execution
            analysis_markets = [m for m, _score in ranked[:max_markets]]
            trade_candidates = await self.market_analyzer.analyze_markets(
                analysis_markets, price_history=price_history,
            )
            results = await self._execute_candidates(trade_candidates, price_history)
        elif self.strategic:
            # Strategic mode: batch analysis with world model + allocate
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
        from auramaur.exchange.models import Signal

        results: list[dict] = []
        alloc_candidates: list[CandidateTrade] = []

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
                                         edge, evidence_summary, action)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (tc.signal.market_id, tc.signal.claude_prob, tc.signal.claude_confidence.value,
                 tc.signal.market_prob, tc.signal.edge, tc.signal.evidence_summary,
                 tc.signal.recommended_side.value if tc.signal.recommended_side else None),
            )
            await self.db.commit()

            # Risk evaluation (pass actual cash for correct Kelly sizing)
            if not hasattr(self, '_cached_cycle_cash'):
                self._cached_cycle_cash = await self._get_available_cash()
            decision = await self.risk_manager.evaluate(
                tc.signal, tc.market, price_history=price_history,
                available_cash=self._cached_cycle_cash,
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
        from auramaur.exchange.models import Confidence, Signal, OrderSide
        from auramaur.strategy.signals import EXCHANGE_FEES

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
        evidence_map: dict[str, list] = {}
        for market in markets:
            skip = self._is_junk_market(market)
            if skip:
                continue
            if not self._try_claim_market(market.id):
                continue
            try:
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

                for query in queries:
                    items = await self.aggregator.gather(query, limit_per_source=8)
                    for item in items:
                        if item.id not in seen_ids:
                            seen_ids.add(item.id)
                            all_evidence.append(item)
                # Cap evidence per market to keep total prompt under context window
                # 12 markets × 5 items × ~500 chars = ~30k chars of evidence
                evidence_map[market.id] = all_evidence[:5]
            except Exception as e:
                log.error("strategic.evidence_error", market_id=market.id, error=str(e))
                evidence_map[market.id] = []

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

        log.info(
            "strategic.processing_results",
            result_count=len(analysis.markets),
            result_ids=[m.market_id for m in analysis.markets[:5]],
            batch_ids=[m.id for m in batch_markets[:5]],
        )

        # Convert strategic results to signals and run risk checks
        results: list[dict] = []
        candidates: list[CandidateTrade] = []

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
            fee_rate = EXCHANGE_FEES.get(market.exchange or self.exchange_name, 0.0)
            edge = abs(raw_edge) - fee_rate

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
                                         edge, evidence_summary, action)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (signal.market_id, signal.claude_prob, signal.claude_confidence.value,
                 signal.market_prob, signal.edge, signal.evidence_summary,
                 signal.recommended_side.value if signal.recommended_side else None),
            )
            await self.db.commit()

            # Risk evaluation (pass actual cash for correct Kelly sizing)
            if not hasattr(self, '_cached_cycle_cash'):
                self._cached_cycle_cash = await self._get_available_cash()
            decision = await self.risk_manager.evaluate(signal, market, price_history=price_history, available_cash=self._cached_cycle_cash)
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
