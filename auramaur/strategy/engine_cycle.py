"""Trading-cycle orchestration — extracted from TradingEngine (Phase 5 split).

Pure structural move: the cycle drivers (run_cycle and its starved / strategic /
allocated variants), candidate execution, and the price-history helper live here
as CycleOrchestrationMixin, mixed into TradingEngine. Behavior is unchanged — the
methods still operate on the engine's self (analyze_market, scan, risk, the
execution gateway, the allocator). Method-local imports moved with the methods.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from auramaur.broker.allocator import CandidateTrade, CapitalAllocator
from auramaur.killswitch import kill_switch_present
from auramaur.exchange.models import Market, OrderSide, Signal
from auramaur.monitoring.display import (
    show_cycle_summary,
    show_risk_decision,
    show_scan_results,
)
from auramaur.nlp.query_decomposer import extract_search_queries
from auramaur.strategy.classifier import blocked_category_hit, ensure_category
from auramaur.strategy.signals import taker_fee_rate

if TYPE_CHECKING:
    from auramaur.strategy.protocols import TradeCandidate

log = structlog.get_logger()


class CycleOrchestrationMixin:
    """Trading-cycle orchestration for TradingEngine (see module docstring)."""

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
        # Live mode: skip candidates outside the live allowlist up front —
        # the gateway's check_category_allowlist would reject them anyway,
        # so analyzing them only burns LLM calls. The gateway remains the
        # enforcement point (this is a pure efficiency filter, like
        # _held_market_ids).
        allowed_live = (set(self.settings.risk.allowed_categories_live)
                        if self.settings.is_live else None)

        for m in candidates:
            # Classify-before-block like the gateway: a market stored 'other'
            # that is really sports slips past a raw `category in blocked` test,
            # so check the stored label OR a fresh classification (mislabel-safe).
            if blocked_category_hit(blocked, m.question, m.description, m.category):
                filtered_count += 1
                continue
            if allowed_live is not None and m.category not in allowed_live:
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

        # Drop markets we already hold — the allocator won't average into an
        # open position, so analyzing them wastes LLM calls and produces
        # approved-but-unexecutable candidates. Exits are managed elsewhere.
        held_ids = await self._held_market_ids()
        if held_ids:
            before = len(fresh_candidates)
            fresh_candidates = [m for m in fresh_candidates if m.id not in held_ids]
            held_filtered = before - len(fresh_candidates)
            if held_filtered > 0:
                log.info("engine.held_filtered", count=held_filtered)
                filtered_count += held_filtered

        # Bench markets the risk gateway recently rejected — their verdict
        # can't flip until something moves, so re-analyzing them every cycle
        # (the recently-analyzed fallback above resurrects them once the
        # 15-minute window lapses) is pure evidence+LLM burn. Applied after
        # that fallback so a fully-benched venue stays quiet rather than
        # re-running its dud markets.
        fresh_candidates, benched_count = await self._apply_rejection_cooldown(fresh_candidates)
        filtered_count += benched_count

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
                 ensure_category(m.question, m.description, m.category),
                 m.outcome_yes_price, m.outcome_no_price,
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
                market_id=tc.signal.market_id, strategy=tc.signal.strategy_source,
                graduation=getattr(decision, "graduation_status", ""),
                mispricing=getattr(tc.signal, "mispricing_reason", ""),
            )
            await self._record_rejection_state(tc.market, decision.approved, decision.reason)

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
                        force_paper=candidate.risk_decision.force_paper,
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
        from auramaur.nlp.strategic import StrategicAnalyzer
        from auramaur.exchange.models import Confidence
        exchange_fees = self.settings.arbitrage.exchange_fees

        # Kill switch check — halt immediately if active
        if kill_switch_present():
            log.warning("engine.kill_switch_active", method="strategic")
            return []

        # Cap batch to 12 markets to keep prompt under context window
        # (35 markets × 12k chars each = 420k chars → too large for Opus)
        _MAX_STRATEGIC_BATCH = 12
        markets = [m for m, _score in ranked_markets[:_MAX_STRATEGIC_BATCH]]

        # Gather evidence for all markets first

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
                    # Cap fan-out by config (mirrors the tool-use path above) so a
                    # market with many queries doesn't over-fetch across sources.
                    per_query_limit = (
                        max(1, self.settings.nlp.evidence_per_source // len(queries))
                        if queries else self.settings.nlp.evidence_per_source
                    )

                    # Resolution criteria are carried in the market block's Desc
                    # field, not duplicated here as synthetic evidence — the
                    # Evidence section is real-world news only.
                    for query in queries:
                        items = await self.aggregator.gather(
                            query, limit_per_source=per_query_limit, category=market.category or None,
                        )
                        for item in items:
                            if item.id not in seen_ids:
                                seen_ids.add(item.id)
                                all_evidence.append(item)

                    # Lever A: globally re-rank the merged candidate set against
                    # the market question (recency x authority x relevance) and
                    # keep the best N — not the first N by query order.
                    from auramaur.nlp.evidence_ranker import rank_evidence
                    nlp = self.settings.nlp
                    ranked = rank_evidence(
                        market.question,
                        all_evidence,
                        top_n=nlp.evidence_top_n,
                        backend=nlp.relevance_backend,
                        model_name=nlp.embedding_model,
                    )
                    return (market.id, ranked)
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

            raw_prob = batch_result.probability
            claude_prob = raw_prob
            # Apply Platt scaling calibration to raw probability
            if self.calibration:
                claude_prob = await self.calibration.adjust(raw_prob, market.category or "")
                # Record the raw prediction so (a) the resolution tracker can
                # later settle this market and (b) Platt scaling can learn from
                # the outcome. The legacy analyze_market path records this too,
                # but it only runs for Polymarket (the price-monitor WebSocket
                # path). The strategic path is the ONLY path Kalshi and
                # Crypto.com take, so without this their markets never enter the
                # calibration table and consequently never get settled — their
                # realized P&L was silently never booked.
                await self.calibration.record_prediction(
                    market.id, raw_prob, market.category or ""
                )
            market_prob = market.outcome_yes_price

            # Edge calculation (same as detect_edge)
            raw_edge = claude_prob - market_prob
            log.info("strategic.edge_calc", market_id=market.id,
                     claude=round(claude_prob, 3), market=round(market_prob, 3),
                     raw_edge=round(raw_edge, 3))
            if abs(raw_edge) < 0.001:
                continue
            side = OrderSide.BUY if raw_edge > 0 else OrderSide.SELL
            # Taker fee coefficient applies to per-contract P*(1-P), not a flat
            # percentage. Polymarket is per-category (see taker_fee_rate); this
            # is a crossing execution path so the taker rate is the right one.
            fee_rate = taker_fee_rate(
                market.exchange or self.exchange_name, market.category, exchange_fees)
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
                 market.description[:500],
             ensure_category(market.question, market.description, market.category),
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
            show_risk_decision(
            decision.approved, decision.reason, checks_passed,
            checks_failed, decision.position_size, market_id=signal.market_id,
            strategy=signal.strategy_source,
            graduation=getattr(decision, "graduation_status", ""),
            mispricing=getattr(signal, "mispricing_reason", ""))
            await self._record_rejection_state(market, decision.approved, decision.reason)

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
                        force_paper=candidate.risk_decision.force_paper,
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
                    force_paper=candidate.risk_decision.force_paper,
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
