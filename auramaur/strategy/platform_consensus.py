"""Platform consensus follower strategy — follow crowd forecasting consensus on Manifold and Metaculus."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

import structlog

from auramaur.broker.execution_gateway import ExecutionGateway, TradeIntent
from auramaur.data_sources.manifold import ManifoldSource
from auramaur.data_sources.metaculus import MetaculusSource
from auramaur.exchange.models import (
    Confidence,
    Market,
    OrderSide,
    Signal,
)
from auramaur.strategy.classifier import blocked_category_hit, ensure_category
from auramaur.experiments.strategies.platform_consensus import (
    ConsensusForecast,
    ConsensusInputs,
    ConsensusRules,
    assess_platform_consensus,
    clean_title,
    parse_probability,
)
from auramaur.strategy.protocols import ExecutionMode
from auramaur.strategy.signals import taker_fee_rate

log = structlog.get_logger()


class PlatformConsensusPillar:
    """Strategy pillar that aligns prediction market prices with play-money consensus."""

    # Uniform Strategy contract (see strategy/protocols.py).
    name = "platform_consensus"
    execution_mode = ExecutionMode.GATEWAY_SINGLE

    def __init__(
        self,
        db,
        settings,
        discovery,
        exchange,
        risk_manager,
        pnl_tracker,
        calibration,
    ) -> None:
        self._db = db
        self._settings = settings
        self._discovery = discovery
        self._exchange = exchange
        self._risk = risk_manager
        self._pnl = pnl_tracker
        self._calibration = calibration
        self._manifold = ManifoldSource()
        self._metaculus = MetaculusSource()
        self._gateway = None

    async def run_once(self) -> int:
        cfg = self._settings.platform_consensus
        if not cfg.enabled:
            return 0

        open_count = await self._open_position_count()
        if open_count >= cfg.max_open:
            log.info(
                "platform_consensus.cycle",
                scanned=0,
                entered=0,
                book_full=True,
                open=open_count,
                cap=cfg.max_open,
            )
            return 0

        markets = await self._discovery.get_markets(limit=cfg.scan_limit)
        from auramaur.data_edge import record_market_snapshot
        await record_market_snapshot(self._db, self.name, markets)

        eligible_markets = []
        for m in markets:
            if not self._eligible(m, cfg):
                continue
            if await self._already_entered_or_held(m.id):
                continue
            eligible_markets.append(m)

        # Rank by liquidity / volume descending to prioritize active markets
        eligible_markets.sort(
            key=lambda x: max(x.liquidity or 0.0, x.volume or 0.0), reverse=True
        )

        eval_limit = min(len(eligible_markets), cfg.max_entries_per_cycle * 2, 10)

        entered = 0
        for market in eligible_markets[:eval_limit]:
            if entered >= cfg.max_entries_per_cycle:
                break
            if open_count + entered >= cfg.max_open:
                break

            try:
                if await self._try_enter(market, cfg):
                    entered += 1
            except Exception as e:
                log.error(
                    "platform_consensus.market_error",
                    market_id=market.id,
                    error=str(e),
                    exc_info=True,
                )

            # Sleep between queries to be nice to public APIs
            await asyncio.sleep(1.0)

        # Log EVERY cycle (the settlement_arb #246 lesson): the previous
        # entered>0 guard made zero-entry cycles silent, so weeks of "running
        # fine, entering nothing" were indistinguishable from a dead task.
        self.last_cycle_detail = {"scanned": len(markets),
                                  "eligible": len(eligible_markets),
                                  "evaluated": eval_limit,
                                  "entered": entered, "open": open_count}
        log.info("platform_consensus.cycle_done", scanned=len(markets),
                 eligible=len(eligible_markets), evaluated=eval_limit,
                 entered=entered, open=open_count)
        return entered

    def _eligible(self, market: Market, cfg) -> bool:
        if not market.active:
            return False

        # Volume or liquidity threshold
        vol_or_liq = max(market.liquidity or 0.0, market.volume or 0.0)
        if vol_or_liq < cfg.min_liquidity:
            return False

        # Category blocklist
        blocked = set(self._settings.risk.blocked_categories)
        if blocked_category_hit(
            blocked, market.question, market.description, market.category
        ):
            return False

        # Time to resolution bounds
        if market.end_date is None:
            return False

        now = datetime.now(timezone.utc)
        end_date = market.end_date
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        hours_to_res = (end_date - now).total_seconds() / 3600.0
        if (
            hours_to_res < cfg.min_hours_to_resolution
            or hours_to_res > cfg.max_days_to_resolution * 24.0
        ):
            return False

        return True

    async def _already_entered_or_held(self, market_id: str) -> bool:
        row = await self._db.fetchone(
            "SELECT 1 FROM signals WHERE market_id = ? AND strategy_source = ? LIMIT 1",
            (market_id, self.name),
        )
        if row is not None:
            return True
        row = await self._db.fetchone(
            "SELECT 1 FROM portfolio WHERE market_id = ? LIMIT 1",
            (market_id,),
        )
        return row is not None

    async def _open_position_count(self) -> int:
        row = await self._db.fetchone(
            """SELECT COUNT(*) AS n FROM portfolio p
               WHERE p.size > 0
                 AND EXISTS (SELECT 1 FROM cost_basis cb
                             WHERE cb.market_id = p.market_id
                               AND UPPER(cb.token) = UPPER(p.token)
                               AND cb.is_paper = p.is_paper AND cb.size > 0)
                 AND EXISTS (SELECT 1 FROM trades t
                             WHERE t.market_id = p.market_id
                               AND t.strategy_source = ?
                               AND t.is_paper = p.is_paper)""",
            (self.name,),
        )
        return int(row["n"]) if row else 0

    def _get_clean_title(self, title: str) -> str:
        return clean_title(title)

    def _parse_probability_from_title(self, title: str) -> float | None:
        return parse_probability(title)

    @staticmethod
    def _quality_ok(item, source_name: str, cfg) -> bool:
        """Fail closed when a crowd forecast lacks the configured sample depth."""
        content = item.content or ""
        if source_name == "Manifold":
            bettors = re.search(r"Unique bettors:\s*([\d,]+)", content)
            liquidity = re.search(r"Liquidity:\s*\$([\d,]+(?:\.\d+)?)", content)
            if bettors is None or liquidity is None:
                return False
            return (
                int(bettors.group(1).replace(",", "")) >= cfg.min_manifold_bettors
                and float(liquidity.group(1).replace(",", ""))
                >= cfg.min_manifold_liquidity
            )
        forecasters = re.search(r"Forecasters:\s*([\d,]+)", content)
        if forecasters is None:
            return False
        return int(forecasters.group(1).replace(",", "")) >= cfg.min_metaculus_forecasters

    async def _try_enter(self, market: Market, cfg) -> bool:
        # Search Manifold
        manifold_items = []
        try:
            manifold_items = await self._manifold.fetch(market.question, limit=5)
        except Exception as e:
            log.debug(
                "platform_consensus.manifold_fetch_error",
                market_id=market.id,
                error=str(e),
            )

        # Search Metaculus
        metaculus_items = []
        try:
            metaculus_items = await self._metaculus.fetch(market.question, limit=5)
        except Exception as e:
            log.debug(
                "platform_consensus.metaculus_fetch_error",
                market_id=market.id,
                error=str(e),
            )

        from auramaur.data_edge import DataDelivery, record_delivery
        forecasts = manifold_items + metaculus_items
        observed = datetime.now(timezone.utc)
        published = [item.published_at for item in forecasts if item.published_at]
        await record_delivery(self._db, DataDelivery(
            strategy=self.name, component="platform_forecasts",
            status="ok" if forecasts else "empty",
            provider="manifold+metaculus", market_id=market.id,
            source_at=max(published) if published else observed,
            item_count=len(forecasts),
        ))

        market_p = market.outcome_yes_price
        fee = (
            taker_fee_rate(
                market.exchange or "polymarket",
                market.category or "",
                self._settings.arbitrage.exchange_fees,
            )
            * market_p
            * (1.0 - market_p)
        )
        blocked = set(self._settings.risk.blocked_categories)
        proposal = assess_platform_consensus(
            ConsensusInputs(
                market_id=market.id,
                question=market.question,
                market_probability=market_p,
                active=market.active,
                liquidity_or_volume=max(market.liquidity or 0.0, market.volume or 0.0),
                category_blocked=blocked_category_hit(
                    blocked, market.question, market.description, market.category
                ),
                end_at=market.end_date,
                as_of=observed,
                fee=fee,
                forecasts=tuple(
                    ConsensusForecast(item.title, item.content or "", source)
                    for source, items in (
                        ("Manifold", manifold_items),
                        ("Metaculus", metaculus_items),
                    )
                    for item in items
                ),
            ),
            ConsensusRules(
                min_liquidity=cfg.min_liquidity,
                min_hours_to_resolution=cfg.min_hours_to_resolution,
                max_days_to_resolution=cfg.max_days_to_resolution,
                match_threshold=cfg.match_threshold,
                min_edge=cfg.min_edge,
                min_manifold_bettors=cfg.min_manifold_bettors,
                min_manifold_liquidity=cfg.min_manifold_liquidity,
                min_metaculus_forecasters=cfg.min_metaculus_forecasters,
                stake_usd=cfg.stake_usd,
            ),
        )
        if proposal is None:
            return False
        recommended_side = OrderSide.BUY if proposal.buy_yes else OrderSide.SELL

        log.info(
            "platform_consensus.signal_detected",
            market_id=market.id,
            market_question=market.question[:50],
            consensus_source=proposal.consensus_source,
            consensus_prob=round(proposal.consensus_probability, 3),
            market_prob=round(market_p, 3),
            edge=round(proposal.edge_percent, 2),
            side=recommended_side.value,
        )

        signal = Signal(
            market_id=market.id,
            market_question=market.question,
            claude_prob=proposal.consensus_probability,
            claude_confidence=Confidence.HIGH
            if proposal.confidence == "high"
            else Confidence.MEDIUM,
            market_prob=market_p,
            edge=proposal.edge_percent,
            evidence_summary=proposal.evidence_summary,
            recommended_side=recommended_side,
            strategy_source=self.name,
        )

        # Insert DB rows for FK constraints
        await self._db.execute(
            """INSERT OR IGNORE INTO markets (id, exchange, condition_id, question, description,
               category, active, outcome_yes_price, outcome_no_price,
               volume, liquidity, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, datetime('now'))""",
            (
                market.id,
                market.exchange or "polymarket",
                market.condition_id,
                market.question,
                (market.description or "")[:500],
                ensure_category(market.question, market.description, market.category),
                market.outcome_yes_price,
                market.outcome_no_price,
                market.volume,
                market.liquidity,
            ),
        )
        await self._db.execute(
            """INSERT INTO signals (market_id, claude_prob, claude_confidence, market_prob,
                                     edge, evidence_summary, action, strategy_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal.market_id,
                signal.claude_prob,
                signal.claude_confidence.value,
                signal.market_prob,
                signal.edge,
                signal.evidence_summary,
                signal.recommended_side.value if signal.recommended_side else None,
                signal.strategy_source,
            ),
        )
        await self._db.commit()

        # Evaluate with risk manager
        decision = await self._risk.evaluate(
            signal, market, available_cash=None
        )
        if not decision.approved or decision.position_size <= 0:
            log.info(
                "platform_consensus.risk_rejected",
                market_id=market.id,
                reason=decision.reason[:80],
            )
            return False

        # Submit order
        if self._gateway is None:
            self._gateway = ExecutionGateway(
                router=None,
                exchange=self._exchange,
                exchange_name="polymarket",
                settings=self._settings,
                db=self._db,
                pnl_tracker=self._pnl,
            )

        size = min(decision.position_size, cfg.stake_usd)
        force_paper = cfg.paper or getattr(decision, "force_paper", False)
        res = await self._gateway.submit(
            TradeIntent(
                signal=signal,
                market=market,
                size_dollars=size,
                force_paper=force_paper,
            )
        )

        if res.status not in ("filled", "paper", "partial", "pending"):
            log.warning(
                "platform_consensus.order_rejected",
                market_id=market.id,
                status=res.status,
                reason=res.reason,
            )
            return False

        log.info(
            "platform_consensus.executed",
            market_id=market.id,
            status=res.status,
            size=size,
            is_paper=res.result.is_paper if res.result else None,
        )
        return True

    async def close(self) -> None:
        await self._manifold.close()
        await self._metaculus.close()
