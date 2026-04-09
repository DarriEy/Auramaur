"""Risk gate orchestrator — runs all 15 checks and sizes positions."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import structlog
from pydantic import BaseModel

from auramaur.db.database import Database
from auramaur.exchange.models import Confidence, Market, Signal
from auramaur.risk.checks import (
    CheckResult,
    check_category_exposure,
    check_confidence_floor,
    check_correlation,
    check_daily_loss,
    check_drawdown_heat,
    check_implied_prob_bounds,
    check_kill_switch,
    check_max_drawdown,
    check_max_positions,
    check_max_spread,
    check_max_stake,
    check_min_edge,
    check_min_liquidity,
    check_second_opinion_divergence,
    check_time_to_resolution,
)
from auramaur.risk.kelly import KellySizer
from auramaur.risk.portfolio import PortfolioTracker

log = structlog.get_logger()


class RiskDecision(BaseModel):
    approved: bool
    checks: list[CheckResult]
    position_size: float
    reason: str


class RiskManager:
    """Orchestrates all risk checks and Kelly sizing for a proposed trade."""

    def __init__(self, settings, db: Database):
        self.settings = settings
        self.db = db
        self.portfolio = PortfolioTracker(db)
        self.kelly = KellySizer(fraction=settings.kelly.fraction)

    async def evaluate(
        self,
        signal: Signal,
        market: Market,
        price_history: dict[str, list[float]] | None = None,
        available_cash: float | None = None,
    ) -> RiskDecision:
        """Run every risk check and, if all pass, compute position size."""
        rc = self.settings.risk  # RiskConfig shortcut

        # Gather portfolio state
        drawdown = await self.portfolio.get_drawdown()
        daily_pnl = await self.portfolio.get_daily_pnl()
        positions = await self.portfolio.get_positions()
        category_exposure = await self.portfolio.get_category_exposure()
        correlated = await self.portfolio.get_correlated_markets(signal.market_id)

        # Time to resolution
        if market.end_date:
            end = market.end_date if market.end_date.tzinfo else market.end_date.replace(tzinfo=timezone.utc)
            hours_remaining = max(
                (end - datetime.now(timezone.utc)).total_seconds() / 3600.0, 0.0
            )
        else:
            hours_remaining = float("inf")

        # Divergence (use 0 if no second opinion available)
        divergence = signal.divergence if signal.divergence is not None else 0.0

        # Category exposure for this market's category
        cat_exp = category_exposure.get(market.category, 0.0)

        # ----------------------------------------------------------------
        # Run all 15 checks
        # ----------------------------------------------------------------
        checks: list[CheckResult] = [
            await check_kill_switch(),
            await check_max_drawdown(drawdown, rc.max_drawdown_pct),
            await check_drawdown_heat(drawdown, rc.max_drawdown_pct),
            await check_max_stake(signal.recommended_size, rc.max_stake_per_market),
            await check_daily_loss(abs(daily_pnl), rc.daily_loss_limit),
            await check_max_positions(len(positions), rc.max_open_positions),
            await check_min_edge(signal.edge, rc.min_edge_pct),
            await check_min_liquidity(max(market.liquidity, market.volume), rc.min_liquidity),
            await check_max_spread(market.spread, rc.max_spread_pct),
            await check_confidence_floor(signal.claude_confidence, rc.confidence_floor),
            await check_implied_prob_bounds(
                signal.market_prob, rc.implied_prob_min, rc.implied_prob_max
            ),
            await check_category_exposure(market.category, cat_exp, rc.category_exposure_cap_pct),
            await check_correlation(signal.market_id, correlated, rc.max_correlated_positions),
            await check_time_to_resolution(hours_remaining, rc.time_to_resolution_min_hours),
            await check_second_opinion_divergence(divergence, rc.second_opinion_divergence_max),
        ]

        all_passed = all(c.passed for c in checks)
        failed = [c for c in checks if not c.passed]

        # ----------------------------------------------------------------
        # Position sizing (only when approved)
        # ----------------------------------------------------------------
        position_size = 0.0
        if all_passed:
            heat_check = next(c for c in checks if c.name == "drawdown_heat")
            heat = heat_check.value  # GREEN / YELLOW / ORANGE

            # Get category multiplier from attribution
            category_mult = 1.0
            try:
                row = await self.db.fetchone(
                    "SELECT kelly_multiplier FROM category_stats WHERE category = ?",
                    (market.category,),
                )
                if row and row["kelly_multiplier"] is not None:
                    category_mult = float(row["kelly_multiplier"])
                    log.debug(
                        "risk.category_mult",
                        category=market.category,
                        multiplier=round(category_mult, 3),
                    )
            except Exception as e:
                log.warning(
                    "risk.category_mult_fallback",
                    category=market.category,
                    error=str(e),
                )

            # Volatility adjustment from price history
            vol_mult = 1.0
            if price_history and signal.market_id in price_history:
                vol_mult = KellySizer.volatility_multiplier(
                    price_history[signal.market_id]
                )

            position_size = self.kelly.calculate(
                claude_prob=signal.claude_prob,
                market_prob=signal.market_prob,
                bankroll=available_cash or self.settings.execution.paper_initial_balance,
                heat_mult=KellySizer.heat_multiplier(heat),
                confidence_mult=KellySizer.confidence_multiplier(signal.claude_confidence),
                liquidity_mult=KellySizer.liquidity_multiplier(max(market.liquidity, market.volume)),
                category_mult=category_mult,
                volatility_mult=vol_mult,
                max_stake=rc.max_stake_per_market,
            )

        reason = (
            "All checks passed"
            if all_passed
            else "; ".join(c.reason for c in failed)
        )

        decision = RiskDecision(
            approved=all_passed,
            checks=checks,
            position_size=position_size,
            reason=reason,
        )

        # ----------------------------------------------------------------
        # Log every decision
        # ----------------------------------------------------------------
        log.debug(
            "risk.decision",
            market_id=signal.market_id,
            approved=decision.approved,
            position_size=decision.position_size,
            checks_passed=sum(1 for c in checks if c.passed),
            checks_failed=len(failed),
            reason=decision.reason,
        )

        return decision
