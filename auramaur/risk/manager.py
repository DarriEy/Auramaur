"""Risk gate orchestrator — runs all 15 checks and sizes positions."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from pydantic import BaseModel

from auramaur.db.database import Database
from auramaur.exchange.models import Market, Signal
from auramaur.risk.checks import (
    CheckResult,
    check_category_exposure,
    check_confidence_floor,
    check_correlation,
    check_daily_loss,
    check_divergence_band,
    check_drawdown_heat,
    check_implied_prob_bounds,
    check_kill_switch,
    check_max_drawdown,
    check_max_positions,
    check_max_spread,
    check_max_stake,
    check_min_edge,
    check_min_liquidity,
    check_blocked_category,
    check_category_allowlist,
    check_dispute_risk,
    check_mispricing_named,
    check_second_opinion_divergence,
    check_time_to_resolution,
)
from auramaur.risk.kelly import KellySizer
from auramaur.risk.portfolio import PortfolioTracker
from auramaur.risk.regime import resolve_regime

log = structlog.get_logger()


class RiskDecision(BaseModel):
    approved: bool
    checks: list[CheckResult]
    position_size: float
    reason: str
    # Graduation ladder (Phase 3): entries from unproven/demoted
    # (strategy × category) cells run dry-run regardless of global live
    # mode. Restriction-only — exits never pass through evaluate().
    force_paper: bool = False
    graduation_status: str = ""


class RiskManager:
    """Orchestrates all risk checks and Kelly sizing for a proposed trade."""

    def __init__(self, settings, db: Database):
        self.settings = settings
        self.db = db
        self.portfolio = PortfolioTracker(db, settings=settings)
        self.kelly = KellySizer(fraction=settings.kelly.fraction)
        from auramaur.risk.graduation import GraduationLadder
        self.graduation = GraduationLadder(db, settings)
        # Optional post-hoc mispricing auditor (nlp/gap_audit.GapAuditor),
        # wired by the bot after the analyzer exists. None -> the gate
        # blocks unexplained divergences without spending an LLM call.
        self.gap_auditor = None
        # Set by the operational live-readiness preflight (monitoring/live_gate):
        # when a BLOCK condition is present, force every ENTRY to paper. Exits
        # bypass evaluate() entirely, so held positions can still get out.
        self.live_entries_blocked = False

    def _paper_forced_strategy(self, strategy_source: str) -> bool:
        """True if this strategy is paper-forced by its own config flag
        (e.g. bias_harvest/entailment_arb/resolution_lens carry ``paper: true``).

        ``is True`` is deliberate: it only trips on a genuine bool, so a
        MagicMock settings object in tests (whose attributes are truthy) does
        NOT read as paper-forced.
        """
        cfg = getattr(self.settings, strategy_source or "", None)
        return getattr(cfg, "paper", False) is True

    async def evaluate(
        self,
        signal: Signal,
        market: Market,
        price_history: dict[str, list[float]] | None = None,
        available_cash: float | None = None,
    ) -> RiskDecision:
        """Run every risk check and, if all pass, compute position size."""
        # Apply the global risk-tolerance lever (0=conservative..100=YOLO) — one
        # dial scales the whole prob/stat/risk surface at this gateway.
        from auramaur.risk.tolerance import scale_risk, current_tolerance
        rc, scaled_kelly = scale_risk(
            self.settings.risk, self.settings.kelly.fraction, current_tolerance(self.settings))

        # Gather portfolio state
        drawdown = await self.portfolio.get_drawdown()
        daily_pnl = await self.portfolio.get_daily_pnl()
        positions = await self.portfolio.get_positions()
        category_exposure = await self.portfolio.get_category_exposure()
        # correlated is computed below, AFTER is_paper_entry is known, so it can be
        # mode-scoped (a paper entry carries no real exposure — see that call site).

        # Cash = what we can actually deploy right now.
        # Equity = cash + position notional — drives regime switching so
        # capital-starved books get growth-mode params while mature books
        # get preservation-tuned config values.  Kelly bankroll uses cash
        # (not equity) because we can only bet money that isn't already
        # deployed.
        cash = available_cash if available_cash is not None else self.settings.execution.paper_initial_balance
        position_notional = sum(
            p.size * (p.current_price or p.avg_price)
            for p in positions
        )
        equity = cash + position_notional
        regime = resolve_regime(
            equity=equity,
            base_kelly=scaled_kelly,
            base_max_stake=rc.max_stake_per_market,
            base_min_edge_pct=rc.min_edge_pct,
        )

        # Interpret max_stake <= 1.0 as a percentage of equity (e.g. 0.02 = 2%)
        max_stake = regime.max_stake
        if max_stake <= 1.0:
            max_stake = equity * max_stake

        # Hard absolute ceiling — binds LAST, after the equity conversion and any
        # regime scaling. The 2%-of-equity cap grows as the book grows and the
        # regime can scale it up, so without this the documented per-market limit
        # silently drifts (a $40 news_speed entry at ~3.3% of equity got through
        # on 2026-06-15). Clamping here flows into both the Kelly cap below
        # (max_stake=min(max_stake, cash)) and the post-sizing max_stake check.
        max_stake = min(max_stake, rc.max_stake_abs_ceiling)

        # Kelly sizes against the full deployable cash; max_stake is the
        # ceiling that actually binds. The previous min(cash, max_stake*3)
        # double-capped: with fraction<=0.55 and edge capped at ±20% (so
        # kelly<=~0.8), kelly*fraction*(3*max_stake) tops out near 0.7*max_stake,
        # meaning the configured per-market cap was unreachable and effective
        # sizing sat ~30% below the documented limit. Cash is the real bankroll;
        # the post-sizing max_stake check still enforces the per-market cap.
        bankroll = cash

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
        # Will this entry be paper-traded? — global paper mode, a per-strategy
        # paper-forced pillar (bias_harvest/entailment_arb/resolution_lens), or
        # a graduation-demoted/unproven cell. Paper exploration must BYPASS the
        # live-only gates (category allowlist + divergence-adverse filter) so
        # paper-forced strategies build a complete pnl_ledger record across all
        # categories for the graduation ladder to evaluate; genuine live entries
        # keep every gate. The blocklist still applies to paper — decided-no-edge
        # categories never even paper-trade. `cell` is cached, so computing it
        # here and reusing it below is free.
        cell = await self.graduation.decide(
            signal.strategy_source, market.category or "")
        is_paper_entry = (
            not self.settings.is_live
            or self.live_entries_blocked  # operational preflight BLOCK
            or self._paper_forced_strategy(signal.strategy_source)
            or cell.force_paper
        )

        # Correlation, MODE-SCOPED. A paper entry adds NO real exposure, so it must
        # only correlate against the PAPER book — counting the live book would let
        # live concentration crowd out paper exploration (e.g. choking long_horizon
        # in categories with a big live book). A live entry correlates against the
        # live book. This matches the mode-scoping the per-market stake cap already
        # uses (_exceeds_market_cap), and only ever loosens paper — live behavior is
        # unchanged (live concentration still measured against live positions).
        correlated = await self.portfolio.get_correlated_markets(
            signal.market_id, is_paper=is_paper_entry)

        # ----------------------------------------------------------------
        # Run pre-sizing checks (max_stake validated after sizing)
        # ----------------------------------------------------------------
        pre_checks: list[CheckResult] = [
            await check_kill_switch(),
            await check_max_drawdown(drawdown, rc.max_drawdown_pct),
            await check_drawdown_heat(drawdown, rc.max_drawdown_pct),
            await check_daily_loss(max(0.0, -daily_pnl), rc.daily_loss_limit),
            await check_max_positions(len(positions), rc.max_open_positions),
            await check_min_edge(signal.edge, regime.min_edge_pct),
            await check_divergence_band(
                signal.claude_prob, signal.market_prob, signal.claude_confidence,
                rc.divergence_filter_enabled and not is_paper_entry,
                rc.divergence_adverse_low,
                rc.divergence_adverse_high, rc.divergence_require_confidence),
            await check_min_liquidity(
                max(market.liquidity, market.volume),
                rc.kalshi_min_liquidity if (market.exchange or "").lower() == "kalshi" else rc.min_liquidity
            ),
            await check_max_spread(market.spread, rc.max_spread_pct),
            await check_confidence_floor(signal.claude_confidence, rc.confidence_floor),
            await check_implied_prob_bounds(
                signal.market_prob, rc.implied_prob_min, rc.implied_prob_max
            ),
            await check_category_exposure(market.category, cat_exp, rc.category_exposure_cap_pct),
            await check_correlation(signal.market_id, correlated, rc.max_correlated_positions),
            await check_time_to_resolution(hours_remaining, rc.time_to_resolution_min_hours, rc.time_to_resolution_max_days * 24.0),
            await check_second_opinion_divergence(divergence, rc.second_opinion_divergence_max),
        ]

        # Blocked categories at the single gateway: the engine-level filter
        # only covers run_cycle's candidate selection — news_speed (via
        # analyze_market) and other entry paths sailed past it (caught live
        # 2026-06-10 buying politics_us). Block on the stored category OR a
        # fresh classification: 247 active sports markets carried stale
        # 'other' labels that dodged the block, so trusting the stored label
        # alone was the residual leak. Structural two-sided strategies
        # (graduation's exempt list) stay free to quote/arb.
        from auramaur.strategy.classifier import classify_market
        fresh_category = classify_market(
            market.question or "", market.description or "")
        category_applies = signal.strategy_source not in set(
            self.settings.graduation.exempt_strategies)
        pre_checks.append(await check_blocked_category(
            market.category or "", rc.blocked_categories,
            applies=category_applies,
            fallback_category=fresh_category,
        ))
        # Live entries additionally require the allowlist (fail-safe): the
        # stored (venue-tag-derived) label must name a category we have
        # demonstrated edge in; unknown/'other' markets stay paper-only, so
        # a classifier gap costs opportunity, not money. The fresh keyword
        # classification stays the #17 tripwire for confidently-bad labels.
        if not is_paper_entry:
            # Per-strategy extensions widen the allowlist ONLY for the named
            # strategy_source (a proven ladder cell earning its category, e.g.
            # bias_harvest x other). Putting the extension category on the
            # GLOBAL list instead would also open it to every direct consumer
            # of allowed_categories_live — the graduation-exempt market maker
            # and arb executor — re-creating the fail-open hole the 2026-06
            # mislabel incident closed.
            allowed = list(rc.allowed_categories_live) + list(
                (rc.allowed_categories_live_extra or {}).get(
                    signal.strategy_source, []))
            pre_checks.append(await check_category_allowlist(
                market.category or "", allowed,
                applies=category_applies,
                fallback_category=fresh_category,
            ))
            # Dispute gate (all live entries, incl. structural strategies — a
            # contested resolution is adverse to arb/MM too): don't enter a
            # market whose UMA resolution is actively disputed.
            pre_checks.append(await check_dispute_risk(market.dispute_risk))

        # ----------------------------------------------------------------
        # Name-the-gap gate: a significant LLM divergence must name the
        # mechanism why the market is wrong, or it doesn't trade. The
        # estimation pipeline is price-blind (anti-anchoring), so the audit
        # is a separate post-hoc LLM call — made lazily, only for signals
        # that already pass everything else (no spend on doomed trades).
        # Strategies that pre-name their mechanism (resolution_lens,
        # bias_harvest's measured bias, entailment bounds) skip the call.
        # ----------------------------------------------------------------
        gap_div = abs(signal.claude_prob - signal.market_prob)
        gate_applies = (rc.mispricing_gate_enabled
                        and signal.strategy_source in ("llm",)
                        and gap_div >= rc.mispricing_min_divergence)
        if (gate_applies and not signal.mispricing_reason
                and self.gap_auditor is not None
                and all(c.passed for c in pre_checks)):
            try:
                signal.mispricing_reason = await self.gap_auditor.audit(signal, market)
            except Exception as e:
                log.warning("risk.gap_audit_failed", market_id=signal.market_id,
                            error=str(e))
        pre_checks.append(await check_mispricing_named(
            signal.mispricing_reason, gap_div,
            enabled=rc.mispricing_gate_enabled,
            min_divergence=rc.mispricing_min_divergence,
            applies=signal.strategy_source in ("llm",),
        ))

        pre_passed = all(c.passed for c in pre_checks)

        # ----------------------------------------------------------------
        # Position sizing (only when pre-checks pass)
        # ----------------------------------------------------------------
        position_size = 0.0
        if pre_passed:
            heat_check = next(c for c in pre_checks if c.name == "drawdown_heat")
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
                bankroll=bankroll,
                heat_mult=KellySizer.heat_multiplier(heat),
                confidence_mult=KellySizer.confidence_multiplier(signal.claude_confidence),
                liquidity_mult=KellySizer.liquidity_multiplier(max(market.liquidity, market.volume)),
                category_mult=category_mult,
                volatility_mult=vol_mult,
                max_stake=min(max_stake, cash),
                fraction_override=regime.kelly_fraction,
            )

        # Run max_stake check on the actual computed position size
        stake_check = await check_max_stake(position_size, max_stake)
        checks = pre_checks + [stake_check]

        all_passed = pre_passed and stake_check.passed
        failed = [c for c in checks if not c.passed]

        reason = (
            "All checks passed"
            if all_passed
            else "; ".join(c.reason for c in failed)
        )

        # Graduation ladder: the cell's measured record (decided above) sets
        # whether this ENTRY trades live, on probation size, or paper-forced.
        # Applied after all checks so it can only restrict an already-approved
        # trade.
        if all_passed and cell.size_multiplier != 1.0:
            position_size = position_size * cell.size_multiplier
            # A zero multiplier (e.g. the unproven-spray cap) is a HARD SKIP:
            # reject the entry outright. Leaving approved=True with size 0 would
            # be unsafe — prepare_order bumps sub-minimum sizes back up to the
            # order floor, which would defeat the cap.
            if position_size <= 0:
                all_passed = False
                reason = cell.reason

        decision = RiskDecision(
            approved=all_passed,
            checks=checks,
            position_size=position_size,
            reason=reason,
            force_paper=cell.force_paper,
            graduation_status=cell.status,
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
            equity=round(equity, 2),
            regime=regime.name,
            kelly_fraction=round(regime.kelly_fraction, 3),
            max_stake=round(regime.max_stake, 2),
            min_edge_pct=round(regime.min_edge_pct, 2),
        )

        return decision
