"""Pre-registered evidence contract for graduating an IBKR paper strategy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import fmean, pstdev


@dataclass(frozen=True, slots=True)
class IBKREvidence:
    observations: int
    elapsed_days: int
    net_pnl_usd: float
    mean_pnl_usd: float
    mean_lower_95_usd: float
    max_drawdown_usd: float
    brier_score: float | None
    baseline_brier_score: float | None
    ready: bool
    reasons: tuple[str, ...]


def _max_drawdown(pnls: list[float]) -> float:
    equity = peak = drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return drawdown


def evaluate_ibkr_evidence(
    net_trade_pnls: list[float], *, elapsed_days: int,
    budget_usd: float, probabilities: list[float] | None = None,
    outcomes: list[int] | None = None, min_observations: int = 200,
    min_elapsed_days: int = 180, max_drawdown_pct: float = 10.0,
) -> IBKREvidence:
    """Evaluate a deliberately demanding, immutable graduation contract.

    P&Ls must already include spreads, slippage, commissions, financing, rolls,
    and intelligence cost. The normal lower bound is intentionally conservative;
    callers should additionally retain a walk-forward/out-of-sample split.
    """
    n = len(net_trade_pnls)
    mean = fmean(net_trade_pnls) if n else 0.0
    se = pstdev(net_trade_pnls) / math.sqrt(n) if n > 1 else math.inf
    lower = mean - 1.96 * se
    drawdown = _max_drawdown(net_trade_pnls)
    brier = baseline = None
    if probabilities is not None and outcomes is not None and probabilities:
        pairs = list(zip(probabilities, outcomes))
        brier = fmean((p - y) ** 2 for p, y in pairs)
        base_p = fmean(y for _, y in pairs)
        baseline = fmean((base_p - y) ** 2 for _, y in pairs)
    reasons = []
    if n < min_observations:
        reasons.append(f"{n}/{min_observations} cost-adjusted observations")
    if elapsed_days < min_elapsed_days:
        reasons.append(f"{elapsed_days}/{min_elapsed_days} elapsed days")
    if lower <= 0:
        reasons.append("95% lower confidence bound on mean P&L is not positive")
    if drawdown > budget_usd * max_drawdown_pct / 100:
        reasons.append("maximum drawdown exceeds the pre-registered budget limit")
    if brier is not None and baseline is not None and brier >= baseline:
        reasons.append("forecast Brier score does not beat the base-rate forecast")
    return IBKREvidence(
        n, elapsed_days, sum(net_trade_pnls), mean, lower, drawdown,
        brier, baseline, not reasons, tuple(reasons),
    )


def evaluate_ibkr_daily_evidence(
    daily_pnls_usd: list[float], *, elapsed_days: int, budget_usd: float,
    min_observations: int = 120, min_elapsed_days: int = 180,
    max_drawdown_pct: float = 10.0,
) -> IBKREvidence:
    """Pre-registered contract over DAILY marked-to-market P&L.

    Adopted 2026-07-20 (docs/ibkr_multiasset_paper.md, "Evidence contract"):
    the round-trip contract is arithmetically unreachable for slow-turnover
    books — an FX book holding 2-6 weeks across 4 slots produces ~15-50
    round trips in 180 days against a 200-trip bar. Daily marks give ~130
    observations over the same window with identical cost realism (marks
    are net of the simulated fills'' spreads/slippage/commissions), so the
    statistical bar stays high while the clock actually runs. Round trips
    remain a SECONDARY realism check (see the preflight), and holding
    brackets must never be shortened merely to manufacture observations.
    """
    n = len(daily_pnls_usd)
    mean = fmean(daily_pnls_usd) if n else 0.0
    se = pstdev(daily_pnls_usd) / math.sqrt(n) if n > 1 else math.inf
    lower = mean - 1.96 * se
    drawdown = _max_drawdown(daily_pnls_usd)
    reasons = []
    if n < min_observations:
        reasons.append(f"{n}/{min_observations} daily observations")
    if elapsed_days < min_elapsed_days:
        reasons.append(f"{elapsed_days}/{min_elapsed_days} elapsed days")
    if n and lower <= 0:
        reasons.append(
            f"daily mean lower bound ${lower:.2f} not positive")
    if budget_usd > 0 and drawdown > budget_usd * max_drawdown_pct / 100:
        reasons.append(
            f"max drawdown ${drawdown:.2f} exceeds {max_drawdown_pct:.0f}% of budget")
    return IBKREvidence(
        observations=n, elapsed_days=elapsed_days,
        net_pnl_usd=sum(daily_pnls_usd), mean_pnl_usd=mean,
        mean_lower_95_usd=lower if n > 1 else 0.0,
        max_drawdown_usd=drawdown, brier_score=None,
        baseline_brier_score=None, ready=not reasons,
        reasons=tuple(reasons),
    )
