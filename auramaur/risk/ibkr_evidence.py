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
