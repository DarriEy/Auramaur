"""15 independent risk checks, each returning a CheckResult."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from auramaur.exchange.models import Confidence


class CheckResult(BaseModel):
    name: str
    passed: bool
    reason: str = ""
    value: Any = None


# ---------------------------------------------------------------------------
# 1. Kill switch
# ---------------------------------------------------------------------------

async def check_kill_switch() -> CheckResult:
    """Fail if KILL_SWITCH file exists on disk."""
    active = Path("KILL_SWITCH").exists()
    return CheckResult(
        name="kill_switch",
        passed=not active,
        reason="KILL_SWITCH file detected — all trading halted" if active else "",
    )


# ---------------------------------------------------------------------------
# 2. Max drawdown
# ---------------------------------------------------------------------------

async def check_max_drawdown(current_drawdown: float, max_pct: float = 15.0) -> CheckResult:
    """Fail if current drawdown exceeds the maximum allowed percentage."""
    exceeded = current_drawdown >= max_pct
    return CheckResult(
        name="max_drawdown",
        passed=not exceeded,
        reason=f"Drawdown {current_drawdown:.1f}% exceeds limit {max_pct:.1f}%" if exceeded else "",
        value=current_drawdown,
    )


# ---------------------------------------------------------------------------
# 3. Drawdown heat
# ---------------------------------------------------------------------------

_HEAT_THRESHOLDS: list[tuple[float, str]] = [
    (5.0, "GREEN"),
    (10.0, "YELLOW"),
    (13.0, "ORANGE"),
]


async def check_drawdown_heat(current_drawdown: float, max_pct: float = 15.0) -> CheckResult:
    """Return a heat level based on drawdown. Fails at RED (>=13% of max)."""
    heat = "RED"
    for threshold, level in _HEAT_THRESHOLDS:
        if current_drawdown < threshold:
            heat = level
            break

    failed = heat == "RED"
    return CheckResult(
        name="drawdown_heat",
        passed=not failed,
        reason=f"Drawdown heat is {heat} ({current_drawdown:.1f}%)" if failed else "",
        value=heat,
    )


# ---------------------------------------------------------------------------
# 4. Max stake
# ---------------------------------------------------------------------------

async def check_max_stake(proposed_stake: float, max_stake: float = 25.0) -> CheckResult:
    """Fail if proposed stake exceeds the per-market limit."""
    exceeded = proposed_stake > max_stake
    return CheckResult(
        name="max_stake",
        passed=not exceeded,
        reason=f"Stake ${proposed_stake:.2f} exceeds limit ${max_stake:.2f}" if exceeded else "",
        value=proposed_stake,
    )


# ---------------------------------------------------------------------------
# 5. Daily loss
# ---------------------------------------------------------------------------

async def check_daily_loss(daily_loss: float, limit: float = 200.0) -> CheckResult:
    """Fail if cumulative daily loss exceeds the limit."""
    exceeded = daily_loss >= limit
    return CheckResult(
        name="daily_loss",
        passed=not exceeded,
        reason=f"Daily loss ${daily_loss:.2f} exceeds limit ${limit:.2f}" if exceeded else "",
        value=daily_loss,
    )


# ---------------------------------------------------------------------------
# 6. Max positions
# ---------------------------------------------------------------------------

async def check_max_positions(open_count: int, max_positions: int = 15) -> CheckResult:
    """Fail if the number of open positions is at the limit."""
    at_limit = open_count >= max_positions
    return CheckResult(
        name="max_positions",
        passed=not at_limit,
        reason=f"{open_count} open positions (limit {max_positions})" if at_limit else "",
        value=open_count,
    )


# ---------------------------------------------------------------------------
# 7. Min edge
# ---------------------------------------------------------------------------

async def check_min_edge(edge: float, min_edge_pct: float = 5.0) -> CheckResult:
    """Fail if the estimated edge is below the minimum threshold."""
    too_small = abs(edge) < min_edge_pct
    return CheckResult(
        name="min_edge",
        passed=not too_small,
        reason=f"Edge {edge:.2f}% below minimum {min_edge_pct:.1f}%" if too_small else "",
        value=edge,
    )


# ---------------------------------------------------------------------------
# 8. Min liquidity
# ---------------------------------------------------------------------------

async def check_min_liquidity(liquidity: float, min_liquidity: float = 1000.0) -> CheckResult:
    """Fail if market liquidity is too thin."""
    too_thin = liquidity < min_liquidity
    return CheckResult(
        name="min_liquidity",
        passed=not too_thin,
        reason=f"Liquidity ${liquidity:.0f} below minimum ${min_liquidity:.0f}" if too_thin else "",
        value=liquidity,
    )


# ---------------------------------------------------------------------------
# 9. Max spread
# ---------------------------------------------------------------------------

async def check_max_spread(spread: float, max_spread_pct: float = 5.0) -> CheckResult:
    """Fail if the bid-ask spread is too wide."""
    too_wide = spread > max_spread_pct
    return CheckResult(
        name="max_spread",
        passed=not too_wide,
        reason=f"Spread {spread:.2f}% exceeds limit {max_spread_pct:.1f}%" if too_wide else "",
        value=spread,
    )


# ---------------------------------------------------------------------------
# 10. Confidence floor
# ---------------------------------------------------------------------------

_CONFIDENCE_ORDER = {
    Confidence.LOW: 0,
    Confidence.MEDIUM_LOW: 1,
    Confidence.MEDIUM: 2,
    Confidence.MEDIUM_HIGH: 3,
    Confidence.HIGH: 4,
}


async def check_confidence_floor(
    confidence: str | Confidence, floor: str = "MEDIUM"
) -> CheckResult:
    """Fail if the confidence level is below the floor."""
    conf_enum = Confidence(confidence) if isinstance(confidence, str) else confidence
    floor_enum = Confidence(floor)
    below = _CONFIDENCE_ORDER[conf_enum] < _CONFIDENCE_ORDER[floor_enum]
    return CheckResult(
        name="confidence_floor",
        passed=not below,
        reason=f"Confidence {conf_enum.value} below floor {floor}" if below else "",
        value=conf_enum.value,
    )


# ---------------------------------------------------------------------------
# 11. Implied probability bounds
# ---------------------------------------------------------------------------

async def check_implied_prob_bounds(
    market_prob: float, min_p: float = 0.05, max_p: float = 0.95
) -> CheckResult:
    """Fail if market probability is outside acceptable bounds."""
    outside = market_prob < min_p or market_prob > max_p
    return CheckResult(
        name="implied_prob_bounds",
        passed=not outside,
        reason=(
            f"Market prob {market_prob:.3f} outside bounds [{min_p}, {max_p}]"
            if outside
            else ""
        ),
        value=market_prob,
    )


# ---------------------------------------------------------------------------
# 12. Category exposure
# ---------------------------------------------------------------------------

async def check_category_exposure(
    category: str, category_exposure: float, cap_pct: float = 30.0
) -> CheckResult:
    """Fail if a single category is too concentrated in the portfolio."""
    too_concentrated = category_exposure >= cap_pct
    return CheckResult(
        name="category_exposure",
        passed=not too_concentrated,
        reason=(
            f"Category '{category}' at {category_exposure:.1f}% (cap {cap_pct:.1f}%)"
            if too_concentrated
            else ""
        ),
        value=category_exposure,
    )


# ---------------------------------------------------------------------------
# 13. Correlation
# ---------------------------------------------------------------------------

async def check_correlation(
    market_id: str, correlation_score: float, max_correlated: int = 5,
) -> CheckResult:
    """Fail if weighted correlation score exceeds *max_correlated*.

    The score is a weighted sum: semantic relationships count at full
    strength (0.5–1.0 each), while same-category positions without a
    semantic link count at 0.3 each.
    """
    too_many = correlation_score > max_correlated
    return CheckResult(
        name="correlation",
        passed=not too_many,
        reason=(
            f"Market {market_id} correlation score {correlation_score:.1f} (max {max_correlated})"
            if too_many
            else ""
        ),
        value=correlation_score,
    )


# ---------------------------------------------------------------------------
# 14. Time to resolution
# ---------------------------------------------------------------------------

async def check_time_to_resolution(
    hours_remaining: float, min_hours: float = 24
) -> CheckResult:
    """Fail if the market is resolving too soon."""
    too_soon = hours_remaining < min_hours
    return CheckResult(
        name="time_to_resolution",
        passed=not too_soon,
        reason=(
            f"{hours_remaining:.1f}h to resolution (minimum {min_hours:.0f}h)"
            if too_soon
            else ""
        ),
        value=hours_remaining,
    )


# ---------------------------------------------------------------------------
# 15. Second opinion divergence
# ---------------------------------------------------------------------------

async def check_second_opinion_divergence(
    divergence: float | None, max_divergence: float = 0.15
) -> CheckResult:
    """Fail if the two model opinions are too far apart."""
    if divergence is None:
        return CheckResult(name="second_opinion_divergence", passed=True, reason="No second opinion", value=None)
    too_far = abs(divergence) > max_divergence
    return CheckResult(
        name="second_opinion_divergence",
        passed=not too_far,
        reason=(
            f"Divergence {divergence:.3f} exceeds max {max_divergence:.3f}"
            if too_far
            else ""
        ),
        value=divergence,
    )
