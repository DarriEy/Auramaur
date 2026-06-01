"""Global risk-tolerance lever.

One dial (settings.risk_tolerance, 0..100) scales the whole prob/stat/risk
surface coherently, so you don't hand-tune 15 parameters:

  0    = most conservative — small Kelly/stake, high edge bar, tight prob bounds,
                             HIGH confidence required, low drawdown/exposure
  50   = neutral           — config values unchanged
  100  = YOLO              — bigger Kelly/stake, low edge bar, wide prob bounds,
                             LOW confidence accepted, high drawdown/exposure

Applied at the RiskManager gateway, so every trade respects it.
"""

from __future__ import annotations

_CONF = ["LOW", "MEDIUM", "HIGH"]


def _aggr(t: float) -> float:
    """Multiplier for params that GROW with aggression. 0->0.4, .5->1.0, 1->1.6."""
    return 0.4 + 1.2 * max(0.0, min(1.0, t))


def _cons(t: float) -> float:
    """Multiplier for params that SHRINK with aggression. 0->1.6, .5->1.0, 1->0.4."""
    return 1.6 - 1.2 * max(0.0, min(1.0, t))


def _confidence_floor(base: str, t: float) -> str:
    i = _CONF.index(base) if base in _CONF else 1
    if t >= 0.66:        # bold: accept lower confidence
        i = max(0, i - 1)
    elif t <= 0.34:      # timid: demand higher confidence
        i = min(2, i + 1)
    return _CONF[i]


def scale_risk(risk, kelly_fraction: float, tolerance: float):
    """Return (tolerance-scaled RiskConfig, scaled kelly fraction).

    `tolerance` is 0..100 (0=conservative, 50=neutral no-op, 100=YOLO).
    Bounds are clamped so extremes stay sane.
    """
    t = max(0.0, min(100.0, tolerance)) / 100.0
    a, c = _aggr(t), _cons(t)
    scaled = risk.model_copy(update={
        # --- risk appetite (grow with aggression) ---
        "max_stake_per_market": risk.max_stake_per_market * a,
        "max_drawdown_pct": min(risk.max_drawdown_pct * a, 90.0),
        "daily_loss_limit": risk.daily_loss_limit * a,
        "max_open_positions": max(1, round(risk.max_open_positions * a)),
        "category_exposure_cap_pct": min(risk.category_exposure_cap_pct * a, 100.0),
        "max_correlated_positions": max(1, round(risk.max_correlated_positions * a)),
        "max_spread_pct": risk.max_spread_pct * a,
        # --- stat thresholds (looser with aggression) ---
        "min_edge_pct": risk.min_edge_pct * c,
        "second_opinion_divergence_max": min(risk.second_opinion_divergence_max * a, 1.0),
        # --- probability handling ---
        "implied_prob_min": max(0.001, risk.implied_prob_min * c),
        "implied_prob_max": min(0.999, 1.0 - (1.0 - risk.implied_prob_max) * c),
        "confidence_floor": _confidence_floor(risk.confidence_floor, t),
    })
    return scaled, kelly_fraction * a
