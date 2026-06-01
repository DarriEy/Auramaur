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

from pathlib import Path

_CONF = ["LOW", "MEDIUM", "HIGH"]

# Runtime override so `auramaur risk <n>` takes effect LIVE (no restart): the CLI
# writes here, every consumer reads it each time, falling back to config.
_OVERRIDE = Path("data/risk_tolerance")


def current_tolerance(settings) -> float:
    """Effective risk tolerance now: the live override file if present, else config."""
    try:
        return max(0.0, min(100.0, float(_OVERRIDE.read_text().strip())))
    except (OSError, ValueError):
        return float(getattr(settings, "risk_tolerance", 50.0))


def set_tolerance(value: float) -> None:
    """Persist a live override (clamped 0..100)."""
    _OVERRIDE.parent.mkdir(parents=True, exist_ok=True)
    _OVERRIDE.write_text(str(max(0.0, min(100.0, float(value)))))


def scale_budget(base_usd: float, tolerance: float) -> float:
    """Scale a directional exposure budget by the lever (0..100)."""
    return base_usd * _aggr(max(0.0, min(100.0, tolerance)) / 100.0)


# Multiplier at YOLO (t=1); 1/_SPAN at most-conservative (t=0). Geometric so the
# lever is symmetric in log-space and far more sensitive near the extremes than a
# linear ramp (e.g. 65 -> 1.39x vs the old 1.18x; 100 -> 3x vs 1.6x).
_SPAN = 3.0


def _aggr(t: float) -> float:
    """Multiplier for params that GROW with aggression. 0->0.33, .5->1.0, 1->3.0."""
    return _SPAN ** (2.0 * max(0.0, min(1.0, t)) - 1.0)


def _cons(t: float) -> float:
    """Multiplier for params that SHRINK with aggression (inverse of _aggr)."""
    return 1.0 / _aggr(t)


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
        # NOTE: the HARD ruin limits (max_drawdown_pct, daily_loss_limit) are
        # deliberately NOT scaled — those are circuit-breakers, not appetite.
        # Loosening them with aggression would remove the brake exactly when
        # sizing is biggest.
        "max_stake_per_market": risk.max_stake_per_market * a,
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
