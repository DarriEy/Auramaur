"""Regime-switched risk parameters: grow aggressively at small equity,
preserve at large equity.

At $450 of equity the geometric advantage of conservative Kelly (0.30) is
irrelevant — the book is capital-starved and the dominant failure mode is
failing to compound, not volatility. As equity grows toward a working-book
size, we smoothly transition into the preservation-tuned config values.

The three dimensions we switch on:
  * kelly_fraction   — size multiplier on edge
  * max_stake        — absolute cap per market (flat $ in preservation,
                       percent-of-equity in growth so stakes scale with book)
  * min_edge_pct     — minimum edge required before a trade is considered

Regime boundaries:
  equity < GROWTH_MAX           -> pure growth
  GROWTH_MAX <= equity < PRES_MIN -> linear transition
  equity >= PRES_MIN            -> pure preservation (config values)
"""

from __future__ import annotations

from dataclasses import dataclass

GROWTH_EQUITY_MAX = 1000.0
PRESERVATION_EQUITY_MIN = 5000.0

GROWTH_KELLY_FRACTION = 0.50
GROWTH_MAX_STAKE_PCT = 10.0
GROWTH_MIN_EDGE_PCT = 2.5


@dataclass(frozen=True)
class RegimeParams:
    name: str
    kelly_fraction: float
    max_stake: float
    min_edge_pct: float


def resolve_regime(
    equity: float,
    base_kelly: float,
    base_max_stake: float,
    base_min_edge_pct: float,
) -> RegimeParams:
    """Pick risk params appropriate for the current book size.

    base_* are the preservation-mode values from config; they kick in at or
    above PRESERVATION_EQUITY_MIN. Below GROWTH_EQUITY_MAX we use the
    growth-mode constants. Between the two, parameters are linearly
    interpolated so there is no discontinuity as the book crosses a boundary.

    A non-positive equity falls back to base (preservation) values — better
    to be conservative when we can't see the book.
    """
    if equity <= 0:
        return RegimeParams("unknown", base_kelly, base_max_stake, base_min_edge_pct)

    growth_stake = equity * GROWTH_MAX_STAKE_PCT / 100.0

    if equity < GROWTH_EQUITY_MAX:
        return RegimeParams(
            name="growth",
            kelly_fraction=GROWTH_KELLY_FRACTION,
            max_stake=growth_stake,
            min_edge_pct=GROWTH_MIN_EDGE_PCT,
        )

    if equity >= PRESERVATION_EQUITY_MIN:
        return RegimeParams(
            name="preservation",
            kelly_fraction=base_kelly,
            max_stake=base_max_stake,
            min_edge_pct=base_min_edge_pct,
        )

    t = (equity - GROWTH_EQUITY_MAX) / (PRESERVATION_EQUITY_MIN - GROWTH_EQUITY_MAX)
    return RegimeParams(
        name="transition",
        kelly_fraction=GROWTH_KELLY_FRACTION + (base_kelly - GROWTH_KELLY_FRACTION) * t,
        max_stake=growth_stake + (base_max_stake - growth_stake) * t,
        min_edge_pct=GROWTH_MIN_EDGE_PCT + (base_min_edge_pct - GROWTH_MIN_EDGE_PCT) * t,
    )
