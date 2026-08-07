"""Read the IBKR ETF arms' forecast calibration from resolved outcomes.

Why this exists. The ETF arms' entry gate is
``AND(confidence >= etf_min_confidence, probability >= etf_min_prob)``, and its
defaults (MEDIUM / 0.62) were tuned for MomentumETFAnalyzer, which returns
0.70/"HIGH" by construction. The OpenAI arms are asked for a *calibrated*
probability and told weak evidence "should remain near 0.50 with LOW
confidence", so they produced 0.43-0.56 topping out at MEDIUM_LOW and the gate
rejected 100% of them. Per-arm overrides now exist (``etf_arm_min_prob`` /
``etf_arm_min_confidence``) but are deliberately empty: picking a number before
any forecast had resolved would have been fitting to unscored output.

This module is what makes that number choosable from evidence instead. It is
pure — rows in, statistics out — so the arithmetic is testable without a DB.

The central honesty device is the Wilson lower bound. A 5-session directional
forecast is a coin flip until proven otherwise, so the question is never "is the
observed hit rate above 50%" (it will wander above 50% by luck) but "is it
*reliably* above 50% at this sample size". ``samples_for_reliable_edge``
answers the operator's real question — how much longer to wait — instead of
inviting a threshold to be read off twenty noisy resolutions.

KNOWN LIMITATION, and it cuts one way only. Every interval here assumes
independent observations. The ETF arms' forecasts are not: ~28 instruments that
co-move strongly (SPY/VTI/QQQ/XLK are near-duplicates of one factor) are
forecast on overlapping 5-session windows, so a day's 28 rows are far fewer
than 28 independent facts. The effective sample is therefore well below the row
count and the true interval is WIDER than reported. That makes every bound here
optimistic, which is the safe direction for a gate-opening decision: a
threshold this report refuses to support is genuinely unsupported, while one it
does support still deserves scrutiny. Clustering the standard error by session
date (or by asset class) is the honest upgrade if these numbers ever get close.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

# 95% two-sided. Matches the spirit of graduation.confidence_z rather than the
# exact value: this is a reporting aid, not a gate, and it must not be mistaken
# for one.
DEFAULT_Z = 1.96

# A directional forecast beats nothing until it beats a coin.
COIN = 0.5

# Ranks mirror IBKRETFPaperPillar._CONF exactly. Duplicated rather than imported
# so a reporting tool cannot drag a strategy module (and its exchange imports)
# into a CLI process.
CONFIDENCE_RANKS = {"LOW": 0, "MEDIUM_LOW": 1, "MEDIUM": 2,
                    "MEDIUM_HIGH": 3, "HIGH": 4}


@dataclass(frozen=True)
class Forecast:
    """One resolved (or unresolved) directional forecast.

    ``reference`` is the benchmark the forecast is scored against — the
    instrument's own trailing up-rate, not 0.5. See horizon_up_rate.
    """
    probability: float
    confidence: str
    actual_outcome: int | None
    # No default. A coin reference is not a conservative stand-in for the
    # instrument's drift — it is EASIER to beat, by exactly (0.5 - q)^2, so
    # defaulting to it handed every forecast free measured edge. At the ~0.56
    # five-session up-rate that is +0.0036 per forecast, which a flat
    # forecaster with zero information rides through the gate on patience
    # alone at ~1,050 resolutions (its 95% lower bound is still -0.00224 at
    # 400, so the defect is real but slower than the first telling claimed —
    # see test_a_coin_benchmark_hands_out_free_edge_of_exactly_the_drift_gap).
    # A forecast with no recorded benchmark cannot be scored; clearance()
    # treats it as unscoreable rather than scoring it against a coin.
    reference: float | None = None

    @property
    def resolved(self) -> bool:
        return self.actual_outcome is not None


@dataclass(frozen=True)
class Band:
    """A group of resolved forecasts and how often they were right."""
    label: str
    n: int
    hits: int
    mean_forecast: float
    realized: float
    lo: float
    hi: float

    @property
    def beats_coin(self) -> bool:
        """True only when the LOWER bound clears a coin flip."""
        return self.lo > COIN

    @property
    def calibration_gap(self) -> float:
        """Realized minus forecast. Positive = the arm is underconfident."""
        return self.realized - self.mean_forecast


@dataclass(frozen=True)
class ThresholdRow:
    """What a candidate ``etf_arm_min_prob`` would have selected."""
    threshold: float
    n: int
    hits: int
    realized: float
    lo: float
    samples_needed: int | None


def wilson_interval(hits: int, n: int, z: float = DEFAULT_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because n here is small (an arm
    produces ~28 forecasts a day and they resolve five sessions later). The
    normal interval famously misbehaves near 0 and 1 and at small n — it can
    even extend past [0, 1] — which is exactly the regime this report runs in.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = hits / n
    denominator = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - spread) / denominator),
            min(1.0, (centre + spread) / denominator))


def brier_score(forecasts) -> float | None:
    """Mean squared error of the resolved forecasts. 0.25 is a coin flip."""
    resolved = [f for f in forecasts if f.resolved]
    if not resolved:
        return None
    return sum((f.probability - f.actual_outcome) ** 2
               for f in resolved) / len(resolved)


def _band(label: str, group, z: float) -> Band | None:
    if not group:
        return None
    n = len(group)
    hits = sum(int(f.actual_outcome) for f in group)
    lo, hi = wilson_interval(hits, n, z)
    return Band(label=label, n=n, hits=hits,
                mean_forecast=sum(f.probability for f in group) / n,
                realized=hits / n, lo=lo, hi=hi)


def probability_bands(forecasts, *, width: float = 0.02,
                      z: float = DEFAULT_Z) -> list[Band]:
    """Calibration curve over the range the arms ACTUALLY produce.

    Fixed decile bins are useless here: a 0.43-0.56 spread collapses into two
    buckets and hides everything. Bins are therefore anchored to the observed
    range at ``width`` granularity.
    """
    resolved = [f for f in forecasts if f.resolved]
    if not resolved:
        return []
    lowest = min(f.probability for f in resolved)
    buckets: dict[int, list] = {}
    for forecast in resolved:
        index = int((forecast.probability - lowest) / width)
        buckets.setdefault(index, []).append(forecast)
    bands = []
    for index in sorted(buckets):
        start = lowest + index * width
        band = _band(f"{start:.2f}-{start + width:.2f}", buckets[index], z)
        if band is not None:
            bands.append(band)
    return bands


def confidence_bands(forecasts, z: float = DEFAULT_Z) -> list[Band]:
    """Does a higher stated confidence actually predict a higher hit rate?

    If the bands do not separate, ``etf_arm_min_confidence`` is not carrying
    information for these arms and should not be the binding gate — which
    matters, because confidence is checked BEFORE probability in run_once.
    """
    resolved = [f for f in forecasts if f.resolved]
    buckets: dict[str, list] = {}
    for forecast in resolved:
        buckets.setdefault(forecast.confidence.upper(), []).append(forecast)
    bands = []
    for label in sorted(buckets, key=lambda c: CONFIDENCE_RANKS.get(c, -1)):
        band = _band(label, buckets[label], z)
        if band is not None:
            bands.append(band)
    return bands


def samples_for_reliable_edge(rate: float, *, z: float = DEFAULT_Z,
                              target: float = COIN,
                              cap: int = 100_000) -> int | None:
    """Resolutions needed before a hit rate of ``rate`` clears ``target``.

    The operator's actual question on a thin sample is not "what is the number"
    but "how much longer until I can trust one". Returns None when the rate is
    at or below target (no sample size rescues it) or when the answer exceeds
    ``cap``, which for a book producing ~28 forecasts a day means "not this
    quarter".
    """
    if rate <= target:
        return None
    n = 2
    while n <= cap:
        if wilson_interval(round(rate * n), n, z)[0] > target:
            return n
        n += 1
    return None


def threshold_sweep(forecasts, thresholds, *, min_confidence: str = "LOW",
                    z: float = DEFAULT_Z) -> list[ThresholdRow]:
    """What each candidate probability threshold would have selected.

    This is the direct decision aid for ``etf_arm_min_prob``: at each candidate,
    how many forecasts clear it, how often they were right, and whether that is
    distinguishable from chance. ``min_confidence`` applies the other half of
    the gate so the two knobs can be read together.
    """
    floor = CONFIDENCE_RANKS.get(min_confidence.upper(), 0)
    eligible = [f for f in forecasts if f.resolved
                and CONFIDENCE_RANKS.get(f.confidence.upper(), 0) >= floor]
    rows = []
    for threshold in sorted(thresholds):
        group = [f for f in eligible if f.probability >= threshold]
        n = len(group)
        hits = sum(int(f.actual_outcome) for f in group)
        realized = hits / n if n else 0.0
        lo = wilson_interval(hits, n, z)[0] if n else 0.0
        rows.append(ThresholdRow(
            threshold=threshold, n=n, hits=hits, realized=realized, lo=lo,
            samples_needed=samples_for_reliable_edge(realized, z=z) if n else None))
    return rows


def suggested_thresholds(forecasts, *, width: float = 0.02) -> list[float]:
    """Candidate thresholds spanning the range the arms actually produce.

    Anything above the observed maximum is not a threshold, it is an off
    switch — which is precisely what 0.62 has been against a 0.56 ceiling.
    """
    probabilities = [f.probability for f in forecasts]
    if not probabilities:
        return []
    lowest, highest = min(probabilities), max(probabilities)
    steps = max(1, int((highest - lowest) / width) + 1)
    return [round(lowest + i * width, 4) for i in range(steps)]


@dataclass(frozen=True)
class TradingClearance:
    """Whether an arm has earned the right to place a trade."""
    cleared: bool
    reason: str
    resolved: int
    brier_edge: float
    brier_edge_lo: float
    max_conviction: float


def clearance(forecasts, *, min_resolved: int = 100,
              z: float = DEFAULT_Z) -> TradingClearance:
    """Gate trading on demonstrated forecast skill, not on elapsed time.

    The previous book traded first and measured later, and the measuring said
    it had been paying ~30bps to act on ~6bps of edge. This inverts that: an
    arm forecasts freely — forecasts cost nothing and resolve in five sessions,
    so ~125 a week accumulate — and may not trade until those forecasts have
    beaten the benchmark with a lower bound clear of zero.

    The statistic is the Brier EDGE against each forecast's own reference: mean
    of ``(reference - outcome)^2 - (probability - outcome)^2``, positive when
    the forecast was closer to the truth than the benchmark was. Its lower
    bound must clear zero, so a run of luck does not open the gate.

    ``max_conviction`` is reported because the economic gate needs it: an arm
    whose forecasts never leave 0.02 of a coin flip cannot trade anything in
    this universe whatever its calibration, and that is a different failure
    from being wrong.
    """
    # Only forecasts carrying the benchmark they were scored against can
    # contribute. Scoring against a coin overstates edge; scoring against
    # nothing is not scoring. Both fail CLOSED here, which is what the gate is
    # for — an arm that cannot demonstrate edge does not get to trade.
    resolved = [f for f in forecasts if f.resolved and f.reference is not None]
    unscoreable = sum(1 for f in forecasts if f.resolved and f.reference is None)
    conviction = max((abs(f.probability - 0.5) for f in forecasts), default=0.0)
    if unscoreable and not resolved:
        return TradingClearance(
            False,
            f"{unscoreable} resolved forecast(s) carry no benchmark to score "
            "against; record horizon_up_rate per forecast to open this gate",
            0, 0.0, float("-inf"), conviction)
    # Two is the floor regardless of configuration: sample variance is
    # undefined below it, so a single lucky forecast could otherwise open the
    # gate (or divide by zero trying).
    if len(resolved) < max(2, min_resolved):
        return TradingClearance(
            False, f"{len(resolved)}/{max(2, min_resolved)} resolved forecasts",
            len(resolved), 0.0, float("-inf"), conviction)
    edges = [((f.reference - f.actual_outcome) ** 2
              - (f.probability - f.actual_outcome) ** 2) for f in resolved]
    mean = sum(edges) / len(edges)
    variance = sum((e - mean) ** 2 for e in edges) / (len(edges) - 1)
    lo = mean - z * math.sqrt(variance / len(edges))
    if lo <= 0:
        return TradingClearance(
            False, f"Brier edge lower bound {lo:+.5f} does not clear zero",
            len(resolved), mean, lo, conviction)
    return TradingClearance(
        True, f"beat the benchmark on {len(resolved)} forecasts",
        len(resolved), mean, lo, conviction)
