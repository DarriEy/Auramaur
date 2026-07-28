"""Does normalized_momentum predict anything on this universe?

The exit study showed the parameter surface is noise, which points at the entry
signal or the premise. Searching entry thresholds would repeat the same
mistake, so this measures the signal's INFORMATION CONTENT directly: no
selection, no best-of-N, nothing to overfit.

Two things make the measurement honest:

**Excess, not raw, returns.** The universe drifted up over 2021-2026, so a
long-only rule's returns are mostly beta. Every forward return here is measured
against the equal-weight mean of the instruments priced on the SAME session, so
"momentum works" cannot be "stocks went up" in disguise. This is the same
base-rate correction the ETF Brier bar needed.

**Non-overlapping statistics.** A forward return over h sessions, sampled every
session, produces h-fold overlapping windows whose autocorrelation makes any
naive t-stat wildly overconfident. The information coefficient is computed
cross-sectionally per session (which handles same-day correlation), and its
t-statistic uses only every h-th session, so the sampled windows do not
overlap. Both sample sizes are reported.

Signals use exactly ``SIGNAL_WINDOW`` trailing closes — what the live book
fetches — for the reason documented there.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from auramaur.backtest.ibkr_replay import SIGNAL_WINDOW
from auramaur.risk.ibkr_math import normalized_momentum


@dataclass(frozen=True)
class Observation:
    date: str
    key: str
    momentum: float
    forward_return: float
    excess_return: float


@dataclass(frozen=True)
class Bucket:
    label: str
    n: int
    mean_momentum: float
    mean_excess: float
    lcb: float
    hit_rate: float


def _mean(values) -> float:
    return sum(values) / len(values) if values else 0.0


def _lcb(values, z: float = 1.96) -> float:
    n = len(values)
    if n < 2:
        return float("-inf")
    mean = _mean(values)
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean - z * math.sqrt(variance / n)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = _mean(rx), _mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else None


def observations(bars_by_key: dict[str, list[tuple[str, float]]], *,
                 horizon: int, window: int = SIGNAL_WINDOW) -> list[Observation]:
    """(momentum at t, return t -> t+horizon, excess over the same session)."""
    dates = sorted({d for bars in bars_by_key.values() for d, _ in bars})
    closes = {k: {d: c for d, c in v} for k, v in bars_by_key.items()}
    index = {d: i for i, d in enumerate(dates)}
    out: list[Observation] = []
    for key, series in bars_by_key.items():
        ordered = sorted(series)
        prices = [c for _, c in ordered]
        days = [d for d, _ in ordered]
        for i in range(window, len(ordered) - horizon):
            # Strictly trailing: prices[:i] excludes session i itself.
            momentum = normalized_momentum(prices[i - window:i])
            if momentum is None:
                continue
            entry, exit_ = prices[i], prices[i + horizon]
            if entry <= 0 or exit_ <= 0:
                continue
            out.append(Observation(days[i], key, momentum, exit_ / entry - 1, 0.0))
    # Excess vs the equal-weight mean of everything priced that session.
    by_date: dict[str, list[Observation]] = {}
    for obs in out:
        by_date.setdefault(obs.date, []).append(obs)
    adjusted: list[Observation] = []
    for day, group in by_date.items():
        bar = _mean([o.forward_return for o in group])
        for o in group:
            adjusted.append(Observation(o.date, o.key, o.momentum,
                                        o.forward_return,
                                        o.forward_return - bar))
    _ = (closes, index)  # kept for readability of the mapping above
    return sorted(adjusted, key=lambda o: (o.date, o.key))


def decile_buckets(obs: list[Observation], buckets: int = 5) -> list[Bucket]:
    """Mean excess return by momentum bucket. Monotonicity is the signal."""
    if not obs:
        return []
    ordered = sorted(obs, key=lambda o: o.momentum)
    size = max(1, len(ordered) // buckets)
    out = []
    for b in range(buckets):
        lo = b * size
        hi = len(ordered) if b == buckets - 1 else (b + 1) * size
        group = ordered[lo:hi]
        if not group:
            continue
        excess = [o.excess_return for o in group]
        out.append(Bucket(
            label=f"Q{b + 1}", n=len(group),
            mean_momentum=_mean([o.momentum for o in group]),
            mean_excess=_mean(excess), lcb=_lcb(excess),
            hit_rate=sum(1 for e in excess if e > 0) / len(excess)))
    return out


def information_coefficient(obs: list[Observation], *, horizon: int):
    """Mean cross-sectional Spearman, with a non-overlapping t-statistic.

    Returns (mean_ic, t_stat, sessions_used, sessions_total). The t-stat uses
    every horizon-th session so the forward windows it samples do not overlap;
    computing it on all sessions would overstate significance h-fold.
    """
    by_date: dict[str, list[Observation]] = {}
    for o in obs:
        by_date.setdefault(o.date, []).append(o)
    per_session = []
    for day in sorted(by_date):
        group = by_date[day]
        ic = _spearman([o.momentum for o in group],
                       [o.excess_return for o in group])
        if ic is not None:
            per_session.append((day, ic))
    if not per_session:
        return (0.0, 0.0, 0, 0)
    sampled = [ic for i, (_, ic) in enumerate(per_session) if i % max(1, horizon) == 0]
    mean_ic = _mean([ic for _, ic in per_session])
    if len(sampled) < 2:
        return (mean_ic, 0.0, len(sampled), len(per_session))
    m = _mean(sampled)
    var = sum((v - m) ** 2 for v in sampled) / (len(sampled) - 1)
    se = math.sqrt(var / len(sampled)) if var > 0 else 0.0
    return (mean_ic, (m / se if se > 0 else 0.0), len(sampled), len(per_session))


def gate_contrast(obs: list[Observation], threshold: float):
    """What the DEPLOYED entry gate selects, versus what it rejects.

    Returns (qualifying, rejected) Buckets. If the gate carries information the
    qualifying bucket's mean excess return is positive and its lower bound
    clears zero; if the two are indistinguishable the threshold is decoration.
    """
    passing = [o for o in obs if o.momentum >= threshold]
    failing = [o for o in obs if o.momentum < threshold]

    def build(label, group):
        if not group:
            return Bucket(label, 0, 0.0, 0.0, float("-inf"), 0.0)
        excess = [o.excess_return for o in group]
        return Bucket(label, len(group),
                      _mean([o.momentum for o in group]), _mean(excess),
                      _lcb(excess),
                      sum(1 for e in excess if e > 0) / len(excess))

    return build(f"momentum >= {threshold:g}", passing), build(
        f"momentum < {threshold:g}", failing)
